let messages = [];
let currentDistillationReport = "";
let currentAgent2Feedback = "";
let currentRewrittenEssay = "";
let currentCouncilVerdict = "";
let currentHumanizedEssay = "";

// ─────────────────────────────────────────────────────────────────────────────
// STATUS BAR (persists through all pipeline states)
// ─────────────────────────────────────────────────────────────────────────────
function setStatus(msg) {
    const bar = document.getElementById('pipeline-status-bar');
    const text = document.getElementById('pipeline-status-text');
    if (!bar || !text) return;
    if (!msg) { bar.classList.add('hidden'); return; }
    bar.classList.remove('hidden');
    text.innerText = msg;
}

// ─────────────────────────────────────────────────────────────────────────────
// INITIALIZE RAG (Agent 1 → handoff to Agent 2)
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('start-btn').addEventListener('click', async () => {
    const essay = document.getElementById('essay-input').value.trim();
    if (!essay) { console.warn('[start-btn] Essay textarea is empty — aborting.'); return; }

    console.log('[start-btn] Essay submitted, length:', essay.length, 'chars');
    document.getElementById('start-btn').disabled = true;
    // Lock essay so it can't be changed mid-pipeline
    const essayTA = document.getElementById('essay-input');
    essayTA.disabled = true;
    essayTA.style.opacity = '0.5';
    essayTA.style.cursor  = 'not-allowed';

    setStatus('Initializing connection to ChromaDB...');

    try {
        console.log('[Agent1] ▶ POST /api/start_chat ...');
        const response = await fetch('/api/start_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_essay: essay })
        });
        console.log('[Agent1] HTTP response status:', response.status, response.statusText);

        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer    = "";
        let chunkCount = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) { console.log('[Agent1] SSE stream closed (done=true). Total raw chunks read:', chunkCount); break; }

            chunkCount++;
            const rawText = decoder.decode(value, { stream: true });
            if (chunkCount <= 5 || chunkCount % 50 === 0) {
                console.log(`[Agent1] Raw SSE chunk #${chunkCount} (${rawText.length} bytes):`, JSON.stringify(rawText.substring(0, 200)));
            }
            buffer += rawText;
            let lines  = buffer.split('\n');
            buffer     = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue; // skip blank lines
                if (!line.startsWith('data: ')) {
                    console.log('[Agent1] Non-data SSE line skipped:', JSON.stringify(line));
                    continue;
                }
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') {
                    console.log('[Agent1] [DONE] signal received — stream officially ended.');
                    continue;
                }
                try {
                    const parsed = JSON.parse(dataStr);
                    console.log('[Agent1] SSE event parsed:', Object.keys(parsed));
                    if (parsed.status)  {
                        console.log('[Agent1]   → status:', parsed.status);
                        setStatus(parsed.status);
                    }
                    if (parsed.error)   {
                        console.error('[Agent1]   ❌ error from server:', parsed.error);
                        setStatus('ERROR: ' + parsed.error);
                        resetSetup();
                        return;
                    }
                    if (parsed.distillation_report) {
                        currentDistillationReport = parsed.distillation_report;
                        console.log('[Agent1]   → distillation_report received:', parsed.distillation_report.length, 'chars');
                    }
                    if (parsed.system_prompt && parsed.user_prompt) {
                        console.log('[Agent1]   → HANDOFF PACKET received!');
                        console.log('[Agent1]     system_prompt length:', parsed.system_prompt.length);
                        console.log('[Agent1]     user_prompt length:', parsed.user_prompt.length);
                        messages.push({ role: 'system', content: parsed.system_prompt });
                        messages.push({ role: 'user',   content: parsed.user_prompt   });
                        console.log('[Agent1]   → messages array now has', messages.length, 'entries');

                        // Reveal chat view
                        document.getElementById('setup-screen').classList.add('hidden');
                        document.getElementById('chat-history').classList.remove('hidden');
                        document.getElementById('input-container').classList.remove('hidden');
                        if (currentDistillationReport) {
                            document.getElementById('show-agent1-btn').classList.remove('hidden');
                        }

                        setStatus('Agent 1 complete ✔  |  Agent 2: Counselor is analyzing your essay...');
                        console.log('[Agent1] ── Calling sendChat() to start Agent 2 now...');
                        await sendChat();
                        console.log('[Agent1] ── sendChat() returned.');
                    }
                } catch (e) {
                    console.error('[Agent1] JSON parse error on SSE line:', JSON.stringify(line), '\nError:', e);
                }
            }
        }
        console.log('[Agent1] ✅ /api/start_chat stream fully consumed.');
    } catch (e) {
        console.error('[start-btn] ❌ Fetch error:', e);
        setStatus('Server error — make sure app.py is running via uvicorn.');
        resetSetup();
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// SESSION MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('new-session-btn').addEventListener('click', () => {
    messages               = [];
    currentDistillationReport = "";
    currentAgent2Feedback  = "";
    currentRewrittenEssay  = "";
    currentCouncilVerdict  = "";
    currentHumanizedEssay  = "";

    document.getElementById('chat-history').innerHTML = '';
    document.getElementById('chat-history').classList.add('hidden');
    document.getElementById('input-container').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');

    const ta = document.getElementById('essay-input');
    ta.value    = '';
    ta.disabled = false;
    ta.style.opacity = '1';
    ta.style.cursor  = 'auto';

    document.getElementById('chat-input').value = '';
    document.getElementById('show-agent1-btn').classList.add('hidden');
    document.getElementById('token-counter').innerText = '0/8192';
    setStatus('');
    resetSetup();
});

function resetSetup() {
    document.getElementById('start-btn').disabled = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// CHAT INPUT
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('send-btn').addEventListener('click', async () => {
    const userInput = document.getElementById('chat-input').value.trim();
    if (!userInput) return;
    document.getElementById('chat-input').value = '';
    document.getElementById('chat-input').style.height = '50px';
    appendMessage('user', userInput);
    messages.push({ role: 'user', content: userInput });
    await sendChat();
});

document.getElementById('chat-input').addEventListener("input", function() {
    this.style.height = '50px';
    this.style.height = (this.scrollHeight) + "px";
});

document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('send-btn').click();
    }
});

const scrollContainer = document.getElementById('content-container');

// ─────────────────────────────────────────────────────────────────────────────
// MESSAGE RENDERING  – Tailwind War Room aesthetic
// Returns the inner content div that can be streamed into
// ─────────────────────────────────────────────────────────────────────────────
function appendMessage(role, text) {
    const history = document.getElementById('chat-history');

    if (role === 'user') {
        const wrapper = document.createElement('div');
        wrapper.className = 'flex flex-col items-end gap-2';
        wrapper.innerHTML = `
            <span class="text-xs font-sans text-[#d0c5af]/60 uppercase tracking-widest mr-4">Your Draft</span>
            <div class="bg-[#0e0e13] p-8 rounded-lg border border-[#4d4635]/10 text-[#e4e1e9] text-base leading-relaxed font-body max-w-3xl shadow-inner relative">
                <div class="absolute top-0 left-0 w-1 h-full bg-[#35343a] rounded-l-lg"></div>
                <div class="msg-content">${escapeHtml(text)}</div>
            </div>`;
        history.appendChild(wrapper);
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
        return wrapper.querySelector('.msg-content');
    }

    // Counselor / Agent bubble
    const wrapper = document.createElement('div');
    wrapper.className = 'flex flex-col items-start gap-3';
    wrapper.innerHTML = `
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-[#2a292f] flex items-center justify-center border border-[#97b0ff]/30 relative">
                <span class="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-[#bfcdff]" style="box-shadow:0 0 8px #bfcdff"></span>
                <span class="material-symbols-outlined text-[#bfcdff] text-sm">psychology</span>
            </div>
            <span class="text-sm font-sans font-bold text-[#bfcdff] tracking-wide uppercase">The Counselor</span>
        </div>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'bg-[#1b1b20] p-8 rounded-lg max-w-3xl border border-[#4d4635]/10 text-[#d0c5af] leading-relaxed msg-content';
    contentDiv.style.boxShadow = '0 20px 40px rgba(0,0,0,0.4)';
    if (text) contentDiv.innerHTML = marked.parse(text);

    wrapper.appendChild(contentDiv);
    history.appendChild(wrapper);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    return contentDiv;
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g,  "&amp;")
        .replace(/</g,  "&lt;")
        .replace(/>/g,  "&gt;")
        .replace(/"/g,  "&quot;")
        .replace(/'/g,  "&#039;")
        .replace(/\n/g, "<br>");
}

function appendRewriterMessage(label, iconName, color, badgeColor) {
    const history = document.getElementById('chat-history');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex flex-col items-start gap-3';
    wrapper.innerHTML = `
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-[#2a292f] flex items-center justify-center border" style="border-color:${color}44">
                <span class="absolute -mt-5 -mr-5 w-2.5 h-2.5 rounded-full" style="background:${color};box-shadow:0 0 8px ${color}"></span>
                <span class="material-symbols-outlined text-sm" style="color:${color}">${iconName}</span>
            </div>
            <span class="text-sm font-sans font-bold tracking-wide uppercase" style="color:${color}">${label}</span>
            <span class="font-mono text-[10px] text-[#d0c5af]/40 uppercase tracking-tighter ml-2 px-1.5 py-0.5 border border-[#4d4635]/30 rounded">${badgeColor}</span>
        </div>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'bg-[#1b1b20] p-8 rounded-lg max-w-3xl border text-[#d0c5af] leading-relaxed msg-content';
    contentDiv.style.borderColor = color + '1a';
    contentDiv.style.boxShadow   = '0 20px 40px rgba(0,0,0,0.4)';

    wrapper.appendChild(contentDiv);
    history.appendChild(wrapper);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    return contentDiv;
}

function renderMetrics(m, contentDiv) {
    document.getElementById('token-counter').innerText = `${m.total_tokens}/8192`;
    const metricsDiv = document.createElement('div');
    metricsDiv.className = 'flex gap-4 mt-6 font-mono text-[10px] uppercase tracking-widest text-[#d0c5af]/40';
    metricsDiv.innerHTML = `<span>${m.tok_s} tok/s</span><span>${m.total_tokens} tokens</span><span>${m.total_time}s</span>`;
    contentDiv.appendChild(metricsDiv);
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 2: COUNSELOR STREAM
// ─────────────────────────────────────────────────────────────────────────────
async function sendChat() {
    console.log('[Agent2] ▶ sendChat() called. messages array length:', messages.length);
    messages.forEach((m, i) => console.log(`[Agent2]   msg[${i}] role=${m.role} len=${m.content.length}`));

    const sendBtn    = document.getElementById('send-btn');
    const inputField = document.getElementById('chat-input');
    sendBtn.disabled    = true;
    inputField.disabled = true;

    // appendMessage returns the streaming target div
    const contentDiv = appendMessage('counselor', '');
    contentDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>The Counselor is analyzing your draft...</span>";

    try {
        console.log('[Agent2] POST /api/chat ...');
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: messages })
        });
        console.log('[Agent2] HTTP response status:', response.status, response.statusText);

        contentDiv.innerHTML = "";

        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullContent = "";
        let buffer      = "";
        let chunkCount  = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) { console.log('[Agent2] SSE stream closed. Total chunks:', chunkCount); break; }
            chunkCount++;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer    = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') {
                    console.log('[Agent2] [DONE] received.');
                    continue;
                }
                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.status) {
                        console.log('[Agent2] status:', parsed.status);
                        setStatus('Agent 2: ' + parsed.status);
                        if (fullContent === "") contentDiv.innerHTML = `<span class='text-[#d0c5af]/40 italic'>${parsed.status}</span>`;
                    } else if (parsed.content) {
                        if (fullContent === "" && contentDiv.innerHTML.includes("italic")) contentDiv.innerHTML = "";
                        fullContent += parsed.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.metrics) {
                        console.log('[Agent2] metrics received:', parsed.metrics);
                        renderMetrics(parsed.metrics, contentDiv);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.error) {
                        console.error('[Agent2] ❌ server error:', parsed.error);
                        contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">Error: ${parsed.error}</span>`;
                    }
                } catch (e) {
                    console.error('[Agent2] JSON parse error:', JSON.stringify(line), e);
                }
            }
        }

        console.log('[Agent2] ✅ Full response received. length:', fullContent.length, 'chars');
        messages.push({ role: 'assistant', content: fullContent });
        currentAgent2Feedback = fullContent;
        setStatus('Agent 2 complete ✔  |  Ask follow-up questions or click "Rewrite Full Essay"');

        // Spawn Rewrite + Fix buttons only on FIRST Agent 2 response (system + user + assistant = 3)
        console.log('[Agent2] messages.length after push:', messages.length, '— Action buttons show if === 3');
        if (messages.length === 3) {
            const btnContainer = document.createElement('div');
            btnContainer.className = 'mt-6 flex flex-wrap gap-3';

            // --- Rewrite button (Agent 3) ---
            const rewriteBtn = document.createElement('button');
            rewriteBtn.className = 'bg-[#2a292f] text-[#bfcdff] border border-[#97b0ff]/30 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:bg-[#35343a] transition-colors flex items-center gap-2';
            rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">edit_note</span> Rewrite Full Essay (Agent 3)';
            rewriteBtn.addEventListener('click', async () => {
                rewriteBtn.disabled = true;
                fixBtn.disabled = true;
                fixBtn.classList.add('opacity-40', 'cursor-not-allowed');
                rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> Agent 3 is rewriting...';
                setStatus('Agent 3: Generating full essay rewrite (~650 words)...');
                console.log('[Agent3] Rewrite button clicked.');
                await executeRewrite(fullContent, rewriteBtn);
            });

            // --- Fix button (Agent 3.5) ---
            const fixBtn = document.createElement('button');
            fixBtn.className = 'bg-[#1b1b20] text-[#f2ca50] border border-[#d4af37]/30 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:bg-[#2a292f] hover:border-[#f2ca50]/50 transition-colors flex items-center gap-2';
            fixBtn.innerHTML = '<span class="material-symbols-outlined text-base">build</span> Fix Mistakes (Agent 3.5)';
            fixBtn.addEventListener('click', async () => {
                fixBtn.disabled = true;
                rewriteBtn.disabled = true;
                rewriteBtn.classList.add('opacity-40', 'cursor-not-allowed');
                fixBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> Agent 3.5 is fixing...';
                setStatus('Agent 3.5: Fixing flagged issues while preserving your voice...');
                console.log('[Agent3.5] Fix button clicked.');
                await executeFix(fullContent, fixBtn);
            });

            btnContainer.appendChild(rewriteBtn);
            btnContainer.appendChild(fixBtn);
            contentDiv.appendChild(btnContainer);
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }

    } catch (e) {
        console.error('[Agent2] ❌ Fetch error:', e);
        contentDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to counselor lost.</span>";
    }

    sendBtn.disabled    = false;
    inputField.disabled = false;
    inputField.focus();
    console.log('[Agent2] sendChat() complete.');
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 3: REWRITER STREAM
// ─────────────────────────────────────────────────────────────────────────────
async function executeRewrite(agent2Feedback, rewriteBtn) {
    const contentDiv = appendRewriterMessage('The Rewriter', 'auto_fix_high', '#f2ca50', 'Agent 3 · PC1');
    contentDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 3 is analyzing benchmark data and rewriting...</span>";

    const userEssay = document.getElementById('essay-input').value.trim();

    try {
        const response = await fetch('/api/rewrite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_essay: userEssay,
                distillation_report: currentDistillationReport,
                agent2_feedback: agent2Feedback
            })
        });

        contentDiv.innerHTML = "";

        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullContent = "";
        let buffer      = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer    = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') continue;
                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.status) {
                        setStatus('Agent 3: ' + parsed.status);
                        if (fullContent === "") contentDiv.innerHTML = `<span class='text-[#d0c5af]/40 italic'>${parsed.status}</span>`;
                    } else if (parsed.content) {
                        if (fullContent === "" && contentDiv.innerHTML.includes("italic")) contentDiv.innerHTML = "";
                        fullContent += parsed.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.metrics) {
                        renderMetrics(parsed.metrics, contentDiv);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.error) {
                        contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">Error: ${parsed.error}</span>`;
                    }
                } catch (e) {}
            }
        }

        messages.push({ role: 'assistant', content: fullContent });
        currentRewrittenEssay = fullContent;
        rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">check_circle</span> Rewrite Complete ✓';
        rewriteBtn.classList.add('opacity-50', 'cursor-not-allowed');
        setStatus('Agent 3 complete ✔  |  Click "Consult the Council" to critique on PC2');

        // Council button
        const councilBtn = document.createElement('button');
        councilBtn.className = 'mt-6 bg-[#131318] text-[#f2ca50] border border-[#d4af37]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#f2ca50] transition-all flex items-center gap-2';
        councilBtn.innerHTML = '<span class="material-symbols-outlined text-base" style="font-variation-settings:\'FILL\' 1">gavel</span> ⚖️ Consult the Council';
        councilBtn.addEventListener('click', async () => {
            councilBtn.disabled = true;
            councilBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> The Council is deliberating...';
            councilBtn.style.opacity = '0.6';
            setStatus('Council: Sending to PC2 critic server at 172.16.0.158:5001...');
            await consultCouncil(councilBtn);
        });
        contentDiv.appendChild(councilBtn);

    } catch (e) {
        contentDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 3 lost.</span>";
        rewriteBtn.disabled = false;
        rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">edit_note</span> Rewrite Full Essay (Agent 3)';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 3.5: FIXER STREAM (preserves voice, fixes flagged issues only)
// ─────────────────────────────────────────────────────────────────────────────
async function executeFix(agent2Feedback, fixBtn) {
    const contentDiv = appendRewriterMessage('The Fixer', 'build', '#f2ca50', 'Agent 3.5 · Targeted');
    contentDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 3.5 is surgically fixing flagged issues while preserving your voice...</span>";

    const userEssay = document.getElementById('essay-input').value.trim();

    try {
        const response = await fetch('/api/fix', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_essay: userEssay,
                distillation_report: currentDistillationReport,
                agent2_feedback: agent2Feedback
            })
        });

        contentDiv.innerHTML = "";

        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullContent = "";
        let buffer      = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer    = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') continue;
                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.status) {
                        setStatus('Agent 3.5: ' + parsed.status);
                        if (fullContent === "") contentDiv.innerHTML = `<span class='text-[#d0c5af]/40 italic'>${parsed.status}</span>`;
                    } else if (parsed.content) {
                        if (fullContent === "" && contentDiv.innerHTML.includes("italic")) contentDiv.innerHTML = "";
                        fullContent += parsed.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.metrics) {
                        renderMetrics(parsed.metrics, contentDiv);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.error) {
                        contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">Error: ${parsed.error}</span>`;
                    }
                } catch (e) {}
            }
        }

        messages.push({ role: 'assistant', content: fullContent });
        currentRewrittenEssay = fullContent;
        fixBtn.innerHTML = '<span class="material-symbols-outlined text-base">check_circle</span> Fix Complete ✓';
        fixBtn.classList.add('opacity-50', 'cursor-not-allowed');
        setStatus('Agent 3.5 complete ✔  |  Click "Consult the Council" to critique on PC2');

        // Council button
        const councilBtn = document.createElement('button');
        councilBtn.className = 'mt-6 bg-[#131318] text-[#f2ca50] border border-[#d4af37]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#f2ca50] transition-all flex items-center gap-2';
        councilBtn.innerHTML = '<span class="material-symbols-outlined text-base" style="font-variation-settings:\'FILL\' 1">gavel</span> ⚖️ Consult the Council';
        councilBtn.addEventListener('click', async () => {
            councilBtn.disabled = true;
            councilBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> The Council is deliberating...';
            councilBtn.style.opacity = '0.6';
            setStatus('Council: Sending to PC2 critic server at 172.16.0.158:5001...');
            await consultCouncil(councilBtn);
        });
        contentDiv.appendChild(councilBtn);

    } catch (e) {
        contentDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 3.5 lost.</span>";
        fixBtn.disabled = false;
        fixBtn.innerHTML = '<span class="material-symbols-outlined text-base">build</span> Fix Mistakes (Agent 3.5)';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 1 MODAL
// ─────────────────────────────────────────────────────────────────────────────
const agent1Btn          = document.getElementById('show-agent1-btn');
const agent1Modal        = document.getElementById('agent1-modal');
const closeAgent1Modal   = document.getElementById('close-agent1-modal');
const agent1ReportContent = document.getElementById('agent1-report-content');

if (agent1Btn && agent1Modal && closeAgent1Modal) {
    agent1Btn.addEventListener('click', () => {
        agent1ReportContent.innerHTML = currentDistillationReport
            ? marked.parse(currentDistillationReport)
            : "<span class='text-[#d0c5af]/40 italic'>No report generated yet.</span>";
        agent1Modal.classList.remove('hidden');
    });
    closeAgent1Modal.addEventListener('click', () => agent1Modal.classList.add('hidden'));
    window.addEventListener('click', (e) => { if (e.target === agent1Modal) agent1Modal.classList.add('hidden'); });
}

// ─────────────────────────────────────────────────────────────────────────────
// COUNCIL: CROSS-PC CRITIQUE (PC2)
// ─────────────────────────────────────────────────────────────────────────────
async function consultCouncil(btn) {
    const userEssay = document.getElementById('essay-input').value.trim();
    try {
        const response = await fetch('http://172.16.0.158:5001/critique', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                original_essay:   userEssay,
                rewritten_essay:  currentRewrittenEssay,
                agent1_report:    currentDistillationReport,
                agent2_feedback:  currentAgent2Feedback
            })
        });
        if (!response.ok) throw new Error(`PC2 HTTP error: ${response.status}`);
        const jsonVerdict     = await response.json();
        currentCouncilVerdict = JSON.stringify(jsonVerdict, null, 2);
        renderCouncilVerdict(jsonVerdict);
        setStatus('Council verdict received ✔  |  Click "Apply the Fix" to run Agent 4');
        btn.style.display = 'none';
    } catch (e) {
        btn.innerHTML  = '<span class="material-symbols-outlined text-base">error</span> Council Unreachable';
        btn.disabled   = false;
        btn.style.opacity = '1';
        setStatus('Council unreachable — check PC2 server at 172.16.0.158:5001');
        console.error("Council Fetch Error:", e);
    }
}

function renderCouncilVerdict(verdict) {
    const history    = document.getElementById('chat-history');
    const verdictStr = (verdict.verdict || verdict.Decision || "FAIL").toUpperCase();
    const isPass     = verdictStr === "PASS";
    const color      = isPass ? '#4ade80' : '#ffb4ab';
    const subTitle   = isPass ? 'All Systems Clear' : 'Red Flags Detected';

    const wrapper = document.createElement('div');
    wrapper.className = 'flex flex-col items-center gap-6 mt-4 mb-4';

    let flagsHtml = '';
    for (const key in verdict) {
        const lk = key.toLowerCase();
        if (lk === 'verdict' || lk === 'decision' || lk === 'note') continue;
        const val = verdict[key];
        if (typeof val === 'object' && val !== null) {
            for (const subKey in val) {
                const items = Array.isArray(val[subKey])
                    ? val[subKey].map(f => `<li class="flex items-start gap-2"><span class="material-symbols-outlined text-sm mt-0.5" style="color:${color}">flag</span><span>${f}</span></li>`).join('')
                    : `<p>${val[subKey]}</p>`;
                flagsHtml += `
                <details class="council-flag bg-[#151518] border border-[#4d4635]/30 rounded overflow-hidden">
                    <summary class="px-4 py-3 font-sans font-bold text-sm uppercase tracking-wider bg-[#1f1f25]" style="color:${color}">${subKey}</summary>
                    <div class="p-4 text-sm text-[#d0c5af] leading-relaxed border-t border-[#4d4635]/20">${Array.isArray(val[subKey]) ? '<ul class="space-y-2">' + items + '</ul>' : items}</div>
                </details>`;
            }
        } else {
            flagsHtml += `
            <details class="council-flag bg-[#151518] border border-[#4d4635]/30 rounded overflow-hidden">
                <summary class="px-4 py-3 font-sans font-bold text-sm uppercase tracking-wider bg-[#1f1f25]" style="color:${color}">${key}</summary>
                <div class="p-4 text-sm text-[#d0c5af] leading-relaxed border-t border-[#4d4635]/20">${val}</div>
            </details>`;
        }
    }

    const noteHtml = (verdict.note || verdict.Note)
        ? `<div class="mt-6 font-bold text-[#e4e1e9] bg-[#1f1f25] p-4 border-l-4 rounded text-sm leading-relaxed" style="border-color:${color}">${verdict.note || verdict.Note}</div>`
        : '';

    wrapper.innerHTML = `
        <div class="flex items-center gap-4 pb-2 w-full max-w-lg justify-center" style="border-bottom:1px solid ${color}44">
            <span class="material-symbols-outlined text-2xl" style="color:${color};font-variation-settings:'FILL' 1">gavel</span>
            <span class="text-xl font-serif font-bold tracking-widest uppercase" style="color:${color}">The Council Decree</span>
            <span class="material-symbols-outlined text-2xl" style="color:${color};font-variation-settings:'FILL' 1">gavel</span>
        </div>
        <div class="bg-[#2a292f] p-10 rounded-lg max-w-3xl w-full ambient-shadow relative overflow-hidden" style="border:1px solid ${color}22">
            <div class="flex flex-col items-center text-center relative z-10">
                <span class="text-7xl leading-none font-serif font-bold mb-2" style="color:${color}">${verdictStr}</span>
                <span class="text-sm font-sans uppercase tracking-[0.2em] font-bold mb-8" style="color:${color}99">${subTitle}</span>
                <div class="text-left w-full space-y-3">${flagsHtml}${noteHtml}</div>
            </div>
        </div>
        <button id="surgeon-btn" class="text-white px-8 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:opacity-90 transition-opacity shadow-lg flex items-center gap-2" style="background:#00796b">
            <span class="material-symbols-outlined text-base">healing</span> 🔧 Apply the Fix (Agent 4)
        </button>`;

    history.appendChild(wrapper);

    document.getElementById('surgeon-btn').addEventListener('click', async (e) => {
        e.target.disabled = true;
        e.target.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> Agent 4 is operating...';
        setStatus('Agent 4: Surgical editor applying targeted fixes...');
        await applySurgeonFix();
    });

    scrollContainer.scrollTop = scrollContainer.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 4: SURGEON STREAM
// ─────────────────────────────────────────────────────────────────────────────
async function applySurgeonFix() {
    const contentDiv = appendRewriterMessage('✅ Final Essay — Council Approved', 'healing', '#4ade80', 'Agent 4 · PC1');
    contentDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 4 (Surgeon) is applying targeted fixes...</span>";

    // Add copy button to the header
    const header = contentDiv.previousElementSibling;
    if (header) {
        const copyBtn = document.createElement('button');
        copyBtn.id        = 'copy-final-btn';
        copyBtn.className = 'hidden ml-4 font-mono text-[10px] text-[#d0c5af]/60 uppercase tracking-widest border border-[#4d4635]/30 px-3 py-1 rounded hover:text-[#f2ca50] hover:border-[#f2ca50]/30 transition-colors';
        copyBtn.innerText = 'Copy';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(contentDiv.innerText);
            copyBtn.innerText = "Copied!";
            setTimeout(() => copyBtn.innerText = "Copy", 2000);
        });
        header.appendChild(copyBtn);
    }
    try {
        console.log('[Agent4] POST /api/agent4...');
        console.log('[Agent4]   verdict_json len:', currentCouncilVerdict.length);
        console.log('[Agent4]   agent1_report len:', currentDistillationReport.length);
        console.log('[Agent4]   agent2_feedback len:', currentAgent2Feedback.length);
        console.log('[Agent4]   rewritten_essay len:', currentRewrittenEssay.length);

        const response = await fetch('/api/agent4', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                verdict_json:    currentCouncilVerdict,
                agent1_report:   currentDistillationReport,
                agent2_feedback: currentAgent2Feedback,
                rewritten_essay: currentRewrittenEssay
            })
        });
        console.log('[Agent4] HTTP response:', response.status, response.statusText);

        contentDiv.innerHTML = "";

        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullContent = "";
        let buffer      = "";
        let chunkCount  = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) { console.log('[Agent4] SSE stream closed. chunks:', chunkCount); break; }
            chunkCount++;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer    = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') { console.log('[Agent4] [DONE] received.'); continue; }
                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.content) {
                        fullContent += parsed.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.metrics) {
                        console.log('[Agent4] metrics:', parsed.metrics);
                        renderMetrics(parsed.metrics, contentDiv);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.error) {
                        console.error('[Agent4] server error:', parsed.error);
                        contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">${parsed.error}</span>`;
                    }
                } catch (e) { console.error('[Agent4] parse error:', e, line); }
            }
        }

        console.log('[Agent4] Done. content length:', fullContent.length);
        setStatus('Agent 4 complete ✔  |  Your council-approved essay is ready to submit');
        const copyBtn = document.getElementById('copy-final-btn');
        if (copyBtn) copyBtn.classList.remove('hidden');

    } catch (e) {
        console.error('[Agent4] fetch error:', e);
        contentDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 4 Surgeon lost.</span>";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB SWITCHING LOGIC (3 tabs: College Essay, Humanizer, Essay Helper)
// ─────────────────────────────────────────────────────────────────────────────
const tabCouncil = document.getElementById('tab-council');
const tabHumanizer = document.getElementById('tab-humanizer');
const tabEssayHelper = document.getElementById('tab-essayhelper');
const councilSection = document.getElementById('council-section');
const humanizerSection = document.getElementById('humanizer-section');
const essayhelperSection = document.getElementById('essayhelper-section');
const inputContainer = document.getElementById('input-container');

const TAB_ACTIVE_GOLD = 'text-[#f2ca50] font-bold border-b-2 border-[#f2ca50] pb-1 font-serif tracking-tight opacity-80 cursor-pointer';
const TAB_INACTIVE = 'text-[#f2ca50]/60 font-medium font-serif tracking-tight hover:text-[#f2ca50] transition-all duration-300 cursor-pointer';
const TAB_ACTIVE_PURPLE = 'text-[#a78bfa] font-bold border-b-2 border-[#a78bfa] pb-1 font-serif tracking-tight opacity-80 cursor-pointer';
const TAB_ACTIVE_GREEN = 'text-[#4ade80] font-bold border-b-2 border-[#4ade80] pb-1 font-serif tracking-tight opacity-80 cursor-pointer';

function activateTab(activeTab) {
    // Reset all tabs to inactive
    tabCouncil.className = TAB_INACTIVE;
    tabHumanizer.className = TAB_INACTIVE;
    if (tabEssayHelper) tabEssayHelper.className = TAB_INACTIVE;

    // Hide all sections
    councilSection.classList.add('hidden');
    humanizerSection.classList.add('hidden');
    if (essayhelperSection) essayhelperSection.classList.add('hidden');
    inputContainer.classList.add('hidden');

    if (activeTab === 'council') {
        tabCouncil.className = TAB_ACTIVE_GOLD;
        councilSection.classList.remove('hidden');
        if (document.getElementById('setup-screen').classList.contains('hidden')) {
            inputContainer.classList.remove('hidden');
        }
    } else if (activeTab === 'humanizer') {
        tabHumanizer.className = TAB_ACTIVE_PURPLE;
        humanizerSection.classList.remove('hidden');
    } else if (activeTab === 'essayhelper') {
        if (tabEssayHelper) tabEssayHelper.className = TAB_ACTIVE_GREEN;
        if (essayhelperSection) essayhelperSection.classList.remove('hidden');
    }
}

if (tabCouncil) tabCouncil.addEventListener('click', () => activateTab('council'));
if (tabHumanizer) tabHumanizer.addEventListener('click', () => activateTab('humanizer'));
if (tabEssayHelper) tabEssayHelper.addEventListener('click', () => activateTab('essayhelper'));

// ─────────────────────────────────────────────────────────────────────────────
// AGENT 5: THE HUMANIZER — standalone RAG anti-AI rewrite
// ─────────────────────────────────────────────────────────────────────────────
const startHumanizerBtn = document.getElementById('start-humanizer-btn');
if (startHumanizerBtn) {
    startHumanizerBtn.addEventListener('click', async () => {
        const essay = document.getElementById('humanizer-input').value.trim();
        if (!essay) return;

        startHumanizerBtn.disabled = true;
        
        document.getElementById('humanizer-setup').classList.add('hidden');
        document.getElementById('humanizer-chat-history').classList.remove('hidden');
        document.getElementById('humanizer-chat-history').innerHTML = '';

        await executeStandaloneHumanizer(essay);
    });
}

async function executeStandaloneHumanizer(essayText) {
    const history = document.getElementById('humanizer-chat-history');
    
    // User message bubble (Original Essay)
    const userWrapper = document.createElement('div');
    userWrapper.className = 'flex flex-col items-end gap-2 mb-8 mt-4';
    userWrapper.innerHTML = `
        <span class="text-xs font-sans text-[#a78bfa]/60 uppercase tracking-widest mr-4">Original Draft</span>
        <div class="bg-[#0e0e13] p-8 rounded-lg border border-[#a78bfa]/20 text-[#e4e1e9] text-base leading-relaxed font-body max-w-3xl shadow-inner relative">
            <div class="absolute top-0 left-0 w-1 h-full bg-[#a78bfa]/40 rounded-l-lg"></div>
            <div class="whitespace-pre-wrap">${escapeHtml(essayText)}</div>
        </div>`;
    history.appendChild(userWrapper);

    // Humanizer agent bubble
    const agentWrapper = document.createElement('div');
    agentWrapper.className = 'flex flex-col items-start gap-3';
    agentWrapper.innerHTML = `
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-[#131318] flex items-center justify-center border border-[#a78bfa]/40 relative">
                <span class="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-[#a78bfa]" style="box-shadow:0 0 8px #a78bfa"></span>
                <span class="material-symbols-outlined text-sm text-[#a78bfa]">genetics</span>
            </div>
            <span class="text-sm font-sans font-bold text-[#a78bfa] tracking-wide uppercase">The Humanizer</span>
            <span class="font-mono text-[10px] text-[#d0c5af]/40 uppercase tracking-tighter ml-2 px-1.5 py-0.5 border border-[#4d4635]/30 rounded">Agent 5 · DNA</span>
            <button id="copy-humanized-btn" class="hidden ml-4 font-mono text-[10px] text-[#a78bfa]/60 uppercase tracking-widest border border-[#a78bfa]/40 px-3 py-1 rounded hover:text-white hover:border-white/50 transition-colors cursor-pointer">Copy</button>
        </div>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'bg-[#131318] p-8 rounded-lg max-w-3xl border border-[#a78bfa]/20 text-[#e4e1e9] leading-relaxed msg-content shadow-[0_20px_40px_rgba(0,0,0,0.6)] w-full';
    contentDiv.innerHTML = "<span class='text-[#a78bfa]/50 italic'>Agent 5 is scanning human DNA patterns and rewriting...</span>";

    agentWrapper.appendChild(contentDiv);
    history.appendChild(agentWrapper);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    
    const copyBtn = agentWrapper.querySelector('#copy-humanized-btn');
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(contentDiv.innerText);
        copyBtn.innerText = "Copied!";
        setTimeout(() => copyBtn.innerText = "Copy", 2000);
    });

    setStatus('Agent 5: Initializing Humanizer RAG...');

    try {
        const response = await fetch('/api/humanize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ essay_text: essayText })
        });
        
        contentDiv.innerHTML = "";
        
        const reader  = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullContent = "";
        let buffer      = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer    = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.substring(6);
                if (dataStr === '[DONE]') continue;
                try {
                    const parsed = JSON.parse(dataStr);
                    if (parsed.status) {
                        setStatus('Agent 5: ' + parsed.status);
                        if (fullContent === "") contentDiv.innerHTML = `<span class='text-[#a78bfa]/50 italic'>${parsed.status}</span>`;
                    } else if (parsed.content) {
                        if (fullContent === "" && contentDiv.innerHTML.includes("italic")) contentDiv.innerHTML = "";
                        fullContent += parsed.content;
                        contentDiv.innerHTML = marked.parse(fullContent);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.metrics) {
                        renderMetrics(parsed.metrics, contentDiv);
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    } else if (parsed.error) {
                        contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">Error: ${parsed.error}</span>`;
                    }
                } catch (e) {}
            }
        }
        
        currentHumanizedEssay = fullContent;
        setStatus('Agent 5 complete ✔  |  Your humanized essay is ready.');
        copyBtn.classList.remove('hidden');
        
        // Add "Humanize Another" button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'mt-6 bg-[#131318] text-[#a78bfa] border border-[#a78bfa]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#a78bfa] hover:bg-[#a78bfa]/10 transition-all flex items-center gap-2';
        resetBtn.innerHTML = '<span class="material-symbols-outlined text-base">refresh</span> Humanize Another Essay';
        resetBtn.addEventListener('click', () => {
             document.getElementById('humanizer-input').value = '';
             document.getElementById('start-humanizer-btn').disabled = false;
             document.getElementById('humanizer-chat-history').classList.add('hidden');
             document.getElementById('humanizer-setup').classList.remove('hidden');
             setStatus('');
        });
        contentDiv.appendChild(resetBtn);

    } catch (e) {
        console.error('Humanizer fetch error:', e);
        contentDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection lost.</span>";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ESSAY HELPER: AGENTS 6, 7, 8
// ─────────────────────────────────────────────────────────────────────────────

let ehExaminerReport = "";
let ehRewrittenEssay = "";

// Helper: create an agent bubble in the Essay Helper chat history
function ehAppendAgentBubble(label, iconName, color, badge) {
    const history = document.getElementById('eh-chat-history');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex flex-col items-start gap-3';
    wrapper.innerHTML = `
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-full bg-[#131318] flex items-center justify-center border" style="border-color:${color}44">
                <span class="absolute -mt-5 -mr-5 w-2.5 h-2.5 rounded-full" style="background:${color};box-shadow:0 0 8px ${color}"></span>
                <span class="material-symbols-outlined text-sm" style="color:${color}">${iconName}</span>
            </div>
            <span class="text-sm font-sans font-bold tracking-wide uppercase" style="color:${color}">${label}</span>
            <span class="font-mono text-[10px] text-[#d0c5af]/40 uppercase tracking-tighter ml-2 px-1.5 py-0.5 border border-[#4d4635]/30 rounded">${badge}</span>
        </div>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'bg-[#131318] p-8 rounded-lg max-w-3xl border text-[#d0c5af] leading-relaxed msg-content w-full';
    contentDiv.style.borderColor = color + '1a';
    contentDiv.style.boxShadow = '0 20px 40px rgba(0,0,0,0.4)';

    wrapper.appendChild(contentDiv);
    history.appendChild(wrapper);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    return { wrapper, contentDiv };
}

// Generic SSE stream reader for Essay Helper agents
async function ehStreamSSE(url, body, contentDiv, agentLabel) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    contentDiv.innerHTML = "";

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullContent = "";
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.trim()) continue;
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.substring(6);
            if (dataStr === '[DONE]') continue;
            try {
                const parsed = JSON.parse(dataStr);
                if (parsed.status) {
                    setStatus(agentLabel + ': ' + parsed.status);
                    if (fullContent === "") contentDiv.innerHTML = `<span class='text-[#d0c5af]/40 italic'>${parsed.status}</span>`;
                } else if (parsed.content) {
                    if (fullContent === "" && contentDiv.innerHTML.includes("italic")) contentDiv.innerHTML = "";
                    fullContent += parsed.content;
                    contentDiv.innerHTML = marked.parse(fullContent);
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                } else if (parsed.metrics) {
                    renderMetrics(parsed.metrics, contentDiv);
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                } else if (parsed.error) {
                    contentDiv.innerHTML += `<br><span class="text-[#ffb4ab]">Error: ${parsed.error}</span>`;
                }
            } catch (e) {}
        }
    }

    return fullContent;
}

// Analyze button click → Agent 6 → Rewrite button → Agent 7 → Coach button → Agent 8
const ehAnalyzeBtn = document.getElementById('eh-analyze-btn');
if (ehAnalyzeBtn) {
    ehAnalyzeBtn.addEventListener('click', async () => {
        const essay = document.getElementById('eh-essay-input').value.trim();
        if (!essay) return;

        const assignment = document.getElementById('eh-assignment').value.trim();
        const essayType = document.getElementById('eh-type').value;
        const gradeLevel = document.getElementById('eh-grade').value;
        const wordCount = parseInt(document.getElementById('eh-wordcount').value) || 500;

        ehAnalyzeBtn.disabled = true;
        ehExaminerReport = "";
        ehRewrittenEssay = "";

        // Switch to chat view
        document.getElementById('essayhelper-setup').classList.add('hidden');
        document.getElementById('eh-chat-history').classList.remove('hidden');
        document.getElementById('eh-chat-history').innerHTML = '';

        // Show user essay bubble
        const history = document.getElementById('eh-chat-history');
        const userWrapper = document.createElement('div');
        userWrapper.className = 'flex flex-col items-end gap-2 mb-4 mt-4';
        userWrapper.innerHTML = `
            <span class="text-xs font-sans text-[#4ade80]/60 uppercase tracking-widest mr-4">Your Essay · ${essayType} · ${gradeLevel}</span>
            <div class="bg-[#0e0e13] p-8 rounded-lg border border-[#4ade80]/20 text-[#e4e1e9] text-base leading-relaxed font-body max-w-3xl shadow-inner relative">
                <div class="absolute top-0 left-0 w-1 h-full bg-[#4ade80]/40 rounded-l-lg"></div>
                <div class="whitespace-pre-wrap">${escapeHtml(essay)}</div>
            </div>`;
        history.appendChild(userWrapper);

        // ── Agent 6: The Examiner ──────────────────────────────────
        setStatus('Agent 6: The Examiner is analyzing your essay...');
        const { contentDiv: examinerDiv } = ehAppendAgentBubble('The Examiner', 'rate_review', '#4ade80', 'Agent 6 · Analysis');
        examinerDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 6 is examining your essay...</span>";

        try {
            ehExaminerReport = await ehStreamSSE('/api/essay/analyze', {
                essay, assignment, essay_type: essayType, grade_level: gradeLevel, word_count_target: wordCount
            }, examinerDiv, 'Agent 6');

            setStatus('Agent 6 complete ✔  |  Click "Rewrite Essay" to run Agent 7');

            // Spawn Rewrite button
            const rewriteBtn = document.createElement('button');
            rewriteBtn.className = 'mt-6 bg-[#131318] text-[#4ade80] border border-[#4ade80]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#4ade80] hover:bg-[#4ade80]/10 transition-all flex items-center gap-2';
            rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">edit_note</span> Rewrite Essay (Agent 7)';
            rewriteBtn.addEventListener('click', async () => {
                rewriteBtn.disabled = true;
                rewriteBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> Agent 7 is rewriting...';
                rewriteBtn.style.opacity = '0.6';
                setStatus('Agent 7: The Rewriter is rebuilding your essay...');

                // ── Agent 7: The Rewriter ──────────────────────────────
                const { wrapper: rewriteWrapper, contentDiv: rewriteDiv } = ehAppendAgentBubble('The Rewriter', 'edit_note', '#38bdf8', 'Agent 7 · Rewrite');
                rewriteDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 7 is rewriting...</span>";

                try {
                    ehRewrittenEssay = await ehStreamSSE('/api/essay/rewrite', {
                        essay, assignment, essay_type: essayType, grade_level: gradeLevel,
                        word_count_target: wordCount, examiner_report: ehExaminerReport
                    }, rewriteDiv, 'Agent 7');

                    setStatus('Agent 7 complete ✔  |  Click "Get Coaching Notes" for Agent 8');

                    // Add copy button to rewrite header
                    const rewriteHeader = rewriteDiv.previousElementSibling;
                    if (rewriteHeader) {
                        const copyEssayBtn = document.createElement('button');
                        copyEssayBtn.className = 'hidden ml-4 font-mono text-[10px] text-[#38bdf8]/60 uppercase tracking-widest border border-[#38bdf8]/40 px-3 py-1 rounded hover:text-white hover:border-white/50 transition-colors cursor-pointer';
                        copyEssayBtn.id = 'eh-copy-essay-btn';
                        copyEssayBtn.innerText = 'Copy Essay';
                        copyEssayBtn.addEventListener('click', () => {
                            navigator.clipboard.writeText(rewriteDiv.innerText);
                            copyEssayBtn.innerText = "Copied!";
                            setTimeout(() => copyEssayBtn.innerText = "Copy Essay", 2000);
                        });
                        rewriteHeader.appendChild(copyEssayBtn);
                    }

                    // Spawn Coach button
                    const coachBtn = document.createElement('button');
                    coachBtn.className = 'mt-6 bg-[#131318] text-[#fbbf24] border border-[#fbbf24]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#fbbf24] hover:bg-[#fbbf24]/10 transition-all flex items-center gap-2';
                    coachBtn.innerHTML = '<span class="material-symbols-outlined text-base">school</span> Get Coaching Notes (Agent 8)';
                    coachBtn.addEventListener('click', async () => {
                        coachBtn.disabled = true;
                        coachBtn.innerHTML = '<span class="material-symbols-outlined text-base">progress_activity</span> Agent 8 is coaching...';
                        coachBtn.style.opacity = '0.6';
                        setStatus('Agent 8: The Coach is preparing your feedback...');

                        // ── Agent 8: The Coach ─────────────────────────────
                        const { contentDiv: coachDiv } = ehAppendAgentBubble('The Coach', 'school', '#fbbf24', 'Agent 8 · Coaching');
                        coachDiv.innerHTML = "<span class='text-[#d0c5af]/40 italic'>Agent 8 is writing your coaching note...</span>";

                        try {
                            await ehStreamSSE('/api/essay/coach', {
                                original_essay: essay,
                                rewritten_essay: ehRewrittenEssay,
                                examiner_report: ehExaminerReport
                            }, coachDiv, 'Agent 8');

                            setStatus('Pipeline complete ✔  |  All 3 agents have finished.');
                            
                            // Show copy essay button
                            const copyBtn = document.getElementById('eh-copy-essay-btn');
                            if (copyBtn) copyBtn.classList.remove('hidden');

                            // Add "Start Over" button
                            const resetBtn = document.createElement('button');
                            resetBtn.className = 'mt-6 bg-[#131318] text-[#4ade80] border border-[#4ade80]/40 px-6 py-3 rounded font-sans font-bold text-sm uppercase tracking-wider hover:border-[#4ade80] hover:bg-[#4ade80]/10 transition-all flex items-center gap-2';
                            resetBtn.innerHTML = '<span class="material-symbols-outlined text-base">refresh</span> Analyze Another Essay';
                            resetBtn.addEventListener('click', () => {
                                document.getElementById('eh-essay-input').value = '';
                                document.getElementById('eh-assignment').value = '';
                                document.getElementById('eh-analyze-btn').disabled = false;
                                document.getElementById('eh-chat-history').classList.add('hidden');
                                document.getElementById('essayhelper-setup').classList.remove('hidden');
                                setStatus('');
                            });
                            coachDiv.appendChild(resetBtn);

                        } catch (e) {
                            console.error('Agent 8 error:', e);
                            coachDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 8 lost.</span>";
                        }
                    });
                    rewriteDiv.appendChild(coachBtn);

                } catch (e) {
                    console.error('Agent 7 error:', e);
                    rewriteDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 7 lost.</span>";
                }
            });
            examinerDiv.appendChild(rewriteBtn);

        } catch (e) {
            console.error('Agent 6 error:', e);
            examinerDiv.innerHTML = "<span class='text-[#ffb4ab] italic'>Connection to Agent 6 lost.</span>";
        }
    });
}
