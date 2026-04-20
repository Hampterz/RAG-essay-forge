import json
import os
import sys

# Force UTF-8 for print() on Windows (avoids cp1252 UnicodeEncodeError for log symbols)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = FastAPI()

# Make sure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Initializing ChromaDB and Model...")
try:
    chroma_client = chromadb.PersistentClient(path="essay_db")
    collection = chroma_client.get_collection(name="essays")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ ChromaDB and embedding model loaded successfully.")
except Exception as e:
    print(f"⚠️  WARNING: Could not initialize ChromaDB or models: {e}")
    # Don't exit here so the server can at least start to show UI.

# ── Agent 5 (Humanizer) collection ──────────────────────────────────────────
human_patterns_collection = None
try:
    human_patterns_collection = chroma_client.get_collection(name="human_patterns")
    hp_count = human_patterns_collection.count()
    print(f"✅ Humanizer collection loaded: {hp_count} sentence patterns")
except Exception as e:
    print(f"⚠️  human_patterns collection not found — run build_humanizer_index.py first ({e})")

openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class InitialChatRequest(BaseModel):
    user_essay: str

class ChatMessage(BaseModel):
    role: str
    content: str

class OngoingChatRequest(BaseModel):
    messages: list[ChatMessage]

class RewriteRequest(BaseModel):
    user_essay: str
    distillation_report: str
    agent2_feedback: str

class Agent4Request(BaseModel):
    verdict_json: str
    agent1_report: str
    agent2_feedback: str
    rewritten_essay: str

class HumanizeRequest(BaseModel):
    essay_text: str
    distillation_report: str = ""
    agent2_feedback: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: emit a properly-formatted SSE event
# CRITICAL FIX: use real newlines (\n\n), NOT escaped \\n\\n
# ─────────────────────────────────────────────────────────────────────────────
def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def sse_done() -> str:
    return "data: [DONE]\n\n"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/start_chat")
async def start_chat(req: InitialChatRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/start_chat] ▶  Agent 1 pipeline started")
        print(f"  Essay length: {len(req.user_essay)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Initializing Pass 1 [Agent 1: Background Distiller]...'})
            print("[Agent 1] Status sent: Initializing...")

            # 1. RAG Retrieval
            print("[Agent 1] Encoding essay for ChromaDB query...")
            user_embedding = embedding_model.encode(req.user_essay, show_progress_bar=False).tolist()
            yield sse({'status': 'Agent 1: Identifying top 20 successful benchmarks from ChromaDB...'})
            print("[Agent 1] Running ChromaDB query (n_results=20)...")

            results = collection.query(
                query_embeddings=[user_embedding],
                n_results=20
            )

            retrieved_docs = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0]
            print(f"[Agent 1] ✅ Retrieved {len(retrieved_docs)} benchmark essays from ChromaDB.")

            formatted_references = ""
            for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
                school = meta.get("school", "unknown")
                formatted_references += f"--- Benchmark Essay {i+1} (Accepted to {school}) ---\n"
                formatted_references += f"{doc}\n\n"
                print(f"[Agent 1]   Benchmark {i+1}: school={school}, length={len(doc)} chars")

            # 2. Execute Agent 1
            agent1_system_prompt = """Role: You are a Senior Admissions Data Analyst. Your task is to perform a cold, clinical, and high-density comparative analysis between a student's raw essay and twenty successful Ivy League-caliber benchmarks.

Task: Synthesize the "DNA" of the successful benchmarks. Identify the specific Delta (the gap) between the student's current draft and the benchmarks.

Output Requirements (Comparative Distillation Report):
1. Structural Velocity: How does the student's pacing compare? (e.g., "Student lingers 30% too long on the intro; Benchmarks hit the 'pivot' by word 200").
2. Tonal Frequency: Is the student too formal? Too casual? Compare the "Vulnerability Score."
3. Thematic Delta: What specific "intangibles" do the benchmarks have that this essay lacks? (e.g., intellectual humility, specific sensory imagery, a non-linear narrative).

Constraint: Do NOT provide feedback to the student. Do NOT use flowery language. Output a dense, technical briefing for an Admissions Counselor to use."""

            agent1_user_prompt = f"""Input Data for Analysis:
* Student Essay: {req.user_essay}
* Benchmarks:
{formatted_references}

Objective: Generate the Comparative Distillation Report. Focus exclusively on the gap between the student's work and the benchmarks."""

            yield sse({'status': 'Agent 1: Distilling 20 benchmarks (This takes ~60-120s)...'})
            print("[Agent 1] Calling LM Studio (stream=True, max_tokens=20000, temp=0.1)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent1_system_prompt},
                    {"role": "user", "content": agent1_user_prompt}
                ],
                temperature=0.1,
                stream=True,
                max_tokens=20000
            )

            distillation_report = ""
            tokens_generated = 0
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    distillation_report += chunk.choices[0].delta.content
                    tokens_generated += 1
                    if tokens_generated % 50 == 0:
                        yield sse({'status': f'Agent 1: Distilling benchmarks ({tokens_generated} tokens analyzed)...'})
                        print(f"[Agent 1]   ... {tokens_generated} tokens streamed from LM Studio")

            print(f"[Agent 1] ✅ LM Studio stream complete. Total tokens: {tokens_generated}")
            print(f"[Agent 1]   Report length: {len(distillation_report)} chars")
            yield sse({'status': 'Pass 1 Complete. Initializing Agent 2 (Primary Counselor)...'})

            # Print Agent 1's full output to server console for verification
            print("\n" + "="*60)
            print("AGENT 1 COMPARATIVE DISTILLATION REPORT:")
            print("="*60)
            print(distillation_report)
            print("="*60 + "\n")

            # 3. Build Agent 2 system prompt
            agent2_system_prompt = f"""Role: You are a warm, world-class College Admissions Mentor with a Master's in Creative Writing. You are empathetic, insightful, and deeply invested in the student's success.

The Secret Sauce: You have been provided with a "Distillation Report" (background intelligence). Use this report to guide your feedback, but never mention the report's existence to the student.

Formatting Constraints (STRICT):
* NO rigid Markdown headers (No `## Grade` or `### Feedback`).
* NO sterile numbered lists.
* DO use elegant, flowing paragraphs.
* DO use "soft" bullet points (using \u2022 or \u25e6) integrated into your prose.
* DO use bold text sparingly for emphasis within sentences.

Tone: Write like a letter from a mentor. Use phrases like "I noticed that you..." or "Your story really shines when..." Avoid "AI-speak" like "In conclusion" or "It is important to note."

Goal: Provide a realistic Ivy League grade (A-F) and actionable advice on pacing, "the hook," and emotional resonance based on the provided distillation.

[INTERNAL DISTILLATION REPORT]:
{distillation_report}"""

            agent2_user_prompt = f"""Student Essay:
{req.user_essay}

Objective: Evaluate this essay. Start with a warm greeting and a holistic impression. Weave the grade and the specific improvements naturally into your conversation. Make the student feel supported, not audited."""

            # ── CRITICAL HANDOFF PACKET ──────────────────────────────────────
            # This event MUST be received by the frontend to trigger sendChat()
            handoff_payload = {
                'system_prompt': agent2_system_prompt,
                'user_prompt': agent2_user_prompt,
                'distillation_report': distillation_report
            }
            print("[Agent 1] ── Emitting HANDOFF packet to frontend (system_prompt + user_prompt + distillation_report)")
            print(f"[Agent 1]   system_prompt length : {len(agent2_system_prompt)} chars")
            print(f"[Agent 1]   user_prompt length   : {len(agent2_user_prompt)} chars")
            print(f"[Agent 1]   distillation_report  : {len(distillation_report)} chars")
            yield sse(handoff_payload)

            print("[Agent 1] ── Emitting [DONE] signal")
            yield sse_done()
            print("[/api/start_chat] ✅ Stream complete — Agent 1 finished, handoff to Agent 2 dispatched.\n")

        except Exception as e:
            import traceback
            print(f"[/api/start_chat] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': str(e)})
            yield sse_done()

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat(req: OngoingChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    def generate():
        print("\n" + "="*60)
        print("[/api/chat] ▶  Agent 2 (Counselor) started")
        print(f"  Incoming message count: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"  msg[{i}] role={m['role']} len={len(m['content'])} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 2: Connecting to counselor persona...'})
            print("[Agent 2] Calling LM Studio (stream=True, max_tokens=20000, temp=0.7)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=0.7,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - call_start
                        print(f"[Agent 2] First token in {ttft:.2f}s (TTFT = prompt processing)")
                        yield sse({'status': 'Generating response...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 2]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            # tok/s = tokens generated / time spent GENERATING (excludes TTFT/prompt-processing)
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1

            metrics = {
                "total_time": round(total_time, 2),
                "ttft": ttft,
                "tok_s": round(tokens / gen_time, 2) if gen_time > 0 else 0,
                "total_tokens": usage.get("total_tokens") if usage else tokens
            }
            print(f"[Agent 2] Done. output_tokens={tokens}, ttft={ttft}s, gen_time={round(gen_time,2)}s, tok/s={metrics['tok_s']}")
            yield sse({'metrics': metrics})

        except Exception as e:
            import traceback
            print(f"[/api/chat] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'LM Studio Error: {str(e)} - If it says model reloaded, you may lack VRAM or have exceeded context size limits.'})

        print("[Agent 2] Emitting [DONE]")
        yield sse_done()
        print("[/api/chat] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/rewrite")
async def rewrite_essay(req: RewriteRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/rewrite] ▶  Agent 3 (Rewriter) started")
        print(f"  user_essay length         : {len(req.user_essay)} chars")
        print(f"  distillation_report length: {len(req.distillation_report)} chars")
        print(f"  agent2_feedback length    : {len(req.agent2_feedback)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 3: Initializing Complete Professional Rewrite...'})
            print("[Agent 3] Building prompts...")

            agent3_system_prompt = """Role: You are a brilliant, highly articulate 17-year-old high school senior applying to an elite university. You are rewriting your personal statement.

Voice Guidelines:
* Vocal Texture: Authentic, soulful, and sophisticated. Use sentence variety (mix short, punchy sentences with longer, lyrical ones).
* The "Human" Element: Keep the student's original core story and "quirks." Elevate the vocabulary, but keep it within the realm of a well-read teenager.

Forbidden "AI-isms" (CRITICAL):
You are strictly banned from using the following words/phrases: furthermore, delving, fostering, tapestry, testament, multifaceted, underscore, in conclusion, embarked, vibrant, pivotal, navigating the complexities, transformational journey.

Instruction: Using the Counselor's Feedback and the Distillation Report, rewrite the entire essay. Focus on "Show, Don't Tell." If the student says they are "hardworking," rewrite a scene that proves it.

Goal: Create a 650-word masterpiece that feels like it was written in a notebook by a teenager at 2:00 AM in a moment of pure inspiration. Do not include any "Introduction" or "Notes" from the AI\u2014output only the rewritten essay."""

            agent3_user_prompt = f"""<original_essay>
{req.user_essay}
</original_essay>

<context_report>
{req.distillation_report}
</context_report>

<counselor_advice>
{req.agent2_feedback}
</counselor_advice>

Objective: Rewrite the essay from the first-person perspective. Integrate all suggestions seamlessly. Output only the rewritten essay without any intro/outro text."""

            print("[Agent 3] Calling LM Studio (stream=True, max_tokens=20000, temp=0.85)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent3_system_prompt},
                    {"role": "user", "content": agent3_user_prompt}
                ],
                temperature=0.85,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - call_start
                        print(f"[Agent 3] First token in {ttft:.2f}s (TTFT)")
                        yield sse({'status': 'Agent 3 generating full rewrite...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 3]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1

            metrics = {
                "total_time": round(total_time, 2),
                "ttft": ttft,
                "tok_s": round(tokens / gen_time, 2) if gen_time > 0 else 0,
                "total_tokens": usage.get("total_tokens") if usage else tokens
            }
            print(f"[Agent 3] Done. output_tokens={tokens}, ttft={ttft}s, gen_time={round(gen_time,2)}s, tok/s={metrics['tok_s']}")
            yield sse({'metrics': metrics})

        except Exception as e:
            import traceback
            print(f"[/api/rewrite] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'LM Studio Error: {str(e)}'})

        print("[Agent 3] Emitting [DONE]")
        yield sse_done()
        print("[/api/rewrite] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


class FixRequest(BaseModel):
    user_essay: str
    distillation_report: str
    agent2_feedback: str


@app.post("/api/fix")
async def fix_essay(req: FixRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/fix] ▶  Agent 3.5 (Fixer) started")
        print(f"  user_essay length         : {len(req.user_essay)} chars")
        print(f"  distillation_report length: {len(req.distillation_report)} chars")
        print(f"  agent2_feedback length    : {len(req.agent2_feedback)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 3.5: Initializing Targeted Fix (preserving your voice)...'})
            print("[Agent 3.5] Building prompts...")

            agent35_system_prompt = """Role: You are a meticulous copy-editor and college admissions essay coach. You are NOT a rewriter. Your job is to surgically fix ONLY the specific problems identified in the Counselor's feedback, while preserving the student's authentic voice, style, tone, and structure as much as humanly possible.

Rules (CRITICAL):
1. PRESERVE the student's original sentence structures, word choices, and personality. The essay should still "sound like them."
2. FIX ONLY what the counselor flagged — grammar issues, weak phrasing, unclear passages, pacing problems, missing specificity, etc.
3. DO NOT rewrite sections that weren't flagged. If a paragraph is fine, leave it EXACTLY as-is.
4. When fixing a flagged issue, make the MINIMUM change necessary. Replace a weak word, tighten a sentence, add a concrete detail — but don't rebuild the whole sentence.
5. If the counselor suggested adding sensory imagery or a specific scene, weave it in naturally using the student's existing voice and vocabulary level.
6. Keep the same overall word count (within ±50 words of the original).
7. The final essay should read like the student revised their own draft after getting good feedback — NOT like someone else rewrote it.

Forbidden: Do NOT change the narrative structure, the opening, or the core story arc unless those were specifically flagged. Do NOT inject vocabulary the student wouldn't naturally use.

Output: The corrected essay only. No notes, no commentary, no preamble."""

            agent35_user_prompt = f"""<original_essay>
{req.user_essay}
</original_essay>

<counselor_feedback>
{req.agent2_feedback}
</counselor_feedback>

<distillation_context>
{req.distillation_report}
</distillation_context>

Objective: Fix ONLY the issues the counselor identified. Keep the student's voice intact. Output only the corrected essay."""

            print("[Agent 3.5] Calling LM Studio (stream=True, max_tokens=20000, temp=0.3)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent35_system_prompt},
                    {"role": "user", "content": agent35_user_prompt}
                ],
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - call_start
                        print(f"[Agent 3.5] First token in {ttft:.2f}s (TTFT)")
                        yield sse({'status': 'Agent 3.5 applying targeted fixes...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 3.5]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1

            metrics = {
                "total_time": round(total_time, 2),
                "ttft": ttft,
                "tok_s": round(tokens / gen_time, 2) if gen_time > 0 else 0,
                "total_tokens": usage.get("total_tokens") if usage else tokens
            }
            print(f"[Agent 3.5] Done. output_tokens={tokens}, ttft={ttft}s, gen_time={round(gen_time,2)}s, tok/s={metrics['tok_s']}")
            yield sse({'metrics': metrics})

        except Exception as e:
            import traceback
            print(f"[/api/fix] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'LM Studio Error: {str(e)}'})

        print("[Agent 3.5] Emitting [DONE]")
        yield sse_done()
        print("[/api/fix] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/agent4")
async def agent4_surgeon(req: Agent4Request):
    def generate():
        print("\n" + "="*60)
        print("[/api/agent4] ▶  Agent 4 (Surgeon) started")
        print(f"  verdict_json length    : {len(req.verdict_json)} chars")
        print(f"  agent1_report length   : {len(req.agent1_report)} chars")
        print(f"  agent2_feedback length : {len(req.agent2_feedback)} chars")
        print(f"  rewritten_essay length : {len(req.rewritten_essay)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 4: Initializing Surgical Fixes...'})
            print("[Agent 4] Building prompts...")

            agent4_system_prompt = """You are a surgical essay editor. You receive a strong rewritten college essay and a red flag report from a cynical admissions council. Your job is to fix ONLY the explicitly flagged issues \u2014 nothing else.

Rules:
- Fix ONLY phrases that appear word-for-word in the council's flagged lists
- If a sentence is not flagged, do not touch it \u2014 not even one word
- Replace flagged AI gloss phrases with something equally specific and vivid \u2014 never simpler or more generic
- Fix voice breaks by making the sentence more personal and grounded, not by dumbing it down
- Remove flagged forced transitions but keep the logical flow intact
- Keep 95%+ of the essay intact
- Preserve all technical vocabulary, specific details, proper nouns, and sensory language \u2014 these are strengths
- NEVER use: furthermore, delving, tapestry, multifaceted, pivotal, fostered, in conclusion, testament, beacon, journey, landscape, unadulterated, resonance, organism
- The voice must sound like a sharp specific 17-year-old \u2014 sophisticated is fine, corporate is not
- If the council verdict is PASS, return the essay completely unchanged
- Output ONLY the corrected essay. No commentary, no explanation, no preamble."""

            agent4_user_prompt = f"""<council_verdict>
{req.verdict_json}
</council_verdict>

<agent1_report>
{req.agent1_report}
</agent1_report>

<agent2_feedback>
{req.agent2_feedback}
</agent2_feedback>

<rewritten_essay>
{req.rewritten_essay}
</rewritten_essay>

Look at the council_verdict carefully. If overall_verdict is PASS, return the rewritten_essay exactly as-is. If FAIL, fix only the exact phrases listed in the flagged fields. Do not rewrite any sentence that is not flagged. Output only the final essay."""

            print("[Agent 4] Calling LM Studio (stream=True, max_tokens=20000, temp=0.4)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent4_system_prompt},
                    {"role": "user", "content": agent4_user_prompt}
                ],
                temperature=0.4,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - call_start
                        print(f"[Agent 4] First token in {ttft:.2f}s (TTFT)")
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 4]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1
            tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0
            print(f"[Agent 4] Done. output_tokens={tokens}, ttft={ttft}s, gen_time={round(gen_time,2)}s, tok/s={tok_s}")
            yield sse({'metrics': {'total_time': round(total_time,2), 'ttft': ttft, 'tok_s': tok_s, 'total_tokens': usage.get('total_tokens') if usage else tokens}})

        except Exception as e:
            import traceback
            print(f"[/api/agent4] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'Agent 4 Error: {str(e)}'})

        print("[Agent 4] Emitting [DONE]")
        yield sse_done()
        print("[/api/agent4] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5: THE HUMANIZER — sentence-level RAG + anti-AI rewrite
# ─────────────────────────────────────────────────────────────────────────────
import re as _re

def _split_sentences_for_rag(text: str) -> list[str]:
    """Quick sentence splitter for RAG queries (mirrors build_humanizer_index.py)."""
    text = _re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    parts = _re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text)
    return [s.strip() for s in parts if len(s.strip()) >= 40]


@app.post("/api/humanize")
async def humanize_essay(req: HumanizeRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/humanize] ▶  Agent 5 (The Humanizer) started")
        print(f"  essay_text length: {len(req.essay_text)} chars")
        print("="*60)

        try:
            if human_patterns_collection is None or human_patterns_collection.count() == 0:
                yield sse({'error': 'Humanizer index not built. Run: python build_humanizer_index.py'})
                yield sse_done()
                return

            # ── Step 1: Split the input essay into sentences ─────────────
            yield sse({'status': 'Agent 5: Splitting essay into sentences...'})
            sentences = _split_sentences_for_rag(req.essay_text)
            print(f"[Agent 5] Split essay into {len(sentences)} sentences")

            if not sentences:
                yield sse({'error': 'Could not extract sentences from the essay.'})
                yield sse_done()
                return

            # ── Step 2: Embed each sentence and query human_patterns ────
            yield sse({'status': f'Agent 5: Querying {len(sentences)} sentences against human DNA database...'})
            top_k_per_sentence = 3
            all_human_sentences = []
            seen = set()

            for i, sent in enumerate(sentences):
                try:
                    sent_embedding = embedding_model.encode(sent, show_progress_bar=False).tolist()
                    results = human_patterns_collection.query(
                        query_embeddings=[sent_embedding],
                        n_results=top_k_per_sentence
                    )
                    for doc in results['documents'][0]:
                        if doc not in seen:
                            seen.add(doc)
                            all_human_sentences.append(doc)
                except Exception as e:
                    print(f"[Agent 5] ⚠️  Error querying sentence {i}: {e}")
                    continue

                if (i + 1) % 5 == 0 or i == len(sentences) - 1:
                    yield sse({'status': f'Agent 5: Processed {i+1}/{len(sentences)} sentences...'})
                    print(f"[Agent 5]   Processed sentence {i+1}/{len(sentences)}, total human matches: {len(all_human_sentences)}")

            # Cap the context to avoid exceeding model limits
            max_context_sentences = 100
            human_context = all_human_sentences[:max_context_sentences]
            print(f"[Agent 5] Total unique human sentences retrieved: {len(all_human_sentences)}, using top {len(human_context)}")

            # ── Step 3: Build the Humanizer prompt ──────────────────────
            yield sse({'status': 'Agent 5: Generating humanized rewrite from sentence DNA...'})

            human_dna_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(human_context)])

            agent5_system_prompt = """You are Agent 5 — The Humanizer. Your only job is to make this essay sound like a real teenager actually typed it. Not a polished teenager. A real one.

You have been given a set of real sentences written by actual high school students. Study them carefully — notice how they start sentences with "And" or "But", how they use dashes mid-thought, how they repeat themselves slightly, how they drop into a specific memory without warning, how their sentences are uneven lengths.

Now rewrite the essay to match that texture. You must:
- Break at least 3 long smooth sentences into shorter rougher ones
- Add at least 2 specific physical details that ground abstract statements
- Start at least 2 sentences with And, But, or So
- Add at least 1 parenthetical aside that sounds like a real thought
- Make the transitions imperfect — real essays lurch forward, they don't glide
- Replace any remaining generic phrasing with something specific and concrete

Burstiness rules — vary sentence length aggressively:
- At least 2 sentences must be under 8 words
- At least 2 sentences must be over 35 words
- Never have more than 3 sentences of similar length in a row

Perplexity rules — make word choices less predictable:
- Replace the most obvious word in each sentence with a more specific or unexpected one
- Avoid starting consecutive sentences with the same word or structure
- Add at least one sentence that takes an unexpected turn mid-thought using a dash

Authenticity rules:
- Add one moment of self-correction or qualification ("or maybe it was", "I think", "I'm not sure why but")
- Add one incomplete thought that trails into something else
- Real teenagers don't conclude — they just stop writing when they've said what they needed to say. The last sentence should feel like an ending, not a conclusion.

Do not make it simpler. Do not make it worse. Make it human.
Do not use: furthermore, tapestry, multifaceted, pivotal, fostered, in conclusion, testament, beacon, resonate, underscore, transformative.
Output only the final essay. No commentary."""

            agent5_user_prompt = f"""<human_dna_bank>
Below are {len(human_context)} real sentences from successful student essays. Use their rhythm, voice, and texture as your calibration reference:

{human_dna_block}
</human_dna_bank>

<essay_to_humanize>
{req.essay_text}
</essay_to_humanize>

Objective: Rewrite this essay so it reads like a real 17-year-old wrote it at 2 AM in their bedroom. Ground every sentence in the patterns you see in the human DNA bank above. Kill all AI-speak. Keep every specific detail and story beat from the original. Output only the humanized essay."""

            print(f"[Agent 5] Prompt built. System: {len(agent5_system_prompt)} chars, User: {len(agent5_user_prompt)} chars")
            print("[Agent 5] Calling LM Studio (stream=True, max_tokens=20000, temp=0.85)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent5_system_prompt},
                    {"role": "user", "content": agent5_user_prompt}
                ],
                temperature=0.85,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - call_start
                        print(f"[Agent 5] First token in {ttft:.2f}s (TTFT)")
                        yield sse({'status': 'Agent 5 generating humanized output...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 5]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1
            tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0

            metrics = {
                'total_time': round(total_time, 2),
                'ttft': ttft,
                'tok_s': tok_s,
                'total_tokens': usage.get('total_tokens') if usage else tokens
            }
            print(f"[Agent 5] Done. output_tokens={tokens}, ttft={ttft}s, gen_time={round(gen_time,2)}s, tok/s={tok_s}")
            yield sse({'metrics': metrics})

        except Exception as e:
            import traceback
            print(f"[/api/humanize] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'Agent 5 Error: {str(e)}'})

        print("[Agent 5] Emitting [DONE]")
        yield sse_done()
        print("[/api/humanize] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# ESSAY HELPER: AGENTS 6, 7, 8
# ─────────────────────────────────────────────────────────────────────────────

class EssayAnalyzeRequest(BaseModel):
    essay: str
    assignment: str = ""
    essay_type: str = "Argumentative"
    grade_level: str = "High School"
    word_count_target: int = 500

class EssayRewriteRequest(BaseModel):
    essay: str
    assignment: str = ""
    essay_type: str = "Argumentative"
    grade_level: str = "High School"
    word_count_target: int = 500
    examiner_report: str = ""

class EssayCoachRequest(BaseModel):
    original_essay: str
    rewritten_essay: str
    examiner_report: str = ""


@app.post("/api/essay/analyze")
async def essay_analyze(req: EssayAnalyzeRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/essay/analyze] ▶  Agent 6 (The Examiner) started")
        print(f"  essay length: {len(req.essay)} chars")
        print(f"  type: {req.essay_type}, grade: {req.grade_level}, target: {req.word_count_target}")
        print("="*60)

        try:
            yield sse({'status': 'Agent 6: The Examiner is analyzing your essay...'})

            agent6_system_prompt = """You are a strict academic writing examiner. Your job is to analyze an essay against its assignment requirements and grade level. Evaluate the following dimensions and give a score out of 10 for each: Thesis clarity, Argument structure, Evidence quality, Transitions and flow, Grammar and mechanics, Adherence to essay type requirements. Then give an overall letter grade A through F. Write your feedback as a clear structured report — use headers for each dimension. Be direct and specific, quote exact lines from the essay when flagging issues. Do not be encouraging for the sake of it — if the thesis is weak, say it is weak and explain exactly why."""

            agent6_user_prompt = f"""<essay>
{req.essay}
</essay>

<assignment_details>
Assignment/Topic: {req.assignment if req.assignment else 'Not specified'}
Essay Type: {req.essay_type}
Grade Level: {req.grade_level}
Word Count Target: {req.word_count_target}
Actual Word Count: {len(req.essay.split())}
</assignment_details>

Evaluate this essay against the criteria above. Be thorough and specific."""

            print("[Agent 6] Calling LM Studio (stream=True, max_tokens=20000, temp=0.3)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent6_system_prompt},
                    {"role": "user", "content": agent6_user_prompt}
                ],
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        print(f"[Agent 6] First token in {now - call_start:.2f}s")
                        yield sse({'status': 'Agent 6 generating examination report...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 6]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1
            tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0
            print(f"[Agent 6] Done. output_tokens={tokens}, tok/s={tok_s}")
            yield sse({'metrics': {'total_time': round(total_time, 2), 'ttft': ttft, 'tok_s': tok_s, 'total_tokens': usage.get('total_tokens') if usage else tokens}})

        except Exception as e:
            import traceback
            print(f"[/api/essay/analyze] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'Agent 6 Error: {str(e)}'})

        yield sse_done()
        print("[/api/essay/analyze] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/essay/rewrite")
async def essay_rewrite(req: EssayRewriteRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/essay/rewrite] ▶  Agent 7 (The Rewriter) started")
        print(f"  essay length: {len(req.essay)} chars")
        print(f"  examiner_report length: {len(req.examiner_report)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 7: The Rewriter is rebuilding your essay...'})

            agent7_system_prompt = """You are an expert academic writing tutor. You have received an essay, its assignment requirements, the grade level, and an examiner's report. Your job is to rewrite the essay to address every issue flagged in the report while preserving the student's original argument and voice.

Rules:
- Keep the student's core thesis and argument intact
- Fix structural issues — if the intro is weak, rebuild it
- Strengthen evidence paragraphs — add logical reasoning where it is thin
- Fix transitions between paragraphs
- Match the writing sophistication to the grade level — middle school should sound different from undergraduate
- Do not add any facts or claims that were not in the original essay
- Output only the rewritten essay, no commentary"""

            agent7_user_prompt = f"""<original_essay>
{req.essay}
</original_essay>

<assignment_details>
Assignment/Topic: {req.assignment if req.assignment else 'Not specified'}
Essay Type: {req.essay_type}
Grade Level: {req.grade_level}
Word Count Target: {req.word_count_target}
</assignment_details>

<examiner_report>
{req.examiner_report}
</examiner_report>

Rewrite the essay to fix every flagged issue. Match the grade level. Output only the rewritten essay."""

            print("[Agent 7] Calling LM Studio (stream=True, max_tokens=20000, temp=0.6)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent7_system_prompt},
                    {"role": "user", "content": agent7_user_prompt}
                ],
                temperature=0.6,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        print(f"[Agent 7] First token in {now - call_start:.2f}s")
                        yield sse({'status': 'Agent 7 generating rewrite...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 7]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1
            tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0
            print(f"[Agent 7] Done. output_tokens={tokens}, tok/s={tok_s}")
            yield sse({'metrics': {'total_time': round(total_time, 2), 'ttft': ttft, 'tok_s': tok_s, 'total_tokens': usage.get('total_tokens') if usage else tokens}})

        except Exception as e:
            import traceback
            print(f"[/api/essay/rewrite] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'Agent 7 Error: {str(e)}'})

        yield sse_done()
        print("[/api/essay/rewrite] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/essay/coach")
async def essay_coach(req: EssayCoachRequest):
    def generate():
        print("\n" + "="*60)
        print("[/api/essay/coach] ▶  Agent 8 (The Coach) started")
        print(f"  original_essay length: {len(req.original_essay)} chars")
        print(f"  rewritten_essay length: {len(req.rewritten_essay)} chars")
        print("="*60)

        try:
            yield sse({'status': 'Agent 8: The Coach is preparing your feedback...'})

            agent8_system_prompt = """You are a patient academic writing coach giving a student personalized feedback after their essay has been rewritten. Your tone is encouraging but honest. 

Write a short coaching note that covers:
- What the student did well in their original essay that was preserved
- The 3 most important improvements made and why they matter
- One specific writing habit the student should practice going forward based on the patterns you noticed in their original draft
- A final encouraging line that is specific to this student's essay — not generic

Write in second person directly to the student. Keep it under 200 words. Do not use bullet points — write in flowing paragraphs like a real teacher would speak."""

            agent8_user_prompt = f"""<original_essay>
{req.original_essay}
</original_essay>

<rewritten_essay>
{req.rewritten_essay}
</rewritten_essay>

<examiner_report>
{req.examiner_report}
</examiner_report>

Write your coaching note to the student. Be specific to their essay. Keep it under 200 words."""

            print("[Agent 8] Calling LM Studio (stream=True, max_tokens=20000, temp=0.7)...")

            response = openai_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": agent8_system_prompt},
                    {"role": "user", "content": agent8_user_prompt}
                ],
                temperature=0.7,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=20000
            )

            call_start = time.time()
            first_token_time = None
            last_token_time = None
            tokens = 0
            usage = None

            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        print(f"[Agent 8] First token in {now - call_start:.2f}s")
                        yield sse({'status': 'Agent 8 writing coaching note...'})
                    last_token_time = now
                    tokens += 1
                    if tokens % 100 == 0:
                        print(f"[Agent 8]   ... {tokens} tokens streamed")
                    yield sse({'content': chunk.choices[0].delta.content})

            total_time = time.time() - call_start
            ttft = round(first_token_time - call_start, 2) if first_token_time else 0
            gen_time = (last_token_time - first_token_time) if (first_token_time and last_token_time and last_token_time > first_token_time) else 1
            tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0
            print(f"[Agent 8] Done. output_tokens={tokens}, tok/s={tok_s}")
            yield sse({'metrics': {'total_time': round(total_time, 2), 'ttft': ttft, 'tok_s': tok_s, 'total_tokens': usage.get('total_tokens') if usage else tokens}})

        except Exception as e:
            import traceback
            print(f"[/api/essay/coach] ❌ EXCEPTION: {e}")
            traceback.print_exc()
            yield sse({'error': f'Agent 8 Error: {str(e)}'})

        yield sse_done()
        print("[/api/essay/coach] ✅ Stream complete.\n")

    return StreamingResponse(generate(), media_type="text/event-stream")
