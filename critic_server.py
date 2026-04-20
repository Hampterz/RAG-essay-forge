# WARNING: This file is provided for archival and reference purposes and is not
# actively integrated into the main pipeline by default.

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import json
import requests
import concurrent.futures
import google.generativeai as genai
from mistralai.client import Mistral

MISTRAL_API_KEY = "YOUR_MISTRAL_KEY_HERE"
GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"

app = Flask(__name__)
CORS(app)

qwen_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

CRITIC_PROMPT = """You are a cynical Ivy League admissions dean with 20 years of experience. 
You have seen every AI-generated essay trick in the book and you are NOT impressed.

You will receive a rewritten college essay. Your job is to red-team it ruthlessly.

Look specifically for:
1. "AI Gloss" — phrases that sound sophisticated but are hollow (furthermore, delving, tapestry, multifaceted, pivotal, fostered, in conclusion, etc.)
2. Forced transitions that no 17-year-old would write
3. Moments where the voice shifts from teenage to robotic
4. Any factual claim that feels invented or generic
5. Structural clichés (starting with a question, ending with "changed me forever", etc.)

Respond ONLY in this exact JSON format, no extra text:
{
  "overall_verdict": "PASS or FAIL",
  "ai_gloss_phrases": ["phrase1", "phrase2"],
  "voice_breaks": ["quote the exact sentence that breaks voice"],
  "structural_flags": ["specific structural issue"],
  "forced_transitions": ["exact transition phrase"],
  "final_note": "one sentence of the most important fix needed"
}"""

def call_mistral(original_essay, rewritten_essay):
    try:
        with Mistral(api_key=MISTRAL_API_KEY) as mistral:
            res = mistral.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": "You are a Harvard admissions officer with 15 years of experience. Your job is to flag writing that feels artificial — but you understand that strong, sophisticated writing from a talented teenager is NOT the same as AI writing. Do not flag a phrase just because it sounds smart or uses technical vocabulary. Only flag something if it genuinely breaks the teenage voice or sounds like it came from a consultant or an AI. Specifically look for: hollow buzzwords with no concrete meaning, sentences where the voice shifts from personal and specific to abstract and corporate, structural clichés like question hooks or \"this experience changed me forever\" endings, and transitions that no teenager would write. Do NOT flag: specific technical terms the student clearly knows, vivid sensory details, sophisticated vocabulary that fits the context, or short punchy sentences. Return ONLY valid JSON with these exact fields: overall_verdict (PASS or FAIL), failure_reasons (array of specific reasons, empty if PASS), ai_gloss_phrases (array), voice_breaks (array of exact quoted sentences), structural_flags (array), forced_transitions (array), final_note (one sentence). No preamble, no markdown."},
                    {"role": "user", "content": f"<original_essay>\n{original_essay}\n</original_essay>\n\n<rewritten_essay>\n{rewritten_essay}\n</rewritten_essay>\n\nReturn only JSON."}
                ],
                stream=False,
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            return ("mistral", json.loads(raw.strip()))
    except Exception as e:
        print(f"[Council] Mistral failed: {e}")
        return ("mistral", {"overall_verdict": "UNAVAILABLE", "error": str(e)})

def call_gemini(original_essay, rewritten_essay):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            system_instruction="You are a Pulitzer Prize-winning journalist who coaches college applicants. You have a sharp ear for the difference between authentic sophisticated writing and performed sophistication. A 17-year-old who genuinely knows their subject can write with precision and depth — that is NOT a red flag. Only flag a sentence if it sounds like it was written to impress a committee rather than to express a real experience. Look for: hollow abstractions that could apply to any student, transitions that sound like a five-paragraph essay, corporate buzzwords disguised as insight, and moments where the personal story disappears into generic statements about society. Do NOT flag: technical vocabulary the student clearly understands, specific sensory details, strong metaphors, or any sentence that is vivid and grounded in a real moment. Return ONLY valid JSON with these exact fields: overall_verdict (PASS or FAIL), failure_reasons (array of specific reasons, empty if PASS), ai_gloss_phrases (array), voice_breaks (array of exact quoted sentences), structural_flags (array), forced_transitions (array), final_note (one sentence). No preamble, no markdown."
        )
        prompt = f"<original_essay>\n{original_essay}\n</original_essay>\n\n<rewritten_essay>\n{rewritten_essay}\n</rewritten_essay>\n\nReturn only JSON."
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return ("gemini3flash", json.loads(raw.strip()))
    except Exception as e:
        print(f"[Council] Gemini 3 Flash failed: {e}")
        return ("gemini3flash", {"overall_verdict": "UNAVAILABLE", "error": str(e)})

@app.route("/critique", methods=["POST"])
def critique():
    data = request.json
    rewritten_essay = data.get("rewritten_essay", "")
    original_essay = data.get("original_essay", "")

    print(f"\n[Council] New request. Calling Mistral Small and Gemini 3 Flash in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_mistral = executor.submit(call_mistral, original_essay, rewritten_essay)
        future_gemini = executor.submit(call_gemini, original_essay, rewritten_essay)
        mistral_name, mistral_result = future_mistral.result()
        gemini_name, gemini_result = future_gemini.result()

    print(f"[Council] Mistral verdict: {mistral_result.get('overall_verdict')} | Gemini verdict: {gemini_result.get('overall_verdict')}")
    print("[Council] Sending both to Qwen arbiter for final synthesis...")

    arbiter_prompt = f"""You are the Chief Council Arbiter for a college essay review panel. Two expert reviewers have independently critiqued the same essay. Your job is to synthesize their findings into one final verdict.

Rules:
- If either reviewer says FAIL, your final verdict is FAIL
- If both say PASS, your final verdict is PASS
- If one reviewer is UNAVAILABLE, base your verdict solely on the other
- Only include a flagged item in your final output if BOTH reviewers flagged it, OR if one reviewer flagged it and it is clearly an AI gloss phrase with no concrete meaning
- Do NOT include flags that target sophisticated vocabulary, technical terms, or vivid specific details — these are strengths not weaknesses
- Combine all qualifying flags, removing exact duplicates
- Your final_note must name the single most important fix in one concrete sentence — not a general observation

Here are the two critiques:

Mistral Small critique:
{json.dumps(mistral_result, indent=2)}

Gemini 3 Flash critique:
{json.dumps(gemini_result, indent=2)}

Return ONLY this JSON, no explanation, no markdown, no extra text:
{{
  "overall_verdict": "PASS or FAIL",
  "council_breakdown": {{
    "mistral_small": "PASS or FAIL",
    "gemini3flash": "PASS or FAIL"
  }},
  "failure_reasons": [],
  "ai_gloss_phrases": [],
  "voice_breaks": [],
  "structural_flags": [],
  "forced_transitions": [],
  "final_note": ""
}}"""

    try:
        response = qwen_client.chat.completions.create(
            model="qwen/qwen3.5-9b",
            messages=[
                {"role": "user", "content": arbiter_prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )

        raw = response.choices[0].message.content
        if not raw:
            raw = getattr(response.choices[0].message, 'reasoning_content', '') or ''
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]

        try:
            final = json.loads(raw.strip())
        except:
            final = {
                "overall_verdict": "FAIL",
                "council_breakdown": {
                    "mistral_small": mistral_result.get("overall_verdict", "UNAVAILABLE"),
                    "gemini3flash": gemini_result.get("overall_verdict", "UNAVAILABLE")
                },
                "failure_reasons": mistral_result.get("failure_reasons", []) + gemini_result.get("failure_reasons", []),
                "ai_gloss_phrases": mistral_result.get("ai_gloss_phrases", []) + gemini_result.get("ai_gloss_phrases", []),
                "voice_breaks": mistral_result.get("voice_breaks", []) + gemini_result.get("voice_breaks", []),
                "structural_flags": mistral_result.get("structural_flags", []) + gemini_result.get("structural_flags", []),
                "forced_transitions": mistral_result.get("forced_transitions", []) + gemini_result.get("forced_transitions", []),
                "final_note": mistral_result.get("final_note", gemini_result.get("final_note", "Review flagged items."))
            }
    except Exception as e:
        print(f"[Council] Qwen Arbiter failed: {e}")
        final = {
            "overall_verdict": "FAIL",
            "council_breakdown": {
                "mistral_small": mistral_result.get("overall_verdict", "UNAVAILABLE"),
                "gemini3flash": gemini_result.get("overall_verdict", "UNAVAILABLE")
            },
            "failure_reasons": mistral_result.get("failure_reasons", []) + gemini_result.get("failure_reasons", []),
            "ai_gloss_phrases": mistral_result.get("ai_gloss_phrases", []) + gemini_result.get("ai_gloss_phrases", []),
            "voice_breaks": mistral_result.get("voice_breaks", []) + gemini_result.get("voice_breaks", []),
            "structural_flags": mistral_result.get("structural_flags", []) + gemini_result.get("structural_flags", []),
            "forced_transitions": mistral_result.get("forced_transitions", []) + gemini_result.get("forced_transitions", []),
            "final_note": "Arbiter failed to process the request."
        }

    print(f"[Council] Final verdict: {final.get('overall_verdict')}\n")
    return jsonify(final)

def check_models():
    try:
        r = requests.get("http://localhost:1234/v1/models")
        if r.status_code == 200:
            models = r.json().get("data", [])
            model_names = [m.get("id") for m in models]
            print("Available models in LM Studio:", model_names)
        else:
            print(f"Failed to fetch models, status: {r.status_code}")
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    check_models()
    app.run(host="0.0.0.0", port=5001)
