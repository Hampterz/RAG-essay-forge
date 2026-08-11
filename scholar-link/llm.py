import json
import google.generativeai as genai

def get_gemini_models(api_key):
    """Fetch available Gemini models that support text generation."""
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        raise Exception(f"Failed to fetch models: {e}")

def generate_draft(profile, professor, template):
    api_key = profile.get("gemini_api_key")
    model_name = profile.get("gemini_model")
    
    if not api_key:
        raise Exception("Gemini API key is not configured. Please set it in your Profile.")
    if not model_name:
        raise Exception("Gemini model is not selected. Please set it in your Profile.")

    genai.configure(api_key=api_key)
    
    system_prompt = (
        "You are helping a student write a short, genuine, specific academic "
        "outreach email to a professor. Write in the student's voice: clear, "
        "respectful, not overly formal or flowery. Reference the professor's "
        "actual research. Keep it under 200 words. "
        "Generate a relevant email subject line and the email body. "
        "You must output ONLY valid JSON with two string keys: 'subject' and 'body'. "
        "Do not include markdown blocks or any other text."
    )

    user_prompt = f"""
Student name: {profile['name']}
Student background: {profile['background']}
Student interests: {profile['interests']}

Professor name: {professor['name']}
Professor affiliation: {professor['affiliation']}
Professor research summary: {professor['research_interests']}

Reference template (tone/structure guide, adapt freely, don't copy placeholders literally):
{template}

Write the email now. Remember to output ONLY JSON.
"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood. I will provide the response strictly as a JSON object with 'subject' and 'body' keys."]},
            {"role": "user", "parts": [user_prompt]}
        ])
        
        text = response.text
        # Strip potential markdown code block backticks if the model ignores the instruction
        if text.strip().startswith("```json"):
            text = text.strip()[7:]
            if text.strip().endswith("```"):
                text = text.strip()[:-3]
        elif text.strip().startswith("```"):
            text = text.strip()[3:]
            if text.strip().endswith("```"):
                text = text.strip()[:-3]
        
        result = json.loads(text.strip())
        subject = result.get("subject", "Interest in your research")
        body = result.get("body", "").strip()
        
        if not body:
             raise Exception("Generated body was empty.")
             
        return subject, body
        
    except Exception as e:
        raise Exception(f"Gemini API error: {e}")
