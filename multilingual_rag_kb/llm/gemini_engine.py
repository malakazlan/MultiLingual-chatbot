import google.generativeai as genai
from multilingual_rag_kb.config import GEMINI_API_KEY

def get_gemini_response(prompt: str, model: str = 'gemini-2.0-flash-001') -> str:
    api_key = GEMINI_API_KEY
    if not api_key or api_key == "your-gemini-api-key-here":
        return '[GEMINI ERROR] GEMINI_API_KEY not set in config.py.'
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[GEMINI ERROR] {str(e)}"