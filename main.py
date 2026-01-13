from fastapi import FastAPI
import requests
import io

import pdfplumber
from PIL import Image
import pytesseract

from openai import OpenAI

app = FastAPI()

import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def extract_text_from_file(file_bytes, content_type):
    # If it's a PDF
    if "pdf" in content_type:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    # Otherwise, treat as image
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)
def explain_letter(letter_text):
    prompt = f"""
You are a South African tax assistant.

Explain the following SARS letter in plain English.
Do NOT give tax advice.

Include:
- Letter type
- What it means
- What SARS wants
- Deadlines
- Consequences
- Safe next steps

SARS LETTER:
{letter_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
@app.post("/analyze-letter")
def analyze_letter(payload: dict):
    file_url = payload.get("file_url")

    if not file_url:
        return {"error": "No file URL provided"}

    # Download the file
    response = requests.get(file_url)
    file_bytes = response.content
    content_type = response.headers.get("Content-Type", "")

    # Extract text
    letter_text = extract_text_from_file(file_bytes, content_type)

    if not letter_text.strip():
        return {"result": "Could not read the letter clearly."}

    # Explain using AI
    explanation = explain_letter(letter_text)

    return {
        "result": explanation
    }
