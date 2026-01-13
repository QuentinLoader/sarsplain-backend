from fastapi import FastAPI
import requests
import io
import os

import pdfplumber
from openai import OpenAI

app = FastAPI()

# Initialise OpenAI client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_letter(letter_text: str) -> str:
    """
    Sends the extracted SARS letter text to OpenAI
    and returns a plain-English explanation.
    """

    prompt = f"""
You are a South African tax letter explanation assistant.

IMPORTANT RULES:
- You must ONLY use the text provided below.
- If the letter text is incomplete, unclear, or missing, you MUST say so.
- DO NOT guess the letter type.
- DO NOT provide generic SARS explanations.
- DO NOT hallucinate content.

Your task:
1. Identify the SARS letter type ONLY if clearly stated.
2. Explain the letter in plain English.
3. List exactly what SARS is requesting.
4. Extract any deadlines explicitly mentioned.
5. Explain consequences ONLY if stated in the letter.
6. Provide safe next steps without giving tax advice.

If the letter content is insufficient, respond with:
"The letter text provided is incomplete or unreadable, so a reliable explanation cannot be given."

LETTER TEXT:
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
    """
    Receives a file URL from ManyChat, downloads the PDF,
    extracts text, and returns a plain-English explanation.
    """

    try:
        file_url = payload.get("file_url")
        if not file_url:
            return {"result": "No file was provided."}

        # Download the file
        response = requests.get(file_url, timeout=20)
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "").lower()

        # Only allow PDFs for v1 (important for stability)
        if "pdf" not in content_type:
            return {
                "result": (
                    "This file does not appear to be a PDF.\n\n"
                    "Please upload a text-based SARS PDF letter."
                )
            }

        # Extract text from PDF
        extracted_text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""

        # Minimum text check (prevents AI guessing)
        if not extracted_text or len(extracted_text.strip()) < 200:
            return {
                "result": (
                    "We couldn’t reliably read the text from this letter.\n\n"
                    "This usually means:\n"
                    "• The PDF is a scanned image\n"
                    "• The text is not machine-readable\n\n"
                    "Please upload a text-based SARS PDF or a clearer version."
                )
            }

        # Explain the letter using AI
        explanation = explain_letter(extracted_text)

        return {"result": explanation}

    except Exception as e:
        # Fail safely — never crash or expose stack traces to users
        return {
            "result": (
                "An internal error occurred while processing the letter. "
                "Please try again later."
            ),
            "debug": str(e)
        }
