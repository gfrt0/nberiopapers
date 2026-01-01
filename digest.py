# from dotenv import load_dotenv
# load_dotenv("./.env")

import os
# import requests
import smtplib
from email.message import EmailMessage
from bs4 import BeautifulSoup
import feedparser
import json
from groq import Groq
import re

# -------------------------
# Load environment variables
# -------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USER)

if not all([GROQ_API_KEY, SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
    raise RuntimeError("Missing required environment variables")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# Step 1: Fetch new IO papers from RSS
# -------------------------
RSS_URL = "https://back.nber.org/rss/newio.xml"

def clean_html(html_text):
    return BeautifulSoup(html_text, "html.parser").get_text(strip=True)

def fetch_new_io_papers():
    feed = feedparser.parse(RSS_URL)
    papers = []
    for entry in feed.entries:
        papers.append({
            "title": entry.title,
            "url": entry.link,
            "abstract": clean_html(entry.summary)
        })
    return papers

# -------------------------
# Step 2: Call Groq LLM for structured extraction
# -------------------------
def parse_groq_json(raw_text):
    """
    Extract JSON from Groq chat output, stripping ```json``` fences if present.
    Returns a Python dict.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        json_text = raw_text.strip()

    return json.loads(json_text)

def summarize_with_groq(paper):
    prompt = f"""
Extract the following fields from this NBER Industrial Organization paper.
Return valid JSON only.

Fields:
- research_question
- method
- data
- main_result (one sentence)

TITLE:
{paper['title']}

ABSTRACT:
{paper['abstract']}
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You extract structured information from economics papers."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    raw_content = chat_completion.choices[0].message.content
    return parse_groq_json(raw_content)

# -------------------------
# Step 3: Build aggregated weekly email
# -------------------------
def build_email(papers_summaries):
    if not papers_summaries:
        return {
            "subject": "NBER IO Digest: No new papers this week",
            "body": "There are no new Industrial Organization papers this week."
        }

    subject = f"NBER IO Digest: {len(papers_summaries)} new papers"

    body_lines = ["This week's NBER Industrial Organization papers:\n"]

    for i, ps in enumerate(papers_summaries, 1):
        paper = ps["paper"]
        summary = ps["summary"]
        body_lines.append(f"{i}. {paper['title']}")
        body_lines.append(f"URL: {paper['url']}")
        body_lines.append(f"Research question: {summary.get('research_question', 'N/A')}")
        body_lines.append(f"Method: {summary.get('method', 'N/A')}")
        body_lines.append(f"Data: {summary.get('data', 'N/A')}")
        body_lines.append(f"Main result: {summary.get('main_result', 'N/A')}")
        body_lines.append("\n")

    body = "\n".join(body_lines)
    return {"subject": subject, "body": body}

# -------------------------
# Step 4: Send email
# -------------------------
def send_email(subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASSWORD)
        smtp.send_message(msg)

# -------------------------
# Main workflow
# -------------------------
def main():
    papers = fetch_new_io_papers()
    papers_summaries = []

    for paper in papers:
        try:
            summary = summarize_with_groq(paper)
            papers_summaries.append({"paper": paper, "summary": summary})
        except Exception as e:
            print(f"Failed to summarize {paper['title']}: {e}")

    email_content = build_email(papers_summaries)
    send_email(email_content["subject"], email_content["body"])
    print(f"Email sent: {email_content['subject']}")

if __name__ == "__main__":
    main()
