# backend/email_sender.py
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")

# LLM config - using Gemini only
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
COMPANY_NAME = os.environ.get("COMPANY_NAME", "Hiring Team")


# -------------------------------
# Templates
# -------------------------------
def _generate_professional_email_text(candidate_name: str, job_title: str, company_name: str) -> str:
    return (
        f"Subject: Interview Invitation – {job_title}\n\n"
        f"Hi {candidate_name},\n\n"
        f"Congratulations! We were impressed with your background and would like to move forward "
        f"with the interview process for the {job_title} role at {company_name}.\n\n"
        f"Next steps:\n"
        f"- Share your availability for a 30–45 minute interview this week\n"
        f"- We will confirm the meeting details and send a calendar invite\n\n"
        f"If you have any questions, feel free to reply directly to this email.\n\n"
        f"Best regards,\n"
        f"{company_name}"
    )


def _generate_professional_email_html(candidate_name: str, job_title: str, company_name: str) -> str:
    return f"""
        <html>
            <body style="font-family: Arial, Helvetica, sans-serif; line-height: 1.6; color: #1f2937;">
                <p>Hi {candidate_name},</p>
                <p>
                    Congratulations! We were impressed with your background and would like to move forward with the
                    interview process for the <strong>{job_title}</strong> role at {company_name}.
                </p>
                <p><strong>Next steps:</strong></p>
                <ul>
                    <li>Share your availability for a 30–45 minute interview this week</li>
                    <li>We will confirm the meeting details and send a calendar invite</li>
                </ul>
                <p>If you have any questions, feel free to reply directly to this email.</p>
                <p style="margin-top: 18px;">Best regards,<br/>{company_name}</p>
            </body>
        </html>
    """


# -------------------------------
# LLM-based email generation
# -------------------------------
def _generate_llm_body(candidate_name: str, job_title: str, company_name: str, job_desc: str, mode: str) -> tuple[str, str]:
    """Internal: mode in {invite, selection}. Returns (text, html)."""
    import json
    import requests

    if mode == "invite":
        system_prompt = (
            "You are an assistant that drafts concise, professional interview invitation emails. "
            "Use a warm, neutral tone. Keep it around 150 words. Include next steps and a call to action."
        )
        user_prompt = (
            f"Draft an interview invitation for {candidate_name} for the role {job_title} at {company_name}. "
            f"Job Description: {job_desc if job_desc else 'General fresher/early career role'}.\n\n"
            f"Congratulate, briefly note fit, and ask for availability. Output plain text only."
        )
    else:  # selection
        system_prompt = (
            "You are an assistant that drafts concise, professional job selection/offer emails. "
            "Use a warm, neutral tone. Keep it around 150 words. Include congratulations, start date/next steps, and contact."
        )
        user_prompt = (
            f"Draft a job selection email for {candidate_name} for the role {job_title} at {company_name}. "
            f"Congratulate them on successfully clearing the interview. Outline next steps. Output plain text only."
        )

    def _to_html(text_content: str) -> str:
        return (
            "<html><body style=\"font-family: Arial, Helvetica, sans-serif; line-height: 1.6; color: #1f2937;\">"
            + "".join(f"<p>{line}</p>" for line in text_content.split("\n\n") if line.strip())
            + "</body></html>"
        )

    # Use Gemini API for email generation
    if GOOGLE_API_KEY:
        try:
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": 0.4},
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
            resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text_content, _to_html(text_content)
            else:
                print(f"⚠️ Gemini API failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"⚠️ Gemini email generation failed: {e}")
    else:
        print("⚠️ No Google API key found, using fallback templates")

    # fallback templates per mode
    if mode == "invite":
        return (
            _generate_professional_email_text(candidate_name, job_title, company_name),
            _generate_professional_email_html(candidate_name, job_title, company_name),
        )
    else:
        # For selection, reuse text but tweak headline in HTML
        txt = (
            f"Subject: Selection – {job_title}\n\n"
            f"Hi {candidate_name},\n\n"
            f"Congratulations! We are pleased to select you for the role of {job_title} at {company_name}.\n\n"
            f"Our HR team will reach out with the offer details and onboarding steps.\n\n"
            f"Best regards,\n{company_name}"
        )
        html = (
            f"<html><body style=\"font-family: Arial, Helvetica, sans-serif; line-height: 1.6; color: #1f2937;\">"
            f"<p>Hi {candidate_name},</p>"
            f"<p><strong>Congratulations!</strong> You have been selected for the role of <strong>{job_title}</strong> at {company_name}.</p>"
            f"<p>Our HR team will contact you with the offer details and onboarding steps.</p>"
            f"<p style=\"margin-top: 18px;\">Best regards,<br/>{company_name}</p>"
            f"</body></html>"
        )
        return txt, html


def generate_llm_email_body(candidate_name: str, job_title: str, company_name: str, job_desc: str = "") -> tuple[str, str]:
    return _generate_llm_body(candidate_name, job_title, company_name, job_desc, mode="invite")

def generate_llm_email_body_selection(candidate_name: str, job_title: str, company_name: str, job_desc: str = "") -> tuple[str, str]:
    return _generate_llm_body(candidate_name, job_title, company_name, job_desc, mode="selection")


# -------------------------------
# Send Email
# -------------------------------
def send_selection_email(to_email: str, candidate_name: str, job_title: str, job_desc: str = "", body: str = None, subject: str | None = None) -> bool:
    """Backward-compat: sends interview invitation by default."""
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASS:
        print("❌ SMTP settings missing. Please check your .env file.")
        return False

    candidate_name = candidate_name.title().strip() if candidate_name else "Candidate"

    if body is None:
        text_body, html_body = generate_llm_email_body(candidate_name, job_title, COMPANY_NAME, job_desc)
    else:
        text_body = body
        html_body = _generate_professional_email_html(candidate_name, job_title, COMPANY_NAME)

    msg = EmailMessage()
    msg["Subject"] = subject or f"Interview Invitation – {job_title}"
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        print(f"✅ Email sent successfully to {candidate_name} ({to_email})")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def send_interview_invite_email(to_email: str, candidate_name: str, job_title: str, job_desc: str = "", subject: str | None = None) -> bool:
    text_body, html_body = generate_llm_email_body(candidate_name, job_title, COMPANY_NAME, job_desc)
    return send_selection_email(to_email, candidate_name, job_title, job_desc, body=text_body, subject=subject or f"Interview Invitation – {job_title}")


def send_job_selection_email(to_email: str, candidate_name: str, job_title: str, job_desc: str = "", subject: str | None = None) -> bool:
    text_body, html_body = generate_llm_email_body_selection(candidate_name, job_title, COMPANY_NAME, job_desc)
    # Use direct SMTP to preserve HTML variant
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASS:
        print("❌ SMTP settings missing. Please check your .env file.")
        return False
    candidate_name = candidate_name.title().strip() if candidate_name else "Candidate"
    msg = EmailMessage()
    msg["Subject"] = subject or f"Selection – {job_title}"
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        print(f"✅ Email sent successfully to {candidate_name} ({to_email})")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
