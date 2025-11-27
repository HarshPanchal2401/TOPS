# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(dotenv_path="../.env")
from resume_parser import extract_text_pdf, simple_section_split, extract_skills, parse_resume_comprehensive
from matcher import tfidf_similarity, semantic_similarity, calculate_ats_score
from email_sender import send_selection_email, send_interview_invite_email, send_job_selection_email
from job_descriptions import get_job_description, get_available_roles
from utils import DATA_DIR, VECTOR_STORE_PATH
from typing import Dict, Any
from fastapi import HTTPException
from fastapi import Body
from fastapi import Depends
from fastapi import Query
from fastapi import Response
from fastapi import status
from fastapi import BackgroundTasks
from fastapi import Request
from fastapi import Header
from fastapi import APIRouter
from pydantic import BaseModel
import time
import hashlib
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore

# Interview utilities
INTERVIEW_SESSIONS: Dict[str, Dict[str, Any]] = {}

def safe_float_env(key: str, default_str: str) -> float:
    """Parse environment float robustly (e.g., handles trailing commas/spaces)."""
    raw = os.environ.get(key, default_str)
    if raw is None:
        return float(default_str)
    try:
        # Strip spaces and common stray characters
        cleaned = raw.strip().rstrip(",")
        return float(cleaned)
    except Exception:
        return float(default_str)

class StartInterviewRequest(BaseModel):
    job_role: str
    candidate_name: str | None = None
    candidate_email: str | None = None
    candidate_phone: str | None = None

class SubmitAnswerResponse(BaseModel):
    question_index: int
    transcript: str
    combined_score: float
    result: str

class FinishInterviewResponse(BaseModel):
    passed_questions: int
    total_questions: int
    overall_pass: bool
    email_sent: bool
    email_status: str
import numpy as np
import faiss
import json
import requests

# LangChain (with backward-compatible imports)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as LCFAISS
except Exception:  # pragma: no cover
    try:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain.vectorstores import FAISS as LCFAISS  # type: ignore
    except Exception:
        HuggingFaceEmbeddings = None  # type: ignore
        LCFAISS = None  # type: ignore

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = DATA_DIR
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# In-memory store for RAG (LangChain vectorstore)
GLOBAL_RAG = {"vectorstore": None, "chunks": [], "embed_model": None, "candidate_data": None}

@app.get("/health")
async def health():
    return {"status": "ok"}


def _generate_with_gemini(full_prompt: str) -> str:
    """Call Google Gemini generateContent API with model fallbacks."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Retrieved resume information only. (Missing GOOGLE_API_KEY)"

    models = [
        os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash"),
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-pro-latest",
    ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 400,
            "topP": 0.8,
            "topK": 10,
        },
    }

    for i, model in enumerate(models):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("candidates"):
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif resp.status_code in (400, 429):
                if i < len(models) - 1:
                    continue
                return f"Gemini API error ({resp.status_code})."
            else:
                if i < len(models) - 1:
                    continue
                return f"Gemini API error ({resp.status_code})."
        except Exception as e:
            if i < len(models) - 1:
                continue
            return f"Gemini error: {str(e)}"

    return "All Gemini models failed."

@app.get("/job_roles")
async def get_job_roles():
    """Get available job roles"""
    return {"roles": get_available_roles()}


@app.post("/upload_and_score")
async def upload_and_score(resume: UploadFile = File(...), job_role: str = Form(...)):
    # save resume file
    rid = str(uuid.uuid4())[:8]
    resume_path = os.path.join(UPLOAD_DIR, f"{rid}_{resume.filename}")
    with open(resume_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    # Parse resume comprehensively
    try:
        resume_data = parse_resume_comprehensive(resume_path)
        resume_text = resume_data.get("text", "")
        
        # Check for parsing errors
        if "error" in resume_data:
            raise ValueError(f"PDF parsing failed: {resume_data['error']}")
        if not resume_text or len(resume_text.strip()) < 20:
            raise ValueError("PDF text extraction failed: extracted text is too short or empty")
    except Exception as e:
        # Clean up file on error
        try:
            os.remove(resume_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}. Please ensure the PDF contains readable text.")
    
    # Get predefined job description
    jd_text = get_job_description(job_role)
    
    # Extract candidate information
    candidate_email = resume_data["email"]
    candidate_name = resume_data["name"] or (resume.filename.split(".")[0]).replace("_", " ")
    candidate_phone = resume_data["phone"]

    # compute comprehensive ATS score
    ats_results = calculate_ats_score(resume_text, jd_text)
    score = ats_results['final_score']
    tfidf_sim = ats_results['tfidf_similarity']
    sem_sim = ats_results['semantic_similarity']
    keyword_sim = ats_results['keyword_similarity']
    bonus_score = ats_results.get('bonus_score', 0)
    base_score = ats_results.get('base_score', score)

    # prepare RAG chunks
    chunks = []
    # simple split paragraphs
    for p in [c.strip() for c in resume_text.split("\n\n") if c.strip()]:
        if len(p) > 40:
            chunks.append(p)
    if not chunks:
        chunks = [resume_text[:1000]]

    # Build LangChain vectorstore (FAISS) for RAG
    embed_model_name = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers/all-MiniLM-L6-v2")
    if HuggingFaceEmbeddings is not None and LCFAISS is not None:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        vectorstore = LCFAISS.from_texts(chunks, embedding=embeddings)
        GLOBAL_RAG["vectorstore"] = vectorstore
        GLOBAL_RAG["chunks"] = chunks
        GLOBAL_RAG["embed_model"] = embed_model_name
    else:
        GLOBAL_RAG["vectorstore"] = None
        GLOBAL_RAG["chunks"] = chunks
        GLOBAL_RAG["embed_model"] = embed_model_name

    # Set threshold and (optionally) send email automatically if score is above threshold
    threshold = safe_float_env("SCORE_THRESHOLD", "75")
    email_sent = False
    email_status = "No email found in resume"
    
    auto_email = os.environ.get("AUTO_EMAIL_ON_SCORE", "0") == "1"
    if auto_email:
        if score >= threshold:
            if candidate_email:
                try:
                    send_interview_invite_email(
                            candidate_email,
                            candidate_name,
                            job_role.replace("_", " ").title(),
                            subject=os.environ.get("RESUME_PASS_SUBJECT", f"Resume Shortlisted – {job_role.replace('_', ' ').title()}")
                        )
                    email_sent = True
                    email_status = f"Selection email sent to {candidate_email}"
                except Exception as e:
                    email_sent = False
                    email_status = f"Failed to send email: {str(e)}"
            else:
                email_status = "Score above threshold but no email found in resume"
        else:
            email_status = f"Score {score:.1f} below threshold {threshold} - no email sent"
    else:
        email_status = "Auto-email on resume score is disabled (set AUTO_EMAIL_ON_SCORE=1)"

    # Store comprehensive candidate data for RAG chatbot
    candidate_data = {
        "resume_analysis": {
            "score": round(float(score), 2),
            "tfidf": tfidf_sim,
            "semantic": sem_sim,
            "keyword": keyword_sim,
            "threshold": threshold,
            "job_role": job_role,
            "passed_threshold": score >= threshold,
            "email_sent": email_sent,
            "email_status": email_status
        },
        "candidate_info": {
            "name": candidate_name,
            "email": candidate_email,
            "phone": candidate_phone,
            "skills": resume_data["skills"]
        },
        "interview": {
            "attempted": False,
            "completed": False,
            "passed": False,
            "total_questions": 0,
            "correct_answers": 0,
            "wrong_answers": 0,
            "final_selection": False
        },
        "resume_text": resume_text
    }
    
    GLOBAL_RAG["candidate_data"] = candidate_data

    return {
        "score": round(float(score), 2), 
        "tfidf": tfidf_sim, 
        "semantic": sem_sim,
        "keyword": keyword_sim,
        "bonus": bonus_score,
        "base_score": base_score,
        "email_sent": email_sent,
        "email_status": email_status,
        "candidate_info": {
            "name": candidate_name,
            "email": candidate_email,
            "phone": candidate_phone,
            "skills": resume_data["skills"]
        },
        "threshold": threshold,
        "job_role": job_role
    }

@app.post("/chat")
async def chat(question: str = Form(...), k: int = Form(3)):
    # Use LangChain vectorstore
    if GLOBAL_RAG["vectorstore"] is None:
        return {"answer": "No resume loaded yet. Upload and analyze a resume first."}

    # If the user asks for a simple metric, return it directly (concise)
    q_lower = (question or "").lower().strip()
    candidate_data = GLOBAL_RAG.get("candidate_data", {})
    resume_analysis = candidate_data.get("resume_analysis", {}) if candidate_data else {}
    if resume_analysis:
        if ("ats" in q_lower and "score" in q_lower) or q_lower in {"ats", "score"}:
            try:
                return {"answer": f"{float(resume_analysis.get('score', 0)):.2f}%"}
            except Exception:
                return {"answer": str(resume_analysis.get('score', 'Not available'))}
        if "threshold" in q_lower:
            try:
                return {"answer": f"{float(resume_analysis.get('threshold', 0)):.2f}%"}
            except Exception:
                return {"answer": str(resume_analysis.get('threshold', 'Not available'))}

    retriever = GLOBAL_RAG["vectorstore"].as_retriever(search_kwargs={"k": int(k)})
    # Support both new and old LangChain retriever APIs
    try:
        # New Runnable-style API
        docs = retriever.invoke(question)  # type: ignore[attr-defined]
    except Exception:
        # Fallbacks for older APIs
        try:
            docs = retriever.get_relevant_documents(question)  # type: ignore[attr-defined]
        except Exception:
            docs = retriever._get_relevant_documents(question)  # type: ignore[attr-defined]
    resume_context = "\n\n".join([d.page_content for d in docs])
    
    # Get comprehensive candidate data
    candidate_data = GLOBAL_RAG.get("candidate_data", {})
    
    # Build comprehensive context
    context_parts = []
    
    # Resume context
    if resume_context.strip():
        context_parts.append(f"Resume Content:\n{resume_context}")
    
    # Candidate information
    if candidate_data:
        candidate_info = candidate_data.get("candidate_info", {})
        if candidate_info:
            context_parts.append(f"""
Candidate Information:
- Name: {candidate_info.get('name', 'Not available')}
- Email: {candidate_info.get('email', 'Not available')}
- Phone: {candidate_info.get('phone', 'Not available')}
- Skills: {', '.join(candidate_info.get('skills', [])) if candidate_info.get('skills') else 'Not detected'}
""")
        
        # Resume analysis results
        resume_analysis = candidate_data.get("resume_analysis", {})
        if resume_analysis:
            context_parts.append(f"""
Resume Analysis Results:
- ATS Score: {resume_analysis.get('score', 0)}%
- Threshold: {resume_analysis.get('threshold', 0)}%
- Passed Threshold: {'Yes' if resume_analysis.get('passed_threshold', False) else 'No'}
- Job Role Applied: {resume_analysis.get('job_role', 'Not specified').replace('_', ' ').title()}
- Email Sent: {resume_analysis.get('email_status', 'Not sent')}
""")
        
        # Interview results
        interview = candidate_data.get("interview", {})
        if interview:
            context_parts.append(f"""
Interview Results:
- Interview Attempted: {'Yes' if interview.get('attempted', False) else 'No'}
- Interview Completed: {'Yes' if interview.get('completed', False) else 'No'}
- Interview Passed: {'Yes' if interview.get('passed', False) else 'No'}
- Total Questions: {interview.get('total_questions', 0)}
- Correct Answers: {interview.get('correct_answers', 0)}
- Wrong Answers: {interview.get('wrong_answers', 0)}
- Final Selection: {'Selected' if interview.get('final_selection', False) else 'Not Selected'}
""")
    
    if not context_parts:
        return {"answer": "No relevant information found."}

    full_context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a comprehensive AI assistant that answers questions about candidates throughout their entire hiring journey. "
        "You have access to resume details, ATS scoring, interview performance, and final selection status. "
        "Answer questions about any aspect of the candidate's profile, resume analysis, interview results, or selection status. "
        "Be concise by default. If the user asks for a single numeric value (e.g., ATS score or threshold), reply with ONLY that value. "
        "If information is not available, clearly state that."
    )
    
    user_prompt = f"""Candidate Data:
{full_context}

Question: {question}

Please provide a clear, comprehensive answer based on the available information above."""
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    ans = _generate_with_gemini(full_prompt)
    
    # Clean markdown formatting from response
    cleaned_ans = ans.replace("**", "").replace("*", "").strip()
    
    return {"answer": cleaned_ans}

# ---------------- Interview: role-based questions, audio, scoring ----------------

# Load interview questions from JSON file
def load_interview_questions():
    """Load interview questions from JSON file."""
    try:
        import json
        from pathlib import Path
        # Look for questions file in parent directory first, then current directory
        questions_file = Path(__file__).parent.parent / "interview_questions.json"
        if not questions_file.exists():
            questions_file = Path(__file__).parent / "interview_questions.json"
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("⚠️ interview_questions.json not found, using fallback questions")
            return _get_fallback_questions()
    except Exception as e:
        print(f"⚠️ Error loading interview questions: {e}, using fallback")
        return _get_fallback_questions()

def _get_fallback_questions():
    """Fallback questions if JSON file is not available."""
    return {
        "software_engineer": [
            {
                "question": "What is Python and why is it used?",
                "answer": "Python is a high-level programming language used for general-purpose programming, known for its simplicity and readability.",
                "keywords": ["high-level programming language", "simplicity", "readability"]
            },
            {
                "question": "Explain the concept of OOP.",
                "answer": "OOP stands for Object-Oriented Programming, which organizes code using objects and classes, supporting encapsulation, inheritance, and polymorphism.",
                "keywords": ["object-oriented programming", "classes", "objects", "encapsulation", "inheritance", "polymorphism"]
            },
        ],
        "data_scientist": [
            {
                "question": "What is overfitting in machine learning?",
                "answer": "Overfitting occurs when a model learns noise in the training data and fails to generalize to unseen data.",
                "keywords": ["overfitting", "generalize", "noise", "training data"]
            },
            {
                "question": "Explain bias-variance tradeoff.",
                "answer": "The bias-variance tradeoff balances underfitting and overfitting to minimize total error.",
                "keywords": ["bias", "variance", "underfitting", "overfitting"]
            },
        ],
        "product_manager": [
            {"question": "What is a product roadmap?", "answer": "A product roadmap is a high-level plan that outlines product vision, priorities, and timeline.", "keywords": ["vision", "priorities", "timeline"]},
            {"question": "How do you prioritize features?", "answer": "Use frameworks like RICE or MoSCoW considering impact, effort, and strategic fit.", "keywords": ["RICE", "MoSCoW", "impact", "effort"]},
        ],
        "devops_engineer": [
            {"question": "What is CI/CD?", "answer": "CI/CD automates building, testing, and deploying code to deliver software quickly and reliably.", "keywords": ["continuous integration", "continuous delivery", "automation"]},
            {"question": "Explain containerization.", "answer": "Containerization packages an application and its dependencies into a portable container image.", "keywords": ["Docker", "Kubernetes", "isolation"]},
        ],
        "ui_ux_designer": [
            {"question": "What is user-centered design?", "answer": "User-centered design focuses on users' needs at every design stage.", "keywords": ["user needs", "usability", "research"]},
            {"question": "Wireframes vs prototypes?", "answer": "Wireframes outline structure; prototypes are interactive representations for testing.", "keywords": ["wireframe", "prototype", "testing"]},
        ],
    }

# Load questions from JSON file at startup
INTERVIEW_QUESTIONS: Dict[str, list[dict[str, Any]]] = load_interview_questions()

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _embedder = None  # type: ignore

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def _embed(text: str) -> np.ndarray:
    if _embedder is None:
        return np.random.rand(384).astype(np.float32)  # fallback to random to avoid crash
    return _embedder.encode(text)

def _concept_score(candidate_answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    cand_vec = _embed(candidate_answer)
    sims: list[float] = []
    for kw in keywords:
        kw_vec = _embed(kw)
        sims.append(_cosine_similarity(cand_vec, kw_vec))
    return float(np.mean(sims)) if sims else 0.0

def _evaluate(candidate_answer: str, correct_answer: str, keywords: list[str], threshold: float) -> tuple[float, str]:
    ans_vec = _embed(candidate_answer)
    cor_vec = _embed(correct_answer)
    ans_sim = _cosine_similarity(ans_vec, cor_vec)
    kw_sim = _concept_score(candidate_answer, keywords)
    combined = (ans_sim + kw_sim) / 2.0
    return float(combined), ("PASS" if combined >= threshold else "FAIL")

def _load_whisper():
    if whisper is None:
        return None
    try:
        return whisper.load_model(os.environ.get("WHISPER_MODEL", "tiny.en"))
    except Exception:
        return None

WHISPER_MODEL = _load_whisper()

@app.post("/interview/start")
async def start_interview(payload: StartInterviewRequest):
    role = payload.job_role.lower()
    qs = INTERVIEW_QUESTIONS.get(role)
    if not qs:
        raise HTTPException(status_code=400, detail="Unsupported job role")
    session_id = hashlib.sha256(f"{time.time()}-{role}-{payload.candidate_email}".encode()).hexdigest()[:16]
    INTERVIEW_SESSIONS[session_id] = {
        "job_role": role,
        "candidate": {
            "name": payload.candidate_name,
            "email": payload.candidate_email,
            "phone": payload.candidate_phone,
        },
        "results": [],
        "threshold": safe_float_env("INTERVIEW_THRESHOLD", "0.6"),
    }
    
    # Update candidate data to mark interview as attempted
    if GLOBAL_RAG.get("candidate_data"):
        GLOBAL_RAG["candidate_data"]["interview"]["attempted"] = True
        GLOBAL_RAG["candidate_data"]["interview"]["total_questions"] = len(qs)
    
    public_questions = [{"question": q["question"]} for q in qs]
    return {"session_id": session_id, "questions": public_questions}

@app.post("/interview/answer", response_model=SubmitAnswerResponse)
async def submit_answer(session_id: str = Form(...), question_index: int = Form(...), audio: UploadFile = File(...)):
    session = INTERVIEW_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    role = session["job_role"]
    qs = INTERVIEW_QUESTIONS[role]
    if question_index < 0 or question_index >= len(qs):
        raise HTTPException(status_code=400, detail="Invalid question index")

    # Save temp audio
    tmp_path = os.path.join(UPLOAD_DIR, f"tmp_{session_id}_{question_index}.wav")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Transcribe using whisper if available, else return error
    if WHISPER_MODEL is None:
        raise HTTPException(status_code=500, detail="Transcription model not available on server")
    try:
        tr = WHISPER_MODEL.transcribe(tmp_path)
        transcript = tr.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    q = qs[question_index]
    combined_score, result = _evaluate(transcript, q["answer"], q.get("keywords", []), session["threshold"])
    session["results"].append({
        "question_index": question_index,
        "transcript": transcript,
        "score": combined_score,
        "result": result,
    })
    
    # Update candidate data with interview progress
    if GLOBAL_RAG.get("candidate_data"):
        if result == "PASS":
            GLOBAL_RAG["candidate_data"]["interview"]["correct_answers"] += 1
        else:
            GLOBAL_RAG["candidate_data"]["interview"]["wrong_answers"] += 1
    
    return SubmitAnswerResponse(question_index=question_index, transcript=transcript, combined_score=combined_score, result=result)

@app.post("/interview/finish", response_model=FinishInterviewResponse)
async def finish_interview(session_id: str = Form(...)):
    session = INTERVIEW_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    role = session["job_role"]
    total_questions = len(INTERVIEW_QUESTIONS[role])
    passed_questions = sum(1 for r in session["results"] if r["result"] == "PASS")
    overall_pass = (passed_questions / max(total_questions, 1)) >= safe_float_env("INTERVIEW_OVERALL_PASS", "0.6")

    email_sent = False
    email_status = "Interview not passed. No email sent."
    if overall_pass:
        cand = session.get("candidate", {})
        to_email = cand.get("email")
        name = cand.get("name")
        if to_email:
            try:
                send_job_selection_email(
                    to_email,
                    name or "Candidate",
                    role.replace("_", " ").title(),
                    subject=os.environ.get("INTERVIEW_PASS_SUBJECT", f"Congratulations – Selected for {role.replace('_',' ').title()}")
                )
                email_sent = True
                email_status = f"Selection email sent to {to_email}"
            except Exception as e:
                email_sent = False
                email_status = f"Failed to send email: {e}"
        else:
            email_status = "Interview passed but no candidate email available."

    # Update candidate data with final interview results
    if GLOBAL_RAG.get("candidate_data"):
        GLOBAL_RAG["candidate_data"]["interview"]["completed"] = True
        GLOBAL_RAG["candidate_data"]["interview"]["passed"] = overall_pass
        GLOBAL_RAG["candidate_data"]["interview"]["final_selection"] = overall_pass and email_sent

    # cleanup session
    try:
        del INTERVIEW_SESSIONS[session_id]
    except Exception:
        pass

    return FinishInterviewResponse(
        passed_questions=passed_questions,
        total_questions=total_questions,
        overall_pass=overall_pass,
        email_sent=email_sent,
        email_status=email_status,
    )
