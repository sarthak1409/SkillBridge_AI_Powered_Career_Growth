# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import io
import re
import csv
import os
import logging
import traceback
from pathlib import Path

import pdfplumber
import docx2txt
import numpy as np

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skillbridge")

# =========================
# GLOBAL PLACEHOLDERS
# =========================
embed_model = None           # SentenceTransformer or None
spacy_nlp = None             # spaCy Language object or None
phrase_matcher = None        # spaCy PhraseMatcher or None (optional)
BASE_SKILLS: List[str] = []  # filled from CSV or fallback
SKILL_CATEGORIES: Dict[str, str] = {}

# =========================
# MODELS / SCHEMAS
# =========================
class GapAnalysisRequest(BaseModel):
    resume_text: str
    job_text: str

class ParseJobIn(BaseModel):
    text: str

class ParseJobOut(BaseModel):
    job_id: str
    parsed_skills: List[str]

class SkillGapItem(BaseModel):
    skill: str
    status: str
    score: float
    reason: str
    inferred: Dict[str, Any]
    category: str

class GapOut(BaseModel):
    matched: List[SkillGapItem]
    missing: List[SkillGapItem]
    suggested_plan: Dict[str, Any]

# =========================
# CONSTANTS / FILES
# =========================
HERE = Path(__file__).parent
SKILLS_FILE = HERE / "skills" / "unique_skills_dataset.csv"

# A minimal built-in fallback so the app still runs if the CSV is missing.
FALLBACK_SKILLS = [
    "Python", "JavaScript", "SQL", "React", "Django",
    "FastAPI", "Machine Learning", "Docker", "Kubernetes", "AWS",
    "Data Analysis", "NLP", "Pandas", "NumPy", "Git", "REST API"
]

# Synonyms / canonicalization
SYNONYM_MAP = {
    'oracle db': 'Oracle DB', 'flink': 'Flink', 'fedora': 'Fedora', 'github': 'GitHub',
    'azure': 'Azure', 'aws': 'AWS', 'photoshop': 'Photoshop', 'heroku': 'Heroku',
    'pandas': 'Pandas', 'python': 'Python', 'scikit-learn': 'Scikit-learn', 'redis': 'Redis',
    'curriculum design': 'Curriculum Design', 'seo': 'SEO', 'communication': 'Communication',
    'go': 'Go', 'teamwork': 'Teamwork', 'kubernetes': 'Kubernetes', 'excel': 'Excel',
    'mysql': 'MySQL', 'scrum': 'Scrum', 'react': 'React', 'xgboost': 'XGBoost',
    'opencv': 'OpenCV', 'ibm cloud': 'IBM Cloud', 'e-learning tools': 'E-learning Tools',
    'tableau': 'Tableau', 'nltk': 'NLTK', 'terraform': 'Terraform', 'swift': 'Swift',
    'tax planning': 'Tax Planning', 'numpy': 'NumPy', 'kanban': 'Kanban', 'django': 'Django',
    'mongodb': 'MongoDB', 'airflow': 'Airflow', 'ruby': 'Ruby', 'time management': 'Time Management',
    'hadoop': 'Hadoop', 'dvc': 'DVC', 'matplotlib': 'Matplotlib', 'illustrator': 'Illustrator',
    'mlops': 'MLOps', 'php': 'PHP', 'digitalocean': 'DigitalOcean', 'spacy': 'spaCy',
    'postgresql': 'PostgreSQL', 'tensorflow': 'TensorFlow', 'financial analysis': 'Financial Analysis',
    'sql': 'SQL', 'leadership': 'Leadership', 'bitbucket': 'Bitbucket', 'google sheets': 'Google Sheets',
    'catboost': 'CatBoost', 'javascript': 'JavaScript', 'seaborn': 'Seaborn', 'ansible': 'Ansible',
    'typescript': 'TypeScript', 'problem solving': 'Problem Solving', 'rust': 'Rust', 'spark': 'Spark',
    'mlflow': 'MLflow', 'market research': 'Market Research', 'angular': 'Angular', 'teaching': 'Teaching',
    'vue.js': 'Vue.js', 'fastapi': 'FastAPI', 'yolo': 'YOLO', 'git': 'Git', 'mediapipe': 'MediaPipe',
    'ubuntu': 'Ubuntu', 'agile': 'Agile', 'accounting': 'Accounting', 'gcp': 'GCP', 'linux': 'Linux',
    'gitlab': 'GitLab', 'email marketing': 'Email Marketing', 'next.js': 'Next.js', 'gpt': 'GPT',
    'docker': 'Docker', 'public speaking': 'Public Speaking', 'windows': 'Windows', 'kafka': 'Kafka',
    'sketch': 'Sketch', 'jenkins': 'Jenkins', 'sqlite': 'SQLite', 'java': 'Java', 'streamlit': 'Streamlit',
    'keras': 'Keras', 'c++': 'C++', 'lightgbm': 'LightGBM',
    'social media marketing': 'Social Media Marketing', 'canva': 'Canva', 'bert': 'BERT',
    'content writing': 'Content Writing', 'budgeting': 'Budgeting', 'pytorch': 'PyTorch',
    'macos': 'MacOS', 'transformers': 'Transformers', 'power bi': 'Power BI', 'figma': 'Figma',
    'flask': 'Flask', 'pmp certification': 'PMP Certification', 'sklearn': 'Scikit-learn',
    'tf': 'TensorFlow', 'postgres': 'PostgreSQL', 'rest api': 'REST API', 'api': 'REST API',
    'nlp': 'NLP', 'ps': 'Photoshop', 'ai': 'Artificial Intelligence', 'smm': 'Social Media Marketing'
}

def canonicalize(skill: str) -> str:
    s = skill.lower().strip()
    return SYNONYM_MAP.get(s, skill.strip().title())

# =========================
# INITIALIZATION HELPERS
# =========================
def load_skills_from_csv(path: Path):
    skills = []
    categories = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skill = (row.get("skill") or "").strip()
                    if not skill:
                        continue
                    cat = (row.get("category") or "Other").strip()
                    skills.append(skill)
                    categories[skill.lower()] = cat
            return skills, categories
        except Exception as e:
            logger.warning(f"Failed reading {path}: {e}")
    # Fallback
    logger.warning("Using fallback skills list (CSV missing or unreadable).")
    return FALLBACK_SKILLS, {s.lower(): "Other" for s in FALLBACK_SKILLS}

def try_load_spacy():
    global spacy_nlp
    try:
        import spacy
        try:
            spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
            logger.info("Loaded spaCy model en_core_web_sm.")
        except Exception:
            logger.warning("spaCy model 'en_core_web_sm' not available; using spacy.blank('en').")
            spacy_nlp = spacy.blank("en")
            if "sentencizer" not in spacy_nlp.pipe_names:
                spacy_nlp.add_pipe("sentencizer")
    except Exception as e:
        logger.exception(f"spaCy import failed: {e}")
        spacy_nlp = None

def try_init_phrase_matcher():
    """
    We keep PhraseMatcher optional. If spaCy is blank, it still works.
    """
    global phrase_matcher
    phrase_matcher = None
    try:
        if spacy_nlp is None:
            return
        from spacy.matcher import PhraseMatcher
        phrase_matcher = PhraseMatcher(spacy_nlp.vocab, attr="LOWER")
        patterns = [spacy_nlp.make_doc(skill) for skill in set(BASE_SKILLS)]
        if patterns:
            phrase_matcher.add("SKILLS", patterns)
        logger.info("PhraseMatcher initialized.")
    except Exception as e:
        logger.warning(f"PhraseMatcher init failed: {e}")
        phrase_matcher = None

def try_load_sentence_transformer():
    global embed_model
    embed_model = None
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("SentenceTransformer loaded.")
    except Exception as e:
        logger.warning(f"SentenceTransformer load failed; will use fallback similarity. Error: {e}")
        embed_model = None

# =========================
# STARTUP LOADING
# =========================
BASE_SKILLS, SKILL_CATEGORIES = load_skills_from_csv(SKILLS_FILE)

# Precompute lowercase variants for quick checks
BASE_SKILLS_LOWER = [s.lower() for s in BASE_SKILLS]
BASE_SKILLS_BY_LEN = sorted(BASE_SKILLS, key=lambda x: len(x), reverse=True)  # Multi-word first

# =========================
# EXTRACTION HELPERS
# =========================
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s\-/+\.]", " ", (text or "").lower()).strip()

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except Exception:
        # Fallback
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(b))
            return "\n".join([p.extract_text() or "" for p in reader.pages if p.extract_text()])
        except Exception:
            return ""

def extract_text_from_docx_bytes(b: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
        tf.write(b)
        tmpname = tf.name
    try:
        return docx2txt.process(tmpname) or ""
    finally:
        try:
            os.unlink(tmpname)
        except Exception:
            pass

async def extract_text_from_upload(file: UploadFile) -> str:
    content = await file.read()
    fname = (file.filename or "").lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    elif fname.endswith((".docx", ".doc")):
        return extract_text_from_docx_bytes(content)
    else:
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

# =========================
# SKILL EXTRACTION
#   (spaCy PhraseMatcher if available; else pure substring scan)
# =========================
def extract_skills_from_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    txt_norm = normalize(text)

    found = set()

    # If PhraseMatcher is available, use it first
    if spacy_nlp is not None and phrase_matcher is not None:
        try:
            doc = spacy_nlp(txt_norm)
            matches = phrase_matcher(doc)
            for _, start, end in matches:
                span = doc[start:end].text.strip()
                if span:
                    found.add(canonicalize(span))
        except Exception as e:
            logger.warning(f"PhraseMatcher use failed, falling back to substring scan: {e}")

    # Fallback / supplement: pure substring scan for multi-word skills
    for skill in BASE_SKILLS_BY_LEN:
        s_low = skill.lower()
        # Quick containment check
        if len(s_low.split()) > 1:
            if s_low in txt_norm:
                found.add(canonicalize(skill))

    # Token-based check for single words
    tokens = set(txt_norm.split())
    for base in BASE_SKILLS_LOWER:
        if " " not in base and base in tokens:
            # map back to original canonical
            orig = next((s for s in BASE_SKILLS if s.lower() == base), base)
            found.add(canonicalize(orig))

    return sorted(found)

# =========================
# PROFICIENCY (lightweight)
# =========================
def infer_proficiency(text: str, skill: str) -> Dict[str, Any]:
    snippet = (text or "").lower()
    years = None
    pattern = rf"(\d+(\.\d+)?)\s+years?.{{0,30}}\b{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\b.{{0,30}}(\d+(\.\d+)?)\s+years?"
    m = re.search(pattern, snippet)
    if m:
        for group in m.groups():
            if group and re.match(r"^\d+(\.\d+)?$", str(group)):
                try:
                    years = float(group)
                    break
                except Exception:
                    pass
    level = "unknown"
    if years is not None:
        level = "advanced" if years >= 3 else "intermediate" if years >= 1 else "beginner"
    return {"skill": skill, "years": years, "level": level}

# =========================
# SIMILARITY
#   Use SentenceTransformer if available; else Jaccard fallback.
# =========================
def cosine_sim_fallback(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def skills_similarity(resume_skills: List[str], job_skills: List[str]) -> float:
    global embed_model
    # SentenceTransformers path
    if embed_model is not None:
        try:
            from sentence_transformers.util import cosine_similarity
            resume_vec = embed_model.encode(" ".join(resume_skills), convert_to_tensor=True)
            job_vec = embed_model.encode(" ".join(job_skills), convert_to_tensor=True)
            sim = float(cosine_similarity(resume_vec, job_vec)[0][0])
            return max(0.0, min(1.0, sim))
        except Exception as e:
            logger.warning(f"ST similarity failed, falling back. Err: {e}")
    # Fallback: Jaccard on lowercased sets
    return cosine_sim_fallback(set(s.lower() for s in resume_skills), set(s.lower() for s in job_skills))

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="SkillBridge API - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _warm_start():
    """
    Warm everything we can, but never crash the app if something fails.
    """
    try:
        try_load_spacy()
    except Exception:
        pass
    try:
        try_init_phrase_matcher()
    except Exception:
        pass
    try:
        try_load_sentence_transformer()
    except Exception:
        pass
    logger.info("Startup warmup complete.")

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {"status": "ok", "service": "skillbridge_optimized"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/parse/resume")
async def parse_resume(file: UploadFile = File(...)):
    txt = await extract_text_from_upload(file)
    skills = extract_skills_from_text(txt)
    inferred = {s: infer_proficiency(txt, s) for s in skills}
    return {
        "resume_id": "resume_demo",
        "parsed_text_preview": txt[:2000],
        "skills": skills,
        "inferred": inferred
    }

@app.post("/parse/job", response_model=ParseJobOut)
async def parse_job(payload: ParseJobIn):
    return {
        "job_id": "job_demo",
        "parsed_skills": extract_skills_from_text(payload.text)
    }

@app.post("/analyze/gap", response_model=GapOut)
async def analyze_gap(payload: GapAnalysisRequest):
    try:
        resume_skills = extract_skills_from_text(payload.resume_text)
        job_skills = extract_skills_from_text(payload.job_text)

        if not job_skills:
            return {"matched": [], "missing": [], "suggested_plan": {}}

        # similarity per-skill using embeddings if available; else string overlap
        matched: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []

        # Precompute list-level similarity as a baseline
        list_sim = skills_similarity(resume_skills, job_skills)

        # For per-skill similarity, try embeddings; else fallback
        per_skill_scores: Dict[str, float] = {}
        if embed_model is not None and resume_skills:
            try:
                from sentence_transformers import util as st_util
                rs_emb = embed_model.encode(resume_skills, convert_to_tensor=True)  # (R, D)
                for j in job_skills:
                    j_emb = embed_model.encode([j], convert_to_tensor=True)  # (1, D)
                    # cosine similarity vs all resume skills, take max
                    sims = st_util.cos_sim(j_emb, rs_emb)  # (1, R)
                    best = float(np.max(sims.cpu().numpy()))
                    per_skill_scores[j] = best
            except Exception as e:
                logger.warning(f"Per-skill embedding similarity failed, fallback. Err: {e}")
                per_skill_scores = {}
        # Fallback per-skill: token overlap
        if not per_skill_scores:
            r_tokens = [set(normalize(s).split()) for s in resume_skills] or [set()]
            for j in job_skills:
                j_tok = set(normalize(j).split())
                best = 0.0
                for rt in r_tokens:
                    best = max(best, cosine_sim_fallback(j_tok, rt))
                per_skill_scores[j] = best

        # Thresholds
        MATCH_THRESH = 0.65

        for j in job_skills:
            direct = any(j.lower() == r.lower() for r in resume_skills)
            best_score = per_skill_scores.get(j, 0.0)
            if direct or best_score >= MATCH_THRESH:
                matched.append({
                    "skill": j,
                    "status": "matched",
                    "score": round(best_score, 3),
                    "reason": "Found in resume or semantically similar",
                    "inferred": infer_proficiency(payload.resume_text, j),
                    "category": SKILL_CATEGORIES.get(j.lower(), "Other")
                })
            else:
                missing.append({
                    "skill": j,
                    "status": "missing",
                    "score": round(1 - best_score, 3),
                    "reason": "Not found or low semantic similarity",
                    "inferred": {},
                    "category": SKILL_CATEGORIES.get(j.lower(), "Other")
                })

        # Suggested plan buckets (based on how far from match a skill is)
        suggested_plan: Dict[str, List[Dict[str, str]]] = {"30_days": [], "60_days": [], "90_days": []}
        for m in sorted(missing, key=lambda x: -x["score"]):
            if m["score"] >= 0.8:
                suggested_plan["30_days"].append({"skill": m["skill"], "task": f"Hands-on project in {m['skill']}"})
            elif m["score"] >= 0.6:
                suggested_plan["60_days"].append({"skill": m["skill"], "task": f"Learn basics of {m['skill']}"})
            else:
                suggested_plan["90_days"].append({"skill": m["skill"], "task": f"Explore intermediate {m['skill']}"})

        # Return the structure your frontend expects
        return {"matched": matched, "missing": missing, "suggested_plan": suggested_plan}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/analyze/gap error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# DEV SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
