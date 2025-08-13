from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from functools import lru_cache
import logging
import traceback
import csv
import re
import io
import os
import numpy as np
import pdfplumber
import docx2txt

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skillbridge")

# =========================
# CONSTANTS / FILES
# =========================
HERE = Path(__file__).parent
SKILLS_FILE = HERE / "skills" / "unique_skills_dataset.csv"

FALLBACK_SKILLS = [
    "Python", "JavaScript", "SQL", "React", "Django",
    "FastAPI", "Machine Learning", "Docker", "Kubernetes", "AWS",
    "Data Analysis", "NLP", "Pandas", "NumPy", "Git", "REST API"
]

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
# LOAD SKILLS
# =========================
def load_skills_from_csv(path: Path):
    skills, categories = [], {}
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
    logger.warning("Using fallback skills list.")
    return FALLBACK_SKILLS, {s.lower(): "Other" for s in FALLBACK_SKILLS}

BASE_SKILLS, SKILL_CATEGORIES = load_skills_from_csv(SKILLS_FILE)
BASE_SKILLS_LOWER = [s.lower() for s in BASE_SKILLS]
BASE_SKILLS_BY_LEN = sorted(BASE_SKILLS, key=lambda x: len(x), reverse=True)

# =========================
# LAZY LOAD MODELS
# =========================
@lru_cache(maxsize=1)
def get_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except Exception:
        logger.warning("spaCy model missing, using blank('en').")
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

@lru_cache(maxsize=1)
def get_phrase_matcher():
    from spacy.matcher import PhraseMatcher
    nlp = get_spacy()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in set(BASE_SKILLS)]
    matcher.add("SKILLS", patterns)
    return matcher

@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")

# =========================
# UTILS
# =========================
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s\-/+\.]", " ", (text or "").lower()).strip()

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
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
        return content.decode("utf-8", errors="ignore")

# =========================
# SKILL EXTRACTION
# =========================
def extract_skills_from_text(text: str) -> List[str]:
    if not text.strip():
        return []
    txt_norm = normalize(text)
    found = set()

    try:
        doc = get_spacy()(txt_norm)
        matches = get_phrase_matcher()(doc)
        for _, start, end in matches:
            span = doc[start:end].text.strip()
            if span:
                found.add(canonicalize(span))
    except Exception:
        pass

    for skill in BASE_SKILLS_BY_LEN:
        if len(skill.split()) > 1 and skill.lower() in txt_norm:
            found.add(canonicalize(skill))

    tokens = set(txt_norm.split())
    for base in BASE_SKILLS_LOWER:
        if " " not in base and base in tokens:
            orig = next((s for s in BASE_SKILLS if s.lower() == base), base)
            found.add(canonicalize(orig))

    return sorted(found)

# =========================
# PROFICIENCY
# =========================
def infer_proficiency(text: str, skill: str) -> Dict[str, Any]:
    snippet = text.lower()
    years = None
    pattern = rf"(\d+(\.\d+)?)\s+years?.{{0,30}}\b{re.escape(skill.lower())}\b"
    m = re.search(pattern, snippet)
    if m:
        try:
            years = float(m.group(1))
        except Exception:
            pass
    level = "unknown"
    if years is not None:
        level = "advanced" if years >= 3 else "intermediate" if years >= 1 else "beginner"
    return {"skill": skill, "years": years, "level": level}

# =========================
# SIMILARITY
# =========================
def cosine_sim_fallback(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def skills_similarity(resume_skills: List[str], job_skills: List[str]) -> float:
    try:
        from sentence_transformers.util import cosine_similarity
        model = get_embed_model()
        r_vec = model.encode(" ".join(resume_skills), convert_to_tensor=True)
        j_vec = model.encode(" ".join(job_skills), convert_to_tensor=True)
        return float(cosine_similarity(r_vec, j_vec)[0][0])
    except Exception:
        return cosine_sim_fallback(set(map(str.lower, resume_skills)), set(map(str.lower, job_skills)))

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="SkillBridge API - Lazy Loaded")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELS
class GapAnalysisRequest(BaseModel):
    resume_text: str
    job_text: str

class ParseJobIn(BaseModel):
    text: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/parse/resume")
async def parse_resume(file: UploadFile = File(...)):
    txt = await extract_text_from_upload(file)
    skills = extract_skills_from_text(txt)
    inferred = {s: infer_proficiency(txt, s) for s in skills}
    return {"skills": skills, "inferred": inferred}

@app.post("/parse/job")
async def parse_job(payload: ParseJobIn):
    return {"parsed_skills": extract_skills_from_text(payload.text)}

@app.post("/analyze/gap")
async def analyze_gap(payload: GapAnalysisRequest):
    try:
        resume_skills = extract_skills_from_text(payload.resume_text)
        job_skills = extract_skills_from_text(payload.job_text)

        matched, missing = [], []
        per_skill_scores = {}
        try:
            from sentence_transformers import util as st_util
            rs_emb = get_embed_model().encode(resume_skills, convert_to_tensor=True)
            for j in job_skills:
                j_emb = get_embed_model().encode([j], convert_to_tensor=True)
                sims = st_util.cos_sim(j_emb, rs_emb)
                per_skill_scores[j] = float(np.max(sims.cpu().numpy()))
        except Exception:
            r_tokens = [set(normalize(s).split()) for s in resume_skills] or [set()]
            for j in job_skills:
                j_tok = set(normalize(j).split())
                per_skill_scores[j] = max(cosine_sim_fallback(j_tok, rt) for rt in r_tokens)

        for j in job_skills:
            if j.lower() in [r.lower() for r in resume_skills] or per_skill_scores[j] >= 0.65:
                matched.append({
                    "skill": j,
                    "status": "matched",
                    "score": round(per_skill_scores[j], 3),
                    "inferred": infer_proficiency(payload.resume_text, j),
                    "category": SKILL_CATEGORIES.get(j.lower(), "Other")
                })
            else:
                missing.append({
                    "skill": j,
                    "status": "missing",
                    "score": round(1 - per_skill_scores[j], 3),
                    "inferred": {},
                    "category": SKILL_CATEGORIES.get(j.lower(), "Other")
                })

        return {"matched": matched, "missing": missing}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
