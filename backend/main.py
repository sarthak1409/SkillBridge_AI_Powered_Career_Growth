from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import io
import re
import pdfplumber
import docx2txt
import numpy as np
import csv
from pathlib import Path
import os
import uvicorn
import logging
from fastapi.responses import JSONResponse
import traceback
from sentence_transformers import util
from fastapi import HTTPException

# Global placeholders
embed_model = None
nlp = None
phrase_matcher = None

# Lazy load SentenceTransformer
def get_embed_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        logging.info("Loading SentenceTransformer model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # CPU only for Render
        logging.info("SentenceTransformer loaded successfully")
    return embed_model

# Lazy load spaCy
def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        logging.info("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])  # Lighter load
        logging.info("spaCy model loaded successfully")
    return nlp

# Lazy load PhraseMatcher
def get_phrase_matcher():
    global phrase_matcher
    if phrase_matcher is None:
        from spacy.matcher import PhraseMatcher
        logging.info("Initializing PhraseMatcher...")
        nlp_local = get_nlp()
        phrase_matcher = PhraseMatcher(nlp_local.vocab, attr="LOWER")
        # BASE_SKILLS must be defined before this call
        patterns = [nlp_local.make_doc(skill) for skill in set(BASE_SKILLS)]
        phrase_matcher.add("SKILLS", patterns)
        logging.info("PhraseMatcher initialized successfully")
    return phrase_matcher

def get_embedding_model():
    """Wrapper for old function name"""
    return get_embed_model()

# ======================
# Load skills from CSV
# ======================
SKILLS_FILE = Path(__file__).parent / "skills" / "unique_skills_dataset.csv"

def load_skills_from_csv(path: Path):
    skills = []
    categories = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill = row["skill"].strip()
            cat = row["category"].strip()
            skills.append(skill)
            categories[skill.lower()] = cat
    return skills, categories

BASE_SKILLS, SKILL_CATEGORIES = load_skills_from_csv(SKILLS_FILE)
BASE_SKILLS_LOWER = [s.lower() for s in BASE_SKILLS]

# Synonyms (same as before, unchanged for brevity)
SYNONYM_MAP = {
    'oracle db': 'Oracle DB',
    'flink': 'Flink',
    'fedora': 'Fedora',
    'github': 'GitHub',
    'azure': 'Azure',
    'aws': 'AWS',
    'photoshop': 'Photoshop',
    'heroku': 'Heroku',
    'pandas': 'Pandas',
    'python': 'Python',
    'scikit-learn': 'Scikit-learn',
    'redis': 'Redis',
    'curriculum design': 'Curriculum Design',
    'seo': 'SEO',
    'communication': 'Communication',
    'go': 'Go',
    'teamwork': 'Teamwork',
    'kubernetes': 'Kubernetes',
    'excel': 'Excel',
    'mysql': 'MySQL',
    'scrum': 'Scrum',
    'react': 'React',
    'xgboost': 'XGBoost',
    'opencv': 'OpenCV',
    'ibm cloud': 'IBM Cloud',
    'e-learning tools': 'E-learning Tools',
    'tableau': 'Tableau',
    'nltk': 'NLTK',
    'terraform': 'Terraform',
    'swift': 'Swift',
    'tax planning': 'Tax Planning',
    'numpy': 'NumPy',
    'kanban': 'Kanban',
    'django': 'Django',
    'mongodb': 'MongoDB',
    'airflow': 'Airflow',
    'ruby': 'Ruby',
    'time management': 'Time Management',
    'hadoop': 'Hadoop',
    'dvc': 'DVC',
    'matplotlib': 'Matplotlib',
    'illustrator': 'Illustrator',
    'mlops': 'MLOps',
    'php': 'PHP',
    'digitalocean': 'DigitalOcean',
    'spacy': 'spaCy',
    'postgresql': 'PostgreSQL',
    'tensorflow': 'TensorFlow',
    'financial analysis': 'Financial Analysis',
    'sql': 'SQL',
    'leadership': 'Leadership',
    'bitbucket': 'Bitbucket',
    'google sheets': 'Google Sheets',
    'catboost': 'CatBoost',
    'javascript': 'JavaScript',
    'seaborn': 'Seaborn',
    'ansible': 'Ansible',
    'typescript': 'TypeScript',
    'problem solving': 'Problem Solving',
    'rust': 'Rust',
    'spark': 'Spark',
    'mlflow': 'MLflow',
    'market research': 'Market Research',
    'angular': 'Angular',
    'teaching': 'Teaching',
    'vue.js': 'Vue.js',
    'fastapi': 'FastAPI',
    'yolo': 'YOLO',
    'git': 'Git',
    'mediapipe': 'MediaPipe',
    'ubuntu': 'Ubuntu',
    'agile': 'Agile',
    'accounting': 'Accounting',
    'gcp': 'GCP',
    'linux': 'Linux',
    'gitlab': 'GitLab',
    'email marketing': 'Email Marketing',
    'next.js': 'Next.js',
    'gpt': 'GPT',
    'docker': 'Docker',
    'public speaking': 'Public Speaking',
    'windows': 'Windows',
    'kafka': 'Kafka',
    'sketch': 'Sketch',
    'jenkins': 'Jenkins',
    'sqlite': 'SQLite',
    'java': 'Java',
    'streamlit': 'Streamlit',
    'keras': 'Keras',
    'c++': 'C++',
    'lightgbm': 'LightGBM',
    'social media marketing': 'Social Media Marketing',
    'canva': 'Canva',
    'bert': 'BERT',
    'content writing': 'Content Writing',
    'budgeting': 'Budgeting',
    'pytorch': 'PyTorch',
    'macos': 'MacOS',
    'transformers': 'Transformers',
    'power bi': 'Power BI',
    'figma': 'Figma',
    'flask': 'Flask',
    'pmp certification': 'PMP Certification',
    'sklearn': 'Scikit-learn',
    'tf': 'TensorFlow',
    'postgres': 'PostgreSQL',
    'rest api': 'REST API',
    'api': 'REST API',
    'nlp': 'NLP',
    'ps': 'Photoshop',
    'ai': 'Artificial Intelligence',
    'smm': 'Social Media Marketing'
}


def canonicalize(skill: str) -> str:
    s = skill.lower().strip()
    return SYNONYM_MAP.get(s, skill.strip().title())

# ======================
# Text Extraction
# ======================
def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except Exception:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(b))
        return "\n".join([p.extract_text() or "" for p in reader.pages if p.extract_text()])

def extract_text_from_docx_bytes(b: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
        tf.write(b)
        tmpname = tf.name
    try:
        return docx2txt.process(tmpname) or ""
    finally:
        os.unlink(tmpname)

def extract_text_from_upload(file: UploadFile) -> str:
    content = file.file.read()
    file.file.seek(0)
    fname = file.filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    elif fname.endswith((".docx", ".doc")):
        return extract_text_from_docx_bytes(content)
    else:
        try:
            return content.decode("utf-8")
        except Exception:
            return ""

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s\-/+\.]", " ", text.lower()).strip()

# ======================
# Skill Extraction
# ======================
def extract_skills_from_text(text: str) -> list:
    """
    Extracts skills from the given text using the preloaded spaCy model
    and PhraseMatcher patterns.
    """
    try:
        if not text or not text.strip():
            return []

        # Get preloaded NLP and PhraseMatcher
        nlp_instance = get_nlp()
        phrase_matcher_instance = get_phrase_matcher()

        # Process text
        doc = nlp_instance(text)
        matches = phrase_matcher_instance(doc)

        # Extract and clean skills
        skills_found = {doc[start:end].text.strip() for match_id, start, end in matches}

        return sorted(skills_found, key=str.lower)

    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        return []

def infer_proficiency(text: str, skill: str) -> Dict[str, Any]:
    snippet = text.lower()
    years = None
    pattern = rf"(\d+(\.\d+)?)\s+years?.{{0,30}}\b{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\b.{{0,30}}(\d+(\.\d+)?)\s+years?"
    m = re.search(pattern, snippet)
    if m:
        for group in m.groups():
            if group and re.match(r"^\d+(\.\d+)?$", str(group)):
                years = float(group)
                break
    level = "unknown"
    if years is not None:
        level = "advanced" if years >= 3 else "intermediate" if years >= 1 else "beginner"
    return {"skill": skill, "years": years, "level": level}

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embed_model()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# ======================
# FastAPI App
# ======================
app = FastAPI(title="SkillBridge API - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ParseJobIn(BaseModel):
    text: str

class ParseJobOut(BaseModel):
    job_id: str
    parsed_skills: List[str]

class GapIn(BaseModel):
    resume_text: str
    job_text: str

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

@app.post("/parse/resume")
async def parse_resume(file: UploadFile = File(...)):
    txt = extract_text_from_upload(file)
    skills = extract_skills_from_text(txt)
    inferred = {s: infer_proficiency(txt, s) for s in skills}
    return {"resume_id": "resume_demo", "parsed_text_preview": txt[:2000], "skills": skills, "inferred": inferred}

@app.post("/parse/job", response_model=ParseJobOut)
async def parse_job(payload: ParseJobIn):
    return {"job_id": "job_demo", "parsed_skills": extract_skills_from_text(payload.text)}

from fastapi.responses import JSONResponse
import traceback

@app.post("/analyze/gap")
async def analyze_gap(payload: GapAnalysisRequest):
    try:
        # Load models lazily
        model_instance = get_embed_model()
        nlp_instance = get_nlp()
        phrase_matcher_instance = get_phrase_matcher()

        # Extract skills
        resume_skills = extract_skills_from_text(payload.resume_text)
        job_skills = extract_skills_from_text(payload.job_text)

        # Quick logging for debug
        logging.info(f"Resume skills found: {resume_skills}")
        logging.info(f"Job skills found: {job_skills}")

        if not resume_skills or not job_skills:
            raise HTTPException(status_code=400, detail="Could not extract skills from resume or job description.")

        # Find missing skills
        missing_skills = list(set(job_skills) - set(resume_skills))

        # Compute similarity score
        resume_embedding = model_instance.encode(" ".join(resume_skills), convert_to_tensor=True)
        job_embedding = model_instance.encode(" ".join(job_skills), convert_to_tensor=True)

        from sentence_transformers.util import cosine_similarity
        similarity_score = float(cosine_similarity(resume_embedding, job_embedding)[0][0])

        return {
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "missing_skills": missing_skills,
            "similarity_score": round(similarity_score, 3),
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("Error in /analyze/gap")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "service": "skillbridge_optimized"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))



