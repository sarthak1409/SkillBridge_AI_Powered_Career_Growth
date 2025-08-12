from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import io
import re
import pdfplumber
import docx2txt
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.matcher import PhraseMatcher
import csv
from pathlib import Path
import os
import uvicorn
from fastapi import FastAPI


app = FastAPI(title="SkillBridge API - Improved Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

nlp = spacy.load("en_core_web_sm")

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

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(s) for s in set(BASE_SKILLS)]
phrase_matcher.add("SKILLS", patterns)

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(b))
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
            tf.write(b)
            tmpname = tf.name
        text = docx2txt.process(tmpname) or ""
        os.unlink(tmpname)
        return text
    except Exception:
        return ""

def extract_text_from_upload(file: UploadFile) -> str:
    content = b""
    try:
        content = file.file.read()
    finally:
        file.file.seek(0)
    fname = file.filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        return extract_text_from_docx_bytes(content)
    else:
        try:
            return content.decode("utf-8")
        except Exception:
            return ""

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s\-/+\.]", " ", text.lower()).strip()

def extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    text = normalize(text)
    doc = nlp(text)
    found = set()
    matches = phrase_matcher(doc)
    for _, start, end in matches:
        span = doc[start:end].text.strip()
        found.add(canonicalize(span))
    for ent in doc.ents:
        ent_text = ent.text.strip()
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "NORP"):
            if len(ent_text) <= 40 and re.search(r"[A-Za-z]", ent_text):
                if ent_text.lower() in BASE_SKILLS_LOWER:
                    found.add(canonicalize(ent_text))
    tokens = set([t.text.lower() for t in doc if not t.is_stop])
    for base in BASE_SKILLS_LOWER:
        if base in tokens:
            orig = next((s for s in BASE_SKILLS if s.lower() == base), base)
            found.add(canonicalize(orig))
    tech_pattern = re.compile(r"\b(" + "|".join([re.escape(s) for s in BASE_SKILLS_LOWER]) + r")\b", re.IGNORECASE)
    for m in tech_pattern.finditer(text):
        matched = m.group(0)
        orig = next((s for s in BASE_SKILLS if s.lower() == matched.lower()), matched)
        found.add(canonicalize(orig))
    return sorted(found)

def infer_proficiency(text: str, skill: str) -> Dict[str, Any]:
    snippet = text.lower()
    years = None
    pattern = rf"(\d+(\.\d+)?)\s+years?.{{0,30}}\b{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\b.{{0,30}}(\d+(\.\d+)?)\s+years?"
    m = re.search(pattern, snippet)
    if m:
        for group in m.groups():
            if group and re.match(r"^\d+(\.\d+)?$", str(group)):
                try:
                    years = float(group)
                    break
                except:
                    pass
    adv_words = ["expert", "lead", "senior", "extensive experience", "strong experience", "3+ years", "more than"]
    beg_words = ["familiar with", "basic", "worked on", "introductory", "intern"]
    level = "unknown"
    for w in adv_words:
        if w in snippet and skill.lower() in snippet:
            level = "advanced"
            break
    if level == "unknown":
        for w in beg_words:
            if w in snippet and skill.lower() in snippet:
                level = "beginner"
                break
    if level == "unknown" and years:
        level = "advanced" if years >= 3 else ("intermediate" if years >= 1 else "beginner")
    return {"skill": skill, "years": years, "level": level}

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, embed_model.get_sentence_embedding_dimension()))
    return embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

class ParseJobIn(BaseModel):
    text: str

class ParseJobOut(BaseModel):
    job_id: str = "job_demo"
    parsed_skills: List[str]

class GapIn(BaseModel):
    resume_text: str
    job_text: str

class SkillGapItem(BaseModel):
    skill: str
    status: str
    score: float
    reason: str
    inferred: Dict[str, Any] = {}
    category: str = "Other"

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
    skills = extract_skills_from_text(payload.text)
    return {"job_id": "job_demo", "parsed_skills": skills}

@app.post("/analyze/gap", response_model=GapOut)
async def analyze_gap(payload: GapIn):
    resume_txt = payload.resume_text
    job_txt = payload.job_text
    resume_skills = extract_skills_from_text(resume_txt)
    job_skills = extract_skills_from_text(job_txt)
    resume_emb = embed_texts(resume_skills)
    matched = []
    missing = []
    for jskill in job_skills:
        j_emb = embed_texts([jskill])
        best_score = 0.0
        if len(resume_skills) > 0:
            sims = util.cos_sim(j_emb, resume_emb).numpy().flatten()
            best_score = float(np.max(sims))
        direct_present = any(jskill.lower() == rs.lower() for rs in resume_skills)
        if direct_present or best_score >= 0.65:
            inferred = infer_proficiency(resume_txt, jskill)
            matched.append({
                "skill": jskill,
                "status": "matched",
                "score": round(best_score, 3),
                "reason": "Found in resume or semantically similar",
                "inferred": inferred,
                "category": SKILL_CATEGORIES.get(jskill.lower(), "Other")
            })
        else:
            severity = round(1 - best_score, 3)
            missing.append({
                "skill": jskill,
                "status": "missing",
                "score": severity,
                "reason": "Not found or low semantic similarity",
                "inferred": {},
                "category": SKILL_CATEGORIES.get(jskill.lower(), "Other")
            })
    suggested_plan = {"30_days": [], "60_days": [], "90_days": []}
    for m in sorted(missing, key=lambda x: -x["score"]):
        if m["score"] >= 0.8:
            suggested_plan["30_days"].append({"skill": m["skill"], "task": f"Hands-on: Build a small project using {m['skill']} (1-3 days)"})
        elif m["score"] >= 0.6:
            suggested_plan["60_days"].append({"skill": m["skill"], "task": f"Learn fundamentals of {m['skill']} and create a demo (3-7 days)"})
        else:
            suggested_plan["90_days"].append({"skill": m["skill"], "task": f"Explore intermediate topics in {m['skill']} and add to portfolio (1-2 weeks)"})
    return {"matched": matched, "missing": missing, "suggested_plan": suggested_plan}

@app.get("/")
def read_root():
    return {"status": "ok", "service": "skillbridge_improved"}
    
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


