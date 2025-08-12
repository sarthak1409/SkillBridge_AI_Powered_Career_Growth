import streamlit as st
import requests
import pandas as pd
from PIL import Image
import random
import os  # Added for path handling

st.set_page_config(page_title="SkillBridge", page_icon="🚀", layout="wide")
API_URL = "https://skillbridge-ai-powered-career-growth-1.onrender.com"

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            background-size: 400% 400%;
            animation: gradientBG 18s ease infinite;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; margin: auto;}
        .main-title {font-size: 55px !important; font-weight: 900 !important; text-align: center; color: #ffffff !important; white-space: nowrap;}
        .subheading {font-size: 26px !important; font-weight: 700 !important; color: #f8f8f8 !important; margin-bottom: 8px;}
        .subheading-line {border: none; height: 3px; background: linear-gradient(90deg, #4facfe, #00f2fe, #43e97b, #38f9d7);
            background-size: 300% 300%; animation: lineGradient 5s ease infinite; margin-bottom: 15px; border-radius: 2px;}
        @keyframes lineGradient {0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; }}
        label, .stTextInput label, .stTextArea label, .stFileUploader label {font-size: 20px !important; font-weight: 700 !important; color: #ffffff !important;}
        p, div, span {font-size: 16px !important; line-height: 1.5;}
        .stTextInput>div>div>input, .stTextArea textarea {background-color: #2a3b5f !important; color: white !important; border: 1px solid #4a6fa5 !important;
            border-radius: 12px !important; padding: 12px !important; font-size: 16px !important;}
        .stTextInput>div>div>input::placeholder, .stTextArea textarea::placeholder {color: rgba(255,255,255,0.6) !important;}
        .stTextInput>div>div>input:focus, .stTextArea textarea:focus {border: 1px solid white !important; outline: none !important;}
        [data-testid="stFileUploader"] section {background-color: #2a3b5f !important; border-radius: 12px !important; border: 1px solid #4a6fa5 !important; padding: 12px;}
        .stFileUploader>div>div>div>button {background-color: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.3) !important;
            color: white !important; border-radius: 12px !important; font-size: 16px !important;}
        .stFileUploader>div>div>div>button:hover {border: 1px solid white !important; background-color: rgba(255,255,255,0.1) !important;}
        .stButton>button {background: linear-gradient(90deg, #4facfe, #00f2fe); color: white; border-radius: 15px; padding: 14px 28px; font-size: 20px !important;
            font-weight: bold; border: none; transition: all 0.3s ease; box-shadow: 0px 4px 15px rgba(0,0,0,0.3);}
        .stButton>button:hover {background: linear-gradient(90deg, #43e97b, #38f9d7); box-shadow: 0px 0px 25px rgba(255,255,255,0.8); transform: scale(1.07);}
        .severity-high {color: white; background-color: #d9534f; padding: 6px 12px; border-radius: 6px; font-size: 16px;}
        .severity-medium {color: white; background-color: #f0ad4e; padding: 6px 12px; border-radius: 6px; font-size: 16px;}
        .severity-low {color: white; background-color: #5cb85c; padding: 6px 12px; border-radius: 6px; font-size: 16px;}
        .skill-item {font-size: 18px !important; padding: 6px; margin: 4px 0; transition: all 0.2s ease-in-out; display: flex; align-items:center;}
        .skill-item:hover {background-color: rgba(255,255,255,0.15); border-radius: 5px; padding: 6px;}
        .stTabs [role="tablist"] button {font-weight: bold; font-size: 18px !important; color: white !important;}
        .progress-bar {
            height: 18px;
            background-color: #4facfe;
            border-radius: 9px;
            transition: width 0.5s ease-in-out;
        }
        .progress-container {
            background-color: #203a43;
            border-radius: 9px;
            width: 100%;
            margin: 10px 0;
        }
        .quote {
            font-style: italic;
            font-size: 14px;
            color: #bbb;
            margin-top: 10px;
            text-align: center;
            padding: 0 12px;
        }
        .checkbox-task {
            font-size: 16px;
            margin-bottom: 6px;
        }
        .skill-proficiency {
            height: 12px;
            background-color: #38f9d7;
            border-radius: 6px;
            margin-left: 10px;
            flex-grow: 1;
            max-width: 150px;
        }
        .skill-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 4px 0;
        }
        .severity-badge {
            padding: 5px 10px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            min-width: 90px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Robust path to logo image relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '..', 'images', 'resume.png')
    logo_img = Image.open(image_path)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_img, width=150)
    st.markdown("<h2 style='text-align:center; color:white; margin-bottom: 10px;'>🚀 SkillBridge</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:white; text-align:center; font-weight:bold; margin-bottom: 5px;'>Resume Match Progress</div>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    quotes = [
        "“The future depends on what you do today.” – Mahatma Gandhi",
        "“Success usually comes to those who are too busy to be looking for it.” – Henry David Thoreau",
        "“Don’t watch the clock; do what it does. Keep going.” – Sam Levenson",
        "“The only way to do great work is to love what you do.” – Steve Jobs",
        "“Your limitation—it’s only your imagination.”"
    ]
    quote = random.choice(quotes)
    st.markdown(f"<div class='quote'>{quote}</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <ul style='color:#ddd; font-size: 15px; padding-left: 20px; line-height: 1.6; margin-top: 25px;'>
            <li>Analyze your resume against any job description</li>
            <li>Find missing skills and weaknesses</li>
            <li>Receive a personalized 30/60/90 day learning plan</li>
            <li>Simple, fast, and AI-powered</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color: #444; margin-top: 25px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; margin-bottom: 8px;'>📌 How to Use</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <ol style='color:#ddd; font-size: 15px; padding-left: 20px; line-height: 1.6;'>
            <li>Upload your resume (PDF, DOCX, or TXT)</li>
            <li>Paste the job description</li>
            <li>Click Analyze</li>
            <li>Explore your skill gaps and learning plan</li>
        </ol>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color: #444; margin-top: 25px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='color:#888; font-size: 12px; text-align: center; margin-top: 20px;'>
        Made with ❤️ by <b>Sarthak Maddi</b>
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div class='main-title'>🚀 SkillBridge – AI-Powered Career Growth</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white; font-size:24px;'>Upload your resume & job description to see matching skills, gaps, and a personalized learning plan.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='subheading'>📄 Upload Resume</div><hr class='subheading-line'>", unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], label_visibility="collapsed")
with col2:
    st.markdown("<div class='subheading'>📝 Paste Job Description</div><hr class='subheading-line'>", unsafe_allow_html=True)
    job_text_input = st.text_area("Paste Job Description", height=100, label_visibility="collapsed", placeholder="Paste Here Your Preferred Job Description")

st.markdown("<div style='display:flex; justify-content:center; margin-top:20px;'>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Skill Gaps")
st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn and resume_file and job_text_input:
    with st.spinner("⏳ Extracting skills and analyzing gaps..."):
        try:
            resume_resp = requests.post(f"{API_URL}/parse/resume", files={"file": resume_file})
            resume_data = resume_resp.json()
            job_resp = requests.post(f"{API_URL}/parse/job", json={"text": job_text_input})
            job_data = job_resp.json()
            gap_resp = requests.post(f"{API_URL}/analyze/gap", json={
                "resume_text": resume_data["parsed_text_preview"],
                "job_text": job_text_input
            })
            gap_data = gap_resp.json()
            match_score = gap_data.get("match_score", None)
            if match_score is not None:
                match_percent = int(match_score * 100)
            else:
                total_skills = len(resume_data.get("skills", [])) + len(gap_data.get("missing", []))
                matched_skills = len(resume_data.get("skills", []))
                match_percent = int((matched_skills / total_skills) * 100) if total_skills > 0 else 0
            progress_bar.progress(match_percent)
        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")
            st.stop()

    tab1, tab2, tab3 = st.tabs(["📌 Resume Skills", "⚠ Missing / Weak Skills", "📅 Learning Plan"])

    with tab1:
        st.markdown("<div class='subheading'>Extracted Skills from Resume</div><hr class='subheading-line'>", unsafe_allow_html=True)
        skills = resume_data.get("skills", [])
        for skill in skills:
            proficiency = random.uniform(0.5, 1.0)
            bar_width = int(proficiency * 100)
            st.markdown(
                f"""
                <div class="skill-row">
                    <div>✅ {skill}</div>
                    <div class="skill-proficiency" style="width:{bar_width}px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with st.expander("🔍 Preview Resume Text"):
            st.text(resume_data.get("parsed_text_preview", ""))

    with tab2:
        st.markdown("<div class='subheading'>Missing / Weak Skills</div><hr class='subheading-line'>", unsafe_allow_html=True)
        df_missing = pd.DataFrame(gap_data.get("missing", []))
        if not df_missing.empty:
            severity_colors = {"high": "#d9534f", "medium": "#f0ad4e", "low": "#5cb85c"}
            for _, row in df_missing.iterrows():
                sev = "high" if row["score"] >= 0.8 else ("medium" if row["score"] >= 0.6 else "low")
                color = severity_colors[sev]
                st.markdown(
                    f"""
                    <div class='skill-item' style='justify-content:space-between;'>
                        <div>⚠ <b>{row['skill']}</b> – {row['reason']}</div>
                        <div class='severity-badge' style='background-color:{color};'>Severity: {row['score']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("🎉 No missing or weak skills found!")

    with tab3:
        st.markdown("<div class='subheading'>Personalized 30 / 60 / 90 Day Learning Plan</div><hr class='subheading-line'>", unsafe_allow_html=True)
        for period in ["30_days", "60_days", "90_days"]:
            tasks = gap_data.get("suggested_plan", {}).get(period, [])
            if tasks:
                st.markdown(f"### ⏳ {period.replace('_', ' ').title()}")
                for idx, task in enumerate(tasks):
                    checkbox_key = f"{period}_task_{idx}"
                    st.checkbox(f"📘 **{task['skill']}** → {task['task']}", key=checkbox_key, value=False)
            else:
                st.markdown(f"✅ No tasks for {period.replace('_', ' ').title()}")
else:
    st.info("📌 Please upload a resume and paste a job description to begin.")



