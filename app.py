"""
JobFit – AI-Powered Resume & Job Matching System
Main Streamlit Application
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Import utils
from utils.resume_parser import parse_resume
from utils.skill_extractor import extract_skills
from utils.job_matcher import match_resume_to_job
from utils.recommendations import generate_recommendations

# Streamlit Page Config
st.set_page_config(
    page_title="JobFit – AI-Powered Resume & Job Matching",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Predefined sample job descriptions for easy evaluation
SAMPLE_JOBS = {
    "-- Select a preset sample job description --": "",
    "Machine Learning & AI Engineer": """We are seeking a Machine Learning Engineer with strong hands-on experience in Python, PyTorch, TensorFlow, and Scikit-learn.
Requirements:
- Strong programming background in Python, C++, and SQL.
- Deep understanding of Machine Learning, Deep Learning, and NLP algorithms.
- Experience with Pandas, NumPy, and data manipulation.
- Experience deploying models using Docker, AWS, and CI/CD pipelines.
- Knowledge of Data Structures, Algorithms, and Object-Oriented Programming (OOP).
- Familiarity with Git, GitHub, and collaborative engineering workflows.
- Excellent communication skills and passion for building scalable AI applications.""",

    "Full-Stack Software Developer": """We are hiring a Full-Stack Developer to design and implement robust, scalable web applications.
Required Qualifications:
- Proficiency in JavaScript, TypeScript, Python, and SQL.
- Strong hands-on experience with React, HTML, CSS, and Tailwind CSS.
- Backend proficiency with Node.js, Express, and Django or FastAPI.
- Relational and NoSQL databases: PostgreSQL, MySQL, and MongoDB.
- Cloud deployment and containerization experience using Docker, Kubernetes, and AWS.
- Familiarity with REST APIs, Microservices, Git, and GitHub.
- Strong foundation in Data Structures, Algorithms, and System Design.""",

    "Data Analyst / BI Specialist": """Looking for a detail-oriented Data Analyst to turn raw data into actionable business intelligence.
Responsibilities & Skills:
- Advanced proficiency in SQL and Excel (complex formulas, Pivot Tables).
- Expertise with business intelligence tools: Power BI or Tableau.
- Strong Python skills for data extraction and analysis using Pandas and NumPy.
- Experience performing exploratory data analysis, Statistics, and Data Analysis.
- Experience working with cloud data warehouses like Snowflake or BigQuery.
- Familiarity with Git and Agile development methodologies.""",

    "DevOps & Cloud Engineer": """Seeking a DevOps & Cloud Engineer to automate deployments and maintain reliable infrastructure.
Key Requirements:
- Hands-on expertise in Cloud Platforms: AWS, Azure, or GCP.
- Deep proficiency with containerization and orchestration: Docker and Kubernetes.
- Infrastructure as Code (IaC) using Terraform.
- CI/CD automation pipelines using GitHub, GitLab CI, or Jenkins.
- Proficient in Linux administration and Bash/Python scripting.
- Solid understanding of networking, MySQL/PostgreSQL databases, and Git version control."""
}


def render_header():
    """Renders the top hero header."""
    st.markdown("""
    <div class="main-header">
        <h1>💼 JobFit</h1>
        <div class="subtitle">AI-Powered Resume & Job Matching System</div>
        <div class="tagline">"Know how well your resume fits your dream job."</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders sidebar navigation, settings, and job seeker advice."""
    with st.sidebar:
        st.markdown("### ⚙️ System Settings")
        
        env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        has_env_key = bool(env_api_key and env_api_key != "your_api_key_here")

        if has_env_key:
            st.success("🟢 Gemini AI: Active (from .env)")
            user_api_key = env_api_key
        else:
            st.info("🟡 Gemini AI: Offline Mode (Rule-Based)")
            with st.expander("🔑 Add Gemini API Key (Optional)"):
                manual_key = st.text_input(
                    "Enter Gemini API Key",
                    type="password",
                    help="Get an API key from Google AI Studio. If left blank, JobFit uses high-accuracy rule-based intelligence."
                )
                if manual_key.strip():
                    user_api_key = manual_key.strip()
                    st.success("Custom Gemini API Key set!")
                else:
                    user_api_key = None

        st.markdown("---")
        st.markdown("### 📌 How It Works")
        st.markdown("""
        1. **Upload Resume**: Upload your resume in **PDF** or **DOCX** format.
        2. **Provide Job Description**: Paste the target job posting or choose a sample.
        3. **NLP & AI Analysis**: JobFit extracts skills, detects sections, measures semantic similarity, and compares keyword coverage.
        4. **Get Recommendations**: View your match percentage, skill gap breakdown, and personalized recommendations.
        """)

        st.markdown("---")
        st.markdown("### 💡 Quick Job Seeker Tips")
        st.markdown("""
        - **ATS Compatibility**: Use clear section headers like *Skills*, *Experience*, and *Education*.
        - **Quantify Impact**: Use numbers, percentages, and metrics to show real results.
        - **Skill Alignment**: Highlight the exact technical tools mentioned in the job description.
        """)

        return user_api_key


def render_skill_badges(skills: list, badge_class: str, empty_msg: str):
    """Renders HTML skill badge chips."""
    if not skills:
        st.markdown(f"<span style='color: #64748b; font-size: 0.9rem;'>{empty_msg}</span>", unsafe_allow_html=True)
        return
    
    html = "".join([f"<span class='{badge_class}'>{skill}</span>" for skill in skills])
    st.markdown(html, unsafe_allow_html=True)


def create_match_donut_chart(matched_count: int, missing_count: int):
    """Creates a clean Altair donut chart comparing matched vs missing skills."""
    if matched_count == 0 and missing_count == 0:
        return None

    data = pd.DataFrame({
        "Status": ["Matching Skills", "Missing Skills"],
        "Count": [matched_count, missing_count],
        "Color": ["#10b981", "#ef4444"]
    })

    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=50, stroke="#ffffff", strokeWidth=2)
        .encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(
                field="Status",
                type="nominal",
                scale=alt.Scale(domain=["Matching Skills", "Missing Skills"], range=["#10b981", "#ef4444"]),
                legend=alt.Legend(orient="bottom", title=None)
            ),
            tooltip=["Status", "Count"]
        )
        .properties(height=240, title=alt.TitleParams("Skills Breakdown", anchor="middle"))
    )
    return chart


def create_category_bar_chart(category_summary: dict):
    """Creates an Altair grouped bar chart for skill categories."""
    records = []
    for cat, counts in category_summary.items():
        records.append({"Category": cat, "Type": "Matched", "Skills": counts["matched"]})
        records.append({"Category": cat, "Type": "Missing", "Skills": counts["missing"]})

    if not records:
        return None

    df = pd.DataFrame(records)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Category:N", title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Skills:Q", title="Number of Skills"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=["Matched", "Missing"], range=["#10b981", "#f87171"]),
                legend=alt.Legend(orient="top", title=None)
            ),
            xOffset="Type:N",
            tooltip=["Category", "Type", "Skills"]
        )
        .properties(height=260, title="Required Skills Coverage by Category")
    )
    return chart


def main():
    render_header()
    user_api_key = render_sidebar()

    st.markdown("""
    Welcome to **JobFit**! Upload your resume and paste any target job description to instantly analyze your compatibility, identify missing technical skills, and receive actionable suggestions to boost your interview chances.
    """)

    # Main Input Layout
    col_upload, col_job = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📄 1. Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file (PDF or DOCX)",
            type=["pdf", "docx"],
            help="Upload your resume in PDF or DOCX format. Scanned PDFs with no extractable text will trigger an OCR check.",
        )
        if uploaded_file:
            st.success(f"📎 Loaded: **{uploaded_file.name}** ({round(uploaded_file.size / 1024, 1)} KB)")

    with col_job:
        st.markdown("### 📋 2. Target Job Description")
        selected_sample = st.selectbox(
            "Quick Demo: Load a sample job description",
            options=list(SAMPLE_JOBS.keys()),
            index=0,
        )
        
        default_jd_text = SAMPLE_JOBS[selected_sample] if selected_sample else ""
        job_description_input = st.text_area(
            "Paste Job Description text here:",
            value=default_jd_text,
            height=200,
            placeholder="Paste the full job posting, required qualifications, and responsibilities...",
        )

    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_clicked = st.button("🚀 Analyze Resume & Match Job", use_container_width=True, type="primary")

    # Perform Analysis
    if analyze_clicked:
        # Input Validation
        if not uploaded_file:
            st.error("⚠️ Please upload a resume (PDF or DOCX) before running analysis.")
            return

        if not job_description_input.strip():
            st.error("⚠️ Please paste a job description or select a sample from the dropdown.")
            return

        with st.spinner("🔍 Reading resume, extracting skills, and calculating match metrics..."):
            file_bytes = uploaded_file.read()
            parsed_resume = parse_resume(file_bytes, uploaded_file.name)

            if not parsed_resume["success"]:
                st.error(f"❌ {parsed_resume['error']}")
                return

            # Perform Job Matching
            match_results = match_resume_to_job(
                parsed_resume["cleaned_text"],
                job_description_input
            )

            # Generate Recommendations
            rec_results = generate_recommendations(
                resume_text=parsed_resume["cleaned_text"],
                job_description=job_description_input,
                matching_skills=match_results["matching_skills"],
                missing_skills=match_results["missing_skills"],
                resume_sections=parsed_resume["sections"],
                resume_metrics=parsed_resume["metrics"],
                api_key=user_api_key,
            )

            # Store in session state for persistence across tab interactions
            st.session_state["analysis_data"] = {
                "parsed_resume": parsed_resume,
                "match_results": match_results,
                "rec_results": rec_results,
                "job_text": job_description_input,
            }

    # Render Results if available
    if "analysis_data" in st.session_state:
        data = st.session_state["analysis_data"]
        parsed_resume = data["parsed_resume"]
        match_results = data["match_results"]
        rec_results = data["rec_results"]

        st.markdown("---")
        st.markdown("## 📊 Analysis & Match Results")

        # Top Metric Cards & Score Indicator
        score = match_results["overall_score"]
        if score >= 75:
            score_color = "#10b981"
            score_status = "Strong Match"
            score_bg = "#ecfdf5"
        elif score >= 50:
            score_color = "#2563eb"
            score_status = "Moderate Match"
            score_bg = "#eff6ff"
        else:
            score_color = "#ef4444"
            score_status = "Needs Improvement"
            score_bg = "#fef2f2"

        score_col, metric_col = st.columns([1, 2], gap="medium")

        with score_col:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-circle" style="background: {score_bg}; border: 4px solid {score_color};">
                    <span class="score-number" style="color: {score_color};">{score}%</span>
                </div>
                <div class="score-label">{score_status}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                    Overall Weighted Fit
                </div>
            </div>
            """, unsafe_allow_html=True)

        with metric_col:
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Skill Match Score",
                    value=f"{match_results['skill_match_score']}%",
                    help="Percentage of job-required skills found in your resume."
                )
                st.metric(
                    label="Matching Skills",
                    value=f"{match_results['total_matched_skills']} skills",
                    help="Total required skills verified in your resume."
                )
            with m2:
                st.metric(
                    label="Semantic Similarity",
                    value=f"{match_results['semantic_score']}%",
                    help="Semantic and contextual alignment measured via Sentence Transformers NLP."
                )
                st.metric(
                    label="Missing Skills",
                    value=f"{match_results['total_missing_skills']} skills",
                    help="Skills required by the job that are not currently in your resume."
                )

            # Informative progress bar
            st.markdown("**Overall Fit Progress**")
            st.progress(score / 100.0)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for organized view
        tab_skills, tab_recs, tab_improvements, tab_details = st.tabs([
            "🎯 Skill Gap Analysis",
            "💡 Learning Recommendations",
            "📝 Resume Improvement Suggestions",
            "🔍 Resume & Job Details",
        ])

        # TAB 1: SKILL GAP ANALYSIS
        with tab_skills:
            st.markdown("### 🔍 Skill Gap Breakdown")

            skill_col1, skill_col2 = st.columns(2, gap="large")

            with skill_col1:
                st.markdown("#### ✅ Matching Skills")
                st.markdown("Skills required by the job that were found in your resume:")
                render_skill_badges(
                    match_results["matching_skills"],
                    "skill-badge-matched",
                    "No exact skill matches detected. Consider tailoring your resume."
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### ❌ Missing Skills")
                st.markdown("Skills requested by the employer that were not detected:")
                render_skill_badges(
                    match_results["missing_skills"],
                    "skill-badge-missing",
                    "None! Your resume covers all required skills detected in this job posting."
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### ➕ Additional Candidate Skills")
                st.markdown("Other skills found in your resume not explicitly listed in the job description:")
                render_skill_badges(
                    match_results["additional_skills"],
                    "skill-badge-additional",
                    "No extra skills detected."
                )

            with skill_col2:
                st.markdown("#### 📊 Visualizations")
                donut = create_match_donut_chart(
                    match_results["total_matched_skills"],
                    match_results["total_missing_skills"]
                )
                if donut:
                    st.altair_chart(donut, use_container_width=True)

                if match_results["category_summary"]:
                    bar_chart = create_category_bar_chart(match_results["category_summary"])
                    if bar_chart:
                        st.altair_chart(bar_chart, use_container_width=True)

        # TAB 2: AI RECOMMENDATIONS
        with tab_recs:
            st.markdown("### 💡 Skill Learning Recommendations")
            
            if rec_results.get("is_ai_generated"):
                st.info(f"✨ Recommendations powered by **{rec_results.get('ai_provider', 'Google Gemini AI')}**")
            else:
                st.caption(f"ℹ️ {rec_results.get('fallback_reason', 'Rule-based recommendations engine.')}")

            missing_skills = match_results["missing_skills"]
            if not missing_skills:
                st.success("🎉 Excellent! You have no missing skills detected for this job. Keep your skills sharp with ongoing projects!")
            else:
                st.markdown("Here is a targeted learning plan for the skills you are currently missing:")
                skill_recs = rec_results.get("skill_recommendations", {})
                for skill in missing_skills:
                    rec_text = skill_recs.get(skill, f"Learn fundamentals of {skill} and build a practical portfolio project.")
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>Missing: {skill}</strong>
                        <p>👉 {rec_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 3: RESUME IMPROVEMENT SUGGESTIONS
        with tab_improvements:
            st.markdown("### 📝 Resume Improvement Suggestions")
            st.markdown("Practical, actionable feedback to optimize your resume for recruiters and ATS filters:")

            # Genuine Strengths Observed
            st.markdown("#### 💪 Resume Strengths Detected")
            strengths = rec_results.get("strengths", [])
            if strengths:
                for s in strengths:
                    st.markdown(f"""
                    <div class="strength-card">
                        <span>✔</span> <span>{s}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No notable strengths highlighted yet. Expand on your project accomplishments.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Suggestions
            st.markdown("#### ⚡ Areas for Improvement")
            suggestions = rec_results.get("suggestions", [])
            if suggestions:
                for item in suggestions:
                    st.markdown(f"""
                    <div class="suggestion-card">
                        <div class="suggestion-header">
                            <span>{item.get('icon', '📌')}</span>&nbsp; {item.get('category', 'Resume Optimization')}
                        </div>
                        <div class="suggestion-issue">
                            <strong>Issue:</strong> {item.get('issue', '')}
                        </div>
                        <div class="suggestion-action">
                            <strong>Action:</strong> {item.get('action', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No critical formatting or content flaws detected. Your resume layout and structure are solid!")

        # TAB 4: DETAILS & STRUCTURE
        with tab_details:
            st.markdown("### 📋 Document Structure & Extracted Data")

            col_doc1, col_doc2 = st.columns(2, gap="large")

            with col_doc1:
                st.markdown("#### 📄 Resume Metadata & Sections")
                st.markdown(f"- **File Name:** `{parsed_resume['file_name']}`")
                st.markdown(f"- **Format:** `{parsed_resume['file_type']}`")
                st.markdown(f"- **Word Count:** `{parsed_resume['word_count']} words`")

                # Contact Info
                contact = parsed_resume.get("contact_info", {})
                st.markdown("##### 👤 Contact Details Detected")
                st.markdown(f"- **Email:** `{contact.get('email') or 'Not detected'}`")
                st.markdown(f"- **Phone:** `{contact.get('phone') or 'Not detected'}`")
                st.markdown(f"- **LinkedIn:** `{contact.get('linkedin') or 'Not detected'}`")
                st.markdown(f"- **GitHub:** `{contact.get('github') or 'Not detected'}`")

                # Section Checklist
                st.markdown("##### 📑 Section Headers Checklist")
                for sec, present in parsed_resume["sections"].items():
                    mark = "✅" if present else "❌"
                    st.markdown(f"{mark} **{sec}**")

            with col_doc2:
                st.markdown("#### 💼 Extracted Job Requirements")
                st.markdown(f"- **Total Technical Skills Identified:** `{match_results['total_required_skills']}`")
                if match_results["job_skills"]:
                    render_skill_badges(match_results["job_skills"], "skill-badge-matched", "None")
                else:
                    st.info("No specific technical skills matched from the predefined database.")

                st.markdown("##### 📈 Quantifiable Metric Samples in Resume")
                sample_metrics = parsed_resume["metrics"].get("sample_lines", [])
                if sample_metrics:
                    for line in sample_metrics[:3]:
                        st.markdown(f"> *\"{line}\"*")
                else:
                    st.caption("No strong metric-containing lines detected.")

            with st.expander("📄 View Cleaned Resume Text"):
                st.text(parsed_resume["cleaned_text"])

            with st.expander("📋 View Target Job Description"):
                st.text(data["job_text"])

    # Footer
    st.markdown("""
    <div class="jobfit-footer">
        <strong>JobFit – AI-Powered Resume & Job Matching System</strong><br>
        Built with Streamlit • Sentence Transformers • Scikit-learn • NLTK • Google Gemini API<br>
        Empowering students and job seekers to land their dream careers.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
