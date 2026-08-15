import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

# 1. Configuration of the Web Dashboard
st.set_page_config(page_title="ResumeAlchemy", layout="centered", page_icon="📝")
st.title("📝 ResumeAlchemy")
st.subheader("Transform raw resumes into clean, structured data using AI")

# 2. Left Collapsible Sidebar Controls
st.sidebar.header("⚙️ Settings & Authentication")

api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key", 
    type="password",
    help="Get a free key from https://aistudio.google.com/apikey"
)

# Block main screen until key is entered
if not api_key:
    st.info("👈 Please open the sidebar and enter your Google Gemini API key to begin.", icon="🔑")
    st.stop()

# Initialize Gemini client with user key
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.sidebar.error(f"Invalid API Key: {e}")
    st.stop()

# Dynamic Model Selection inside the Sidebar
try:
    raw_models = client.models.list()
    model_options = [
        m.name for m in raw_models 
        if "generateContent" in (getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None) or ["generateContent"])
    ]
    if not model_options:
        model_options = ["models/gemini-2.5-flash", "models/gemini-2.0-flash"]
except Exception:
    model_options = ["models/gemini-2.5-flash", "models/gemini-2.0-flash"]

selected_model = st.sidebar.selectbox("Choose AI Model", model_options)

# 3. Main Screen: Interactive File Upload UI
uploaded_file = st.file_uploader("Upload a candidate's resume (PDF format)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Preparing file binary for Gemini AI Engine..."):
        pdf_bytes = uploaded_file.read()

        resume_part = types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf"
        )
        st.success("File uploaded successfully! Sending to AI Engine...")

    # 4. JSON Output Schema Definition
    class ContactInfo(BaseModel):
        email: str
        phone: str
        location: str

    class Experience(BaseModel):
        job_title: str
        company: str
        duration: str
        responsibilities: list[str]

    class ResumeSchema(BaseModel):
        candidate_name: str
        contact: ContactInfo
        skills: list[str]
        work_experience: list[Experience]
        education: list[str]

    with st.spinner(f"AI is scanning the PDF using {selected_model}..."):
        try:
            # 5. Multimodal Extraction Call
            response = client.models.generate_content(
                model=selected_model,
                contents=[
                    "You are an expert HR data parsing system. Analyze the attached resume PDF document. Extract all available information and organize it perfectly.",
                    resume_part
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeSchema,
                    temperature=0.1
                ),
            )

            # 6. Render Results Dashboard
            st.balloons()
            st.header("⚡ Parsed Candidate Profile")
            st.json(response.text)

        except Exception as e:
            st.error(f"An error occurred during AI processing: {e}")