import streamlit as st
import base64
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import matplotlib.pyplot as plt
import cv2
import av
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)
from sympy import div

from backend.phase1_report import (
    generate_phase1_pdf
)

from backend.phase2_eye_tracking import (
    process_eye_tracking,
    save_phase2_report,
    generate_phase2_pdf
)

from backend.phase3_emotion import (
    process_phase3,
    save_phase3_pdf
)

from backend.phase4_fusion import (

    compute_final_score,

    generate_phase4_pdf
)

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autism Detection | Eye Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
##--BG IMAGE
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(file):
    try:
        bin_str = get_base64(file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(6, 14, 14, 0.94);
            z-index: -1;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        pass

set_background("./assets/eye1.jpg") 

# --- 5. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(40px, 8vw, 70px);
        font-weight: 900;
        line-height: 1.1;
        background: linear-gradient(to right, #ffffff, #4df2c1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-highlight {
        color: #4df2c1;
        text-shadow: 0 0 20px rgba(77, 242, 193, 0.4);
    }

    /* Keep the style definition but we won't use the class in the HTML tags */
    .auth-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        padding: 35px;
        border-radius: 24px;
        border: 1px solid rgba(77, 242, 193, 0.15);
        margin-bottom: 20px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('autisense_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS researchers (
            name TEXT, 
            email TEXT PRIMARY KEY, 
            phone TEXT, 
            password TEXT
        )
    ''')
    conn.commit()
    return conn, c

conn, c = init_db()

# ---------------------------------------------------------
# LOAD PHASE 1 MODEL
# ---------------------------------------------------------
# ---------------------------------------------------------
# LOAD PHASE 1 MODEL
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

phase1_model_path = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "phase1_logistic_model.pkl"
)

phase1_scaler_path = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "phase1_scaler.pkl"
)

phase1_model = joblib.load(phase1_model_path)
phase1_scaler = joblib.load(phase1_scaler_path)
phase1_feature_columns = joblib.load(
    os.path.join(
        BASE_DIR,
        "backend",
        "models",
        "phase1_feature_columns.pkl"
    )
)

# phase1_feature_columns_path = os.path.join(
#     BASE_DIR,
#     "backend",
#     "models",
#     "phase1_feature_columns.pkl"
# )

# phase1_feature_columns = joblib.load(
#     phase1_feature_columns_path
# )

# --- 3. SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_auth' not in st.session_state:
    st.session_state.show_auth = False
if 'show_admin' not in st.session_state:
    st.session_state.show_admin = False
if 'admin_verified' not in st.session_state:
    st.session_state.admin_verified = False
if 'phase1_completed' not in st.session_state:
        st.session_state.phase1_completed = False

if 'phase1_probability' not in st.session_state:
        st.session_state.phase1_probability = 0

if 'phase1_prediction' not in st.session_state:
        st.session_state.phase1_prediction = 0

# --- 4. BACKGROUND HANDLER ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(file):
    try:
        bin_str = get_base64(file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(6, 14, 14, 0.94);
            z-index: -1;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        pass

set_background("./assets/eye1.jpg") 

# --- 5. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(40px, 8vw, 70px);
        font-weight: 900;
        line-height: 1.1;
        background: linear-gradient(to right, #ffffff, #4df2c1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-highlight {
        color: #4df2c1;
        text-shadow: 0 0 20px rgba(77, 242, 193, 0.4);
    }

    /* Keep the style definition but we won't use the class in the HTML tags */
    .auth-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        padding: 35px;
        border-radius: 24px;
        border: 1px solid rgba(77, 242, 193, 0.15);
        margin-bottom: 20px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 6. VISUALIZATION ---
def draw_live_graph():
    t = np.linspace(0, 10, 100)
    y = np.sin(t) * np.exp(-0.1 * t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', line=dict(color='#4df2c1', width=3)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=0, b=0), height=250, showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig

# ---------------------------------------------------------
# PHASE 2 VIDEO TRANSFORMER
# ---------------------------------------------------------

class EyeTrackingTransformer(VideoProcessorBase):

    def __init__(self):

        self.latest_score = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        processed_frame, gaze_status, eye_score = process_eye_tracking(img)

        # Store internally (NOT session state)
        self.latest_score = eye_score

        return av.VideoFrame.from_ndarray(
            processed_frame,
            format="bgr24"
        )
# --- 7. AUTHENTICATION VIEW ---
def auth_view():
    # Box removed from here
    mode = st.selectbox("Select Action", ["Login", "Sign Up"], label_visibility="collapsed")
    
    if mode == "Login":
        st.markdown('<h2 style="font-family:Orbitron; color:#4df2c1;">LOGIN</h2>', unsafe_allow_html=True)
        login_email = st.text_input("📧 Email", placeholder="researcher@institution.edu")
        login_pw = st.text_input("🔑 Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("INITIALIZE"):
                c.execute('SELECT * FROM researchers WHERE email=? AND password=?', (login_email, login_pw))
                if c.fetchone():
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        with col2:
            if st.button("BACK"):
                st.session_state.show_auth = False
                st.rerun()

    else:
        st.markdown('<h2 style="font-family:Orbitron; color:#4df2c1;">NEW REGISTRY</h2>', unsafe_allow_html=True)
        new_name = st.text_input("👤 Full Name")
        new_email = st.text_input("✉️ Institutional Email")
        new_phone = st.text_input("📞 Phone Number")
        new_pw = st.text_input("🛡️ Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("REGISTER"):
                if not all([new_name, new_email, new_phone, new_pw]):
                    st.warning("All fields are required.")
                else:
                    try:
                        c.execute('INSERT INTO researchers VALUES (?,?,?,?)', (new_name, new_email, new_phone, new_pw))
                        conn.commit()
                        st.success("Registration Successful!")
                    except sqlite3.IntegrityError:
                        st.error("Email already registered.")
        with col2:
            if st.button("CANCEL"):
                st.session_state.show_auth = False
                st.rerun()

# --- 8. PAGE ROUTING ---

# ADMIN PANEL
if st.session_state.show_admin:
    st.markdown('<h1 class="hero-title">ADMIN <span class="hero-highlight">PORTAL</span></h1>', unsafe_allow_html=True)
    
    with st.container():
        # Box removed from here
        if not st.session_state.admin_verified:
            st.subheader("Restricted Access")
            adm_user = st.text_input("Admin ID")
            adm_pw = st.text_input("Admin Password", type="password")
            
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                if st.button("VERIFY IDENTITY"):
                    if adm_user == "admin" and adm_pw == "password123":
                        st.session_state.admin_verified = True
                        st.rerun()
                    else:
                        st.error("Invalid Admin Credentials")
            with c_a2:
                if st.button("EXIT"):
                    st.session_state.show_admin = False
                    st.rerun()
        else:
            st.success("Welcome, Administrator.")
            df = pd.read_sql("SELECT name, email, phone FROM researchers", conn)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv, file_name="registrations.csv")
            if st.button("LOGOUT ADMIN"):
                st.session_state.admin_verified = False
                st.session_state.show_admin = False
                st.rerun()

# LOGGED IN VIEW
# elif st.session_state.logged_in:
#     st.markdown('<h1 class="hero-title">RESEARCH <span class="hero-highlight">TERMINAL</span></h1>', unsafe_allow_html=True)
#     st.info("You are currently viewing the secure ASD dataset.")
#     if st.button("LOGOUT"):
#         st.session_state.logged_in = False
#         st.rerun()

elif st.session_state.logged_in:

    st.markdown(
        '<h1 class="hero-title">RESEARCH <span class="hero-highlight">TERMINAL</span></h1>',
        unsafe_allow_html=True
    )

    st.info("Phase 1: Toddler Autism Screening")

    st.markdown("## Screening Questionnaire")

    col1, col2 = st.columns(2)

    with col1:

        A1 = st.selectbox(
            "Q1. Does your child look at you when you call his/her name?",
            ["Yes", "No"]
        )

        A2 = st.selectbox(
            "Q2. Is it easy for you to get eye contact with your child?",
            ["Yes", "No"]
        )

        A3 = st.selectbox(
            "Q3. Does your child point to indicate that she/he wants something?",
            ["Yes", "No"]
        )

        A4 = st.selectbox(
            "Q4. Does your child point to share interest with you?",
            ["Yes", "No"]
        )

        A5 = st.selectbox(
            "Q5. Does your child pretend?",
            ["Yes", "No"]
        )

    with col2:

        A6 = st.selectbox(
            "Q6. Does your child follow where you’re looking?",
            ["Yes", "No"]
        )

        A7 = st.selectbox(
            "Q7. If someone is upset, does your child try to comfort them?",
            ["Yes", "No"]
        )

        A8 = st.selectbox(
            "Q8. Would you describe your child’s first words as normal?",
            ["Yes", "No"]
        )

        A9 = st.selectbox(
            "Q9. Does your child use simple gestures?",
            ["Yes", "No"]
        )

        A10 = st.selectbox(
            "Q10. Does your child stare at nothing with no apparent purpose?",
            ["Yes", "No"]
        )

    age_months = st.number_input(
        "Age (Months)",
        min_value=12,
        max_value=72,
        value=24
    )

    if st.button(
        "RUN PHASE 1 SCREENING",
        key="phase1_screen"
    ):

        try:

            # -------------------------
            # BUILD INPUT
            # -------------------------

            input_dict = {}

            for col in phase1_feature_columns:
                input_dict[col] = 0

            answers = [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]

            for i in range(10):
                feature = f"A{i+1}"

                if feature in input_dict:

                    input_dict[feature] = (
                    0
                    if answers[i] == "Yes"
                    else 1
                )


            if "Age_Mons" in input_dict:

                input_dict["Age_Mons"] = age_months

            # -------------------------
            # DATAFRAME
            # -------------------------

            input_df = pd.DataFrame(
                [input_dict]
            )

            input_df = input_df[
                phase1_feature_columns
            ]

            # st.write(
            #     "Input Sent To Model"
            # )

            # st.dataframe(
            #     input_df
            # )

            # st.write("Input")
            # st.dataframe(input_df)

            # -------------------------
            # PREDICT
            # -------------------------

            # -------------------------
# PREDICT
# -------------------------

            scaled_input = (phase1_scaler.transform(input_df))

# REAL MODEL OUTPUT
            probs = phase1_model.predict_proba(
    scaled_input
)[0]

            prediction = (
    phase1_model
    .predict(
        scaled_input
    )[0]
)

            probability = round(probs[1] * 100,2)

            st.write(
    "Model Output:",
    probs
)

            # -------------------------
            # SAVE
            # -------------------------

            st.session_state.phase1_completed = True

            st.session_state.phase1_prediction = prediction

            st.session_state.phase1_probability = probability

            st.session_state.phase1_questions = [
                "Does your child look at you when you call his/her name?",
                "How easy is it for you to get eye contact?",
                "Does your child point to indicate needs?",
                "Does your child point to share interest?",
                "Does your child pretend?",
                "Does your child follow your gaze?",
                "Does your child comfort others?",
                "Would you describe first words as normal?",
                "Does your child use gestures?",
                "Does your child stare at nothing?"
            ]

            st.session_state.phase1_answers = [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]

            st.session_state.phase1_age = (age_months)

            st.success(
                "Screening Completed Successfully"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

    

    # =========================================================
    # PHASE 1 PERSISTENT RESULTS
    # =========================================================

    if st.session_state.phase1_completed:

        st.metric(
            "Autism Risk Probability",
            f"{st.session_state.phase1_probability:.2f}%"
        )

        if st.session_state.phase1_prediction == 1:

            st.error(
                "High ASD Risk Detected"
            )

        else:

            st.success(
                "Low ASD Risk Detected"
            )

        # -------------------------------------------------
        # SHOW SAVED REPORT
        # -------------------------------------------------

        # -------------------------------------------------
# DYNAMIC USER REPORT
# -------------------------------------------------

        report_text = f"""
        PHASE 1 SCREENING REPORT
        ========================

        Current Prediction:
        {"High ASD Risk"
        if st.session_state.phase1_prediction==1
        else
        "Low ASD Risk"}

        Current Autism Risk:
        {st.session_state.phase1_probability:.2f}%

        Questionnaire Features:
        A1–A10 + Age

        Model:
        Logistic Regression

        Age:
        {st.session_state.phase1_age} Months

        Status:
        Completed
        """

        st.text_area(
            "Phase 1 Detailed Report",
            report_text,
            height=300
        )

        if st.button("GENERATE PHASE 1 PDF",key="phase1_pdf"):

            pdf_path = generate_phase1_pdf(

            st.session_state.phase1_questions,

            st.session_state.phase1_answers,

            st.session_state.phase1_age,

            st.session_state.phase1_prediction,

            st.session_state.phase1_probability
            )

            with open(
                pdf_path,
                "rb"
            ) as f:

                st.download_button(

                    "DOWNLOAD PHASE 1 PDF",

                    f,

                    file_name="phase1_report.pdf",

                    mime="application/pdf",
                    key="phase1_pdf_download"
                )

        # -------------------------------------------------
        # SHOW FEATURE IMPORTANCE GRAPH
        # -------------------------------------------------

        graph_path = os.path.join(
            BASE_DIR,
            "backend",
            "reports",
            "phase1_feature_importance.png"
        )

        if os.path.exists(graph_path):

            st.image(
                graph_path,
                caption="Feature Importance Analysis",
                use_container_width=True
            )

        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        # if os.path.exists(report_path):

        #     with open(report_path, "rb") as file:

        #         st.download_button(
        #             label="DOWNLOAD PHASE 1 REPORT",
        #             data=file,
        #             file_name="phase1_results.txt",
        #             mime="text/plain"
        #         )

       # =========================================================
    # PHASE 2 UI
    # =========================================================

    # =========================================================
    # PHASE 2 UI
    # =========================================================

    st.markdown("---")

    st.markdown("## 👁️ Phase 2: Eye Tracking Analysis")

    st.info(
        "Real-time gaze and eye-contact behavioral analysis"
    )

    # ---------------------------------------------------------
    # WEBCAM STREAM
    # ---------------------------------------------------------

    ctx = webrtc_streamer(
        key="eye-tracking",
        video_processor_factory=EyeTrackingTransformer,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    # ---------------------------------------------------------
    # LIVE SCORE DISPLAY
    # ---------------------------------------------------------

    if ctx.video_processor:

        live_score = ctx.video_processor.latest_score
        st.session_state.latest_score = (live_score)

        st.metric(
            "Live Eye Contact Score",
            f"{live_score}%"
        )

    # ---------------------------------------------------------
    # SAVE PHASE 2 RESULTS
    # ---------------------------------------------------------

        # ---------------------------------------------------------
    # SAVE PHASE 2 RESULTS
    # ---------------------------------------------------------

    if st.button("SAVE PHASE 2 RESULTS"):

        if ctx.video_processor:

            avg_score = ctx.video_processor.latest_score

            # TXT REPORT
            report_path = save_phase2_report(
                avg_score
            )

            # PDF REPORT
            pdf_path = generate_phase2_pdf(
                avg_score
            )

            st.success(
                "Phase 2 Report Saved Successfully"
            )

            st.metric(
                "Average Eye Contact Score",
                f"{avg_score:.2f}%"
            )

            # SHOW TXT REPORT

            with open(report_path, "r") as f:

                report_text = f.read()

            st.text_area(
                "Phase 2 Report",
                report_text,
                height=250
            )

            # DOWNLOAD PDF

            with open(pdf_path, "rb") as pdf_file:

                st.download_button(
                    label="DOWNLOAD PHASE 2 PDF REPORT",
                    data=pdf_file,
                    file_name="phase2_eye_tracking_report.pdf",
                    mime="application/pdf"
                )

        else:

            st.warning(
                "Webcam stream not active."
            )


    # if st.button("LOGOUT"):
    #     st.session_state.logged_in = False
    #     st.rerun()

        # =========================================================
    # PHASE 3 UI
    # =========================================================

    st.markdown("---")

    st.markdown(
        "## 😊 Phase 3: Emotion & Social Responsiveness"
    )

    st.info(
        "Real-time emotional response analysis"
    )

    # ---------------------------------------------------------
    # SESSION VARIABLES
    # ---------------------------------------------------------

    if "phase3_emotion" not in st.session_state:
        st.session_state.phase3_emotion = "Unknown"

    if "phase3_variability" not in st.session_state:
        st.session_state.phase3_variability = 0

    if "phase3_neutral" not in st.session_state:
        st.session_state.phase3_neutral = 0

    # ---------------------------------------------------------
    # CAMERA CONTROL
    # ---------------------------------------------------------

    if "phase3_active" not in st.session_state:

        st.session_state.phase3_active = False

    if not st.session_state.phase3_active:

        if st.button(
            "START PHASE 3 ANALYSIS",
            key="phase3_start"
        ):

            st.session_state.phase3_active = True

            st.rerun()

    # ---------------------------------------------------------
    # CAMERA OPEN
    # ---------------------------------------------------------

    if st.session_state.phase3_active:

        phase3_img = st.camera_input(
            "Capture Emotion Frame"
        )

        if phase3_img:

            file_bytes = np.asarray(
                bytearray(
                    phase3_img.read()
                ),
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                file_bytes,
                1
            )

            processed, emotion, variability, neutral = (
                process_phase3(
                    frame
                )
            )

            st.image(
                processed,
                channels="BGR",
                use_container_width=True
            )

            st.session_state.phase3_emotion = (
                emotion
            )

            st.session_state.phase3_variability = (
                variability
            )

            st.session_state.phase3_neutral = (
                neutral
            )

        # ---------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Emotion",
                st.session_state.phase3_emotion
            )

        with c2:

            st.metric(
                "Emotional Variability",
                f"{st.session_state.phase3_variability:.2f}%"
            )

        with c3:

            st.metric(
                "Neutral Dominance",
                f"{st.session_state.phase3_neutral:.2f}%"
            )

        # ---------------------------------------------------------
        # SAVE REPORT
        # ---------------------------------------------------------

        if st.button(
            "SAVE PHASE 3 REPORT",
            key="phase3_save"
        ):

            pdf = save_phase3_pdf(

                st.session_state.phase3_emotion,

                st.session_state.phase3_variability,

                st.session_state.phase3_neutral
            )

            st.success(
                "Phase 3 PDF Generated"
            )

            with open(
                pdf,
                "rb"
            ) as f:

                st.download_button(

                    "DOWNLOAD PHASE 3 PDF",

                    f,

                    file_name="phase3_report.pdf",

                    mime="application/pdf"
                )

        # ---------------------------------------------------------
        # CLOSE CAMERA
        # ---------------------------------------------------------

        if st.button(
            "CLOSE PHASE 3",
            key="phase3_close"
        ):

            st.session_state.phase3_active = False

            st.rerun()

    # ---------------------------------------------------------
    # LOGOUT
    # ---------------------------------------------------------

    # if st.button(
    #     "LOGOUT",
    #     key="main_logout_button"
    # ):

    #     st.session_state.logged_in = False

    #     st.rerun()

        # =========================================================
    # PHASE 4
    # FINAL MULTI-MODAL FUSION
    # =========================================================

    st.markdown("---")

    st.markdown(
        "## 🧠 Phase 4: Final ASD Decision"
    )

    st.info(
        "Combining screening + eye tracking + emotional responsiveness"
    )

    # ---------------------------------------------------------
    # DEFAULTS
    # ---------------------------------------------------------

    phase1_score = (
        st.session_state.phase1_probability
        if "phase1_probability"
        in st.session_state
        else 0
    )

    eye_score = 0

    if (
        "latest_score"
        in st.session_state
    ):

        eye_score = (
            st.session_state.latest_score
        )

    emotion_var = (
        st.session_state.phase3_variability
        if "phase3_variability"
        in st.session_state
        else 0
    )

    neutral = (
        st.session_state.phase3_neutral
        if "phase3_neutral"
        in st.session_state
        else 0
    )

    # ---------------------------------------------------------
    # RUN FUSION
    # ---------------------------------------------------------

    if st.button(
        "RUN FINAL ASD ANALYSIS",
        key="phase4_run"
    ):

        score, label = (

            compute_final_score(

                phase1_score,

                eye_score,

                emotion_var,

                neutral
            )
        )

        st.session_state.phase4_score = (
            score
        )

        st.session_state.phase4_label = (
            label
        )

    # ---------------------------------------------------------
    # SHOW RESULTS
    # ---------------------------------------------------------

    if (
        "phase4_score"
        in st.session_state
    ):

        st.metric(
            "Final ASD Risk",
            f"{st.session_state.phase4_score:.2f}%"
        )

        if (
            st.session_state.phase4_label
            ==
            "High ASD Risk"
        ):

            st.error(
                st.session_state.phase4_label
            )

        elif (
            st.session_state.phase4_label
            ==
            "Moderate ASD Risk"
        ):

            st.warning(
                st.session_state.phase4_label
            )

        else:

            st.success(
                st.session_state.phase4_label
            )

        # -----------------------------------------------------
        # SAVE PDF
        # -----------------------------------------------------

        if st.button(
            "SAVE FINAL REPORT",
            key="phase4_pdf"
        ):

            pdf = (
                generate_phase4_pdf(

                    phase1_score,

                    eye_score,

                    emotion_var,

                    neutral,

                    st.session_state.phase4_score,

                    st.session_state.phase4_label
                )
            )

            st.success(
                "Final Report Generated"
            )

            with open(
                pdf,
                "rb"
            ) as f:

                st.download_button(

                    "DOWNLOAD FINAL PDF",

                    f,

                    file_name="final_asd_report.pdf",

                    mime="application/pdf"
                )
## ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------

    if st.button("LOGOUT", key="main_logout_button"):

        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# AUTH SCREEN
# =========================================================

elif st.session_state.show_auth:

    _, c_center, _ = st.columns([1, 1.5, 1])

    with c_center:

        auth_view()

# =========================================================
# PUBLIC HOME
# =========================================================

else:

    nav_l, nav_r1, nav_r2 = st.columns([4, 1, 1])

    with nav_l:

        st.markdown(
            '<p style="font-family:Orbitron; color:#4df2c1; font-weight:900; font-size: 50px; margin:0;">AutiSense</p>',
            unsafe_allow_html=True
        )

    with nav_r1:

        if st.button("Login / Sign Up"):

            st.session_state.show_auth = True
            st.rerun()

    # with nav_r2:

    #     if st.button("🛠️ Admin"):

    #         st.session_state.show_admin = True
    #         st.rerun()

    st.markdown(
        '<div style="margin-top: 80px;"></div>',
        unsafe_allow_html=True
    )

    col_main, col_side = st.columns([1.5, 1])

    with col_main:

        st.markdown(
        '''
        <h1 class="hero-title">
        Autism Prediction<br>
        <span class="hero-highlight">
        using 
        </span>
        via<br>
        Eye Movement Analysis
        </h1>
        ''',
        unsafe_allow_html=True
    )
        

    with col_side:

        st.markdown(
            '<div style="margin-top: 100px;"></div>',
            unsafe_allow_html=True
        )

    
        st.plotly_chart(
            draw_live_graph(),
            use_container_width=True,
            config={'displayModeBar': False}
        )

      