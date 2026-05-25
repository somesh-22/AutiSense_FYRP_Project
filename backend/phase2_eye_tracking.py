# =========================================================
# PHASE 2: EYE TRACKING + GAZE ANALYSIS
# =========================================================

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------
# MEDIAPIPE FACE MESH
# ---------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------
# EYE LANDMARKS
# ---------------------------------------------------------

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# ---------------------------------------------------------
# PROCESS FRAME
# ---------------------------------------------------------

def process_eye_tracking(frame):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(frame_rgb)

    gaze_status = "Face Not Detected"

    eye_contact_score = 0

    if results.multi_face_landmarks:

        gaze_status = "Face Detected"

        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # -------------------------------------------------
        # LEFT EYE
        # -------------------------------------------------

        left_eye_points = []

        for idx in LEFT_EYE:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            left_eye_points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        # -------------------------------------------------
        # RIGHT EYE
        # -------------------------------------------------

        right_eye_points = []

        for idx in RIGHT_EYE:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            right_eye_points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        # -------------------------------------------------
        # DYNAMIC EYE CONTACT SCORING
        # -------------------------------------------------

        # LEFT EYE CENTER
        left_eye_center_x = int(
            (left_eye_points[0][0] + left_eye_points[1][0]) / 2
        )

        # RIGHT EYE CENTER
        right_eye_center_x = int(
            (right_eye_points[0][0] + right_eye_points[1][0]) / 2
        )

        # FACE CENTER
        face_center_x = int(
            (left_eye_center_x + right_eye_center_x) / 2
        )

        # SCREEN CENTER
        screen_center_x = w // 2

        # DISTANCE FROM CENTER
        distance = abs(
            face_center_x - screen_center_x
        )

        # NORMALIZE SCORE
        max_distance = w // 2

        eye_contact_score = max(
            0,
            100 - int(
                (distance / max_distance) * 100
            )
        )

        # -------------------------------------------------
        # STATUS LABEL
        # -------------------------------------------------

        if eye_contact_score > 75:

            gaze_label = "Direct Eye Contact"

        elif eye_contact_score > 45:

            gaze_label = "Partial Attention"

        else:

            gaze_label = "Low Eye Contact"

        # -------------------------------------------------
        # DISPLAY SCORE
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Eye Contact Score: {eye_contact_score}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            gaze_label,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    return frame, gaze_status, eye_contact_score

# =========================================================
# SAVE PHASE 2 REPORT
# =========================================================

import os
from datetime import datetime

def save_phase2_report(avg_score):

    # -----------------------------------------------------
    # CREATE REPORTS DIRECTORY
    # -----------------------------------------------------

    reports_dir = os.path.join(
        os.path.dirname(__file__),
        "reports"
    )

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # ATTENTION LABEL
    # -----------------------------------------------------

    if avg_score > 75:

        attention_label = "Direct Eye Contact"

    elif avg_score > 45:

        attention_label = "Partial Attention"

    else:

        attention_label = "Low Eye Contact"

    # -----------------------------------------------------
    # REPORT FILE
    # -----------------------------------------------------

    report_path = os.path.join(
        reports_dir,
        "phase2_eye_tracking_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "PHASE 2: EYE TRACKING ANALYSIS\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Generated: {datetime.now()}\n\n"
        )

        f.write(
            f"Average Eye Contact Score: {avg_score:.2f}%\n"
        )

        f.write(
            f"Attention Classification: {attention_label}\n"
        )

    return report_path

# =========================================================
# GENERATE PHASE 2 PDF REPORT
# =========================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

def generate_phase2_pdf(avg_score):

    reports_dir = os.path.join(
        os.path.dirname(__file__),
        "reports"
    )

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    pdf_path = os.path.join(
        reports_dir,
        "phase2_eye_tracking_report.pdf"
    )

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    title = Paragraph(
        "PHASE 2: EYE TRACKING ANALYSIS REPORT",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    if avg_score > 75:

        label = "Direct Eye Contact"

    elif avg_score > 45:

        label = "Partial Attention"

    else:

        label = "Low Eye Contact"

    # -----------------------------------------------------
    # CONTENT
    # -----------------------------------------------------

    content = f"""
    Average Eye Contact Score: {avg_score:.2f}%<br/><br/>
    Attention Classification: {label}
    """

    elements.append(
        Paragraph(
            content,
            styles['BodyText']
        )
    )

    doc.build(elements)

    return pdf_path