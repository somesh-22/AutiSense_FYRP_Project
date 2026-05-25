import cv2
from deepface import DeepFace
import numpy as np
import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import (
    getSampleStyleSheet
)

emotion_counts = {}

total_frames = 0


def process_phase3(frame):

    global emotion_counts
    global total_frames

    dominant = "Unknown"

    variability = 0

    neutral = 0

    try:

        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        dominant = result[0][
            "dominant_emotion"
        ]

        emotion_counts[
            dominant
        ] = (
            emotion_counts.get(
                dominant,
                0
            )
            + 1
        )

        total_frames += 1

        variability = (
            len(
                emotion_counts
            ) / 7
        ) * 100

        neutral = (
            emotion_counts.get(
                "neutral",
                0
            )
            /
            total_frames
        ) * 100

    except:
        pass

    cv2.putText(
        frame,
        f"Emotion: {dominant}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    return (
        frame,
        dominant,
        variability,
        neutral
    )


def save_phase3_pdf(
    emotion,
    variability,
    neutral
):

    reports = os.path.join(
        os.path.dirname(
            __file__
        ),
        "reports"
    )

    os.makedirs(
        reports,
        exist_ok=True
    )

    path = os.path.join(
        reports,
        "phase3_report.pdf"
    )

    doc = SimpleDocTemplate(
        path
    )

    styles = (
        getSampleStyleSheet()
    )

    content = []

    content.append(
        Paragraph(
            "PHASE 3 REPORT",
            styles["Title"]
        )
    )

    content.append(
        Spacer(
            1,
            20
        )
    )

    txt = f"""
Emotion: {emotion}<br/><br/>
Emotional Variability:
{variability:.2f}%<br/><br/>
Neutral Dominance:
{neutral:.2f}%
"""

    content.append(
        Paragraph(
            txt,
            styles[
                "BodyText"
            ]
        )
    )

    doc.build(
        content
    )

    return path