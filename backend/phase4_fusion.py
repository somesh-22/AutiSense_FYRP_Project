# =========================================================
# PHASE 4
# FINAL MULTI-MODAL ASD FUSION
# =========================================================

import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


def compute_final_score(

    phase1,

    eye_score,

    variability,

    neutral
):

    # -------------------------
    # Normalize
    # -------------------------

    p1 = phase1

    p2 = (
        100
        -
        eye_score
    )

    p3 = (
        (100 - variability)
        *
        0.6
        +
        neutral
        *
        0.4
    )

    # -------------------------
    # Weighted Fusion
    # -------------------------

    final_score = (

        p1 * 0.5 +

        p2 * 0.3 +

        p3 * 0.2

    )

    final_score = round(
        final_score,
        2
    )

    # -------------------------
    # Label
    # -------------------------

    if final_score > 70:

        label = (
            "High ASD Risk"
        )

    elif final_score > 40:

        label = (
            "Moderate ASD Risk"
        )

    else:

        label = (
            "Low ASD Risk"
        )

    return (

        final_score,

        label
    )


def generate_phase4_pdf(

    phase1,

    eye,

    variability,

    neutral,

    final_score,

    label
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

    pdf = os.path.join(

        reports,

        "phase4_final_report.pdf"
    )

    doc = SimpleDocTemplate(
        pdf
    )

    styles = (
        getSampleStyleSheet()
    )

    content = []

    content.append(

        Paragraph(

            "FINAL ASD REPORT",

            styles["Title"]

        )

    )

    content.append(
        Spacer(
            1,
            20
        )
    )

    body = f"""
Phase 1 Risk:
{phase1:.2f}%<br/><br/>

Eye Contact:
{eye:.2f}%<br/><br/>

Emotion Variability:
{variability:.2f}%<br/><br/>

Neutral:
{neutral:.2f}%<br/><br/>

FINAL SCORE:
{final_score:.2f}%<br/><br/>

Classification:
{label}
"""

    content.append(

        Paragraph(

            body,

            styles["BodyText"]

        )

    )

    doc.build(
        content
    )

    return pdf