import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


def generate_phase1_pdf(

    questions,

    answers,

    age,

    prediction,

    probability
):

    reports = os.path.join(
        os.path.dirname(__file__),
        "reports"
    )

    os.makedirs(
        reports,
        exist_ok=True
    )

    pdf = os.path.join(
        reports,
        "phase1_report.pdf"
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
            "PHASE 1 AUTISM SCREENING REPORT",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    result = (
        "High ASD Risk"
        if prediction == 1
        else
        "Low ASD Risk"
    )

    content.append(
        Paragraph(
            f"""
Prediction:
{result}

<br/><br/>

Autism Risk:
{probability:.2f}%

<br/><br/>

Age:
{age} months
""",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(
            1,
            20
        )
    )

    content.append(
        Paragraph(
            "QUESTIONNAIRE",
            styles["Heading1"]
        )
    )

    for i in range(10):

        content.append(

            Paragraph(

                f"""
Q{i+1}:
{questions[i]}

<br/>

Answer:
{answers[i]}
""",

                styles[
                    "BodyText"
                ]
            )
        )

        content.append(
            Spacer(
                1,
                10
            )
        )

    doc.build(
        content
    )

    return pdf