"""
MediVision AI - Professional PDF Report Generation
Uses ReportLab to generate publication-grade clinical research reports with side-by-side imagery.
"""
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

from config.settings import settings


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' and confidential headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header
        self.drawString(54, 755, "MediVision AI — Educational Research Report")
        self.drawRightString(558, 755, f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 748, 558, 748)

        # Running Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 34, "EXPERIMENTAL AI PROTOTYPE — NOT FOR MEDICAL DIAGNOSIS")
        self.drawRightString(558, 34, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


class PDFReportGenerator:
    """Generates standardized, downloadable educational research reports."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else settings.REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Register custom typography hierarchy matching UI brand guidelines."""
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(settings.COLOR_PRIMARY),
            spaceAfter=4,
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor(settings.COLOR_TEXT_MUTED),
            spaceAfter=12,
        )
        self.heading2_style = ParagraphStyle(
            "Heading2Custom",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor(settings.COLOR_PRIMARY),
            spaceBefore=10,
            spaceAfter=6,
        )
        self.body_style = ParagraphStyle(
            "BodyCustom",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor(settings.COLOR_TEXT_MAIN),
            spaceAfter=6,
        )
        self.disclaimer_style = ParagraphStyle(
            "DisclaimerCustom",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(settings.COLOR_ALERT),
        )

    def generate_report(
        self,
        analysis_id: str,
        original_image_path: Union[str, Path],
        gradcam_image_path: Optional[Union[str, Path]],
        prediction_data: Dict[str, Any],
        ai_explanation: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Builds and saves the complete PDF report.
        Returns the absolute Path of the generated PDF.
        """
        if not output_filename:
            output_filename = f"medivision_report_{analysis_id}.pdf"

        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("MediVision AI", self.title_style))
        elements.append(Paragraph("Intelligent Medical Image Analysis & Explainable AI Research Report", self.subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(settings.COLOR_SECONDARY), spaceAfter=10))

        # 2. Metadata & Classification Summary Table
        predicted_cat = prediction_data.get("predicted_category", "UNKNOWN")
        confidence_score = float(prediction_data.get("confidence_score", 0.0))
        normal_prob = float(prediction_data.get("class_probabilities", {}).get("NORMAL", 0.0))
        pneumonia_prob = float(prediction_data.get("class_probabilities", {}).get("PNEUMONIA", 0.0))
        latency = float(prediction_data.get("inference_time_ms", 0.0))
        model_name = prediction_data.get("model_name", "DenseNet121")

        cat_color = "#E63946" if predicted_cat == "PNEUMONIA" else "#2A9D8F"

        meta_table_data = [
            [
                Paragraph("<b>Analysis Identifier:</b>", self.body_style),
                Paragraph(f"<code>{analysis_id}</code>", self.body_style),
                Paragraph("<b>Generated Timestamp:</b>", self.body_style),
                Paragraph(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), self.body_style),
            ],
            [
                Paragraph("<b>Model Architecture:</b>", self.body_style),
                Paragraph(f"{model_name} (Transfer Learning)", self.body_style),
                Paragraph("<b>Inference Latency:</b>", self.body_style),
                Paragraph(f"{latency:.1f} ms", self.body_style),
            ],
            [
                Paragraph("<b>Predicted Category:</b>", self.body_style),
                Paragraph(f"<font color='{cat_color}'><b>{predicted_cat}</b></font>", self.body_style),
                Paragraph("<b>Model Confidence:</b>", self.body_style),
                Paragraph(f"<b>{confidence_score * 100:.2f}%</b>", self.body_style),
            ],
            [
                Paragraph("<b>Normal Probability:</b>", self.body_style),
                Paragraph(f"{normal_prob * 100:.2f}%", self.body_style),
                Paragraph("<b>Pneumonia Probability:</b>", self.body_style),
                Paragraph(f"{pneumonia_prob * 100:.2f}%", self.body_style),
            ],
        ]

        meta_table = Table(meta_table_data, colWidths=[115, 137, 115, 137])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # 3. Side-by-Side Visualizations (Original X-Ray & Grad-CAM Heatmap)
        elements.append(Paragraph("Radiographic Imagery & Grad-CAM Spatial Activation", self.heading2_style))

        img_table_data = []
        orig_img_flow = None
        grad_img_flow = None

        if original_image_path and Path(original_image_path).exists():
            orig_img_flow = RLImage(str(original_image_path), width=2.4 * inch, height=2.4 * inch)
        if gradcam_image_path and Path(gradcam_image_path).exists():
            grad_img_flow = RLImage(str(gradcam_image_path), width=2.4 * inch, height=2.4 * inch)

        if orig_img_flow and grad_img_flow:
            img_table_data = [
                [orig_img_flow, grad_img_flow],
                [
                    Paragraph("<para align='center'><b>Input Chest Radiograph</b></para>", self.body_style),
                    Paragraph("<para align='center'><b>Grad-CAM Attention Heatmap</b></para>", self.body_style),
                ]
            ]
            vis_table = Table(img_table_data, colWidths=[252, 252])
            vis_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(KeepTogether(vis_table))
        elif orig_img_flow:
            elements.append(orig_img_flow)

        elements.append(Spacer(1, 10))

        # 4. AI-Generated Educational Findings
        elements.append(Paragraph("AI-Generated Educational Explanation", self.heading2_style))

        exec_summary = ai_explanation.get("executive_summary", "")
        tech_findings = ai_explanation.get("technical_findings", "")
        gradcam_interp = ai_explanation.get("gradcam_interpretation", "")
        model_limits = ai_explanation.get("model_limitations", "")
        dataset_limits = ai_explanation.get("dataset_limitations", "")

        explanation_flow = [
            Paragraph("<b>Executive Summary:</b>", self.body_style),
            Paragraph(exec_summary, self.body_style),
            Spacer(1, 4),
            Paragraph("<b>Technical Analysis:</b>", self.body_style),
            Paragraph(tech_findings, self.body_style),
            Spacer(1, 4),
            Paragraph("<b>Grad-CAM Spatial Interpretation:</b>", self.body_style),
            Paragraph(gradcam_interp, self.body_style),
            Spacer(1, 4),
            Paragraph("<b>Model & Dataset Limitations:</b>", self.body_style),
            Paragraph(f"{model_limits} {dataset_limits}", self.body_style),
        ]
        elements.extend(explanation_flow)
        elements.append(Spacer(1, 10))

        # 5. Prominent Legal / Educational Disclaimer Box
        disclaimer_box = Table(
            [[
                Paragraph(
                    f"<b>IMPORTANT RESEARCH DISCLAIMER:</b> {settings.DISCLAIMER_TEXT}",
                    self.disclaimer_style,
                )
            ]],
            colWidths=[504],
        )
        disclaimer_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#F87171")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        elements.append(KeepTogether(disclaimer_box))

        # Build document with running canvas
        doc.build(elements, canvasmaker=NumberedCanvas)
        return pdf_path


pdf_generator = PDFReportGenerator()
