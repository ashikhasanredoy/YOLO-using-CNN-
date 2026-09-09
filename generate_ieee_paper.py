#!/usr/bin/env python3
"""
IEEE Research Paper PDF Generator
Title: Adaptive Image Enhancement for Robust Object Detection Under Adverse Weather Conditions
Author: Ashik Hasan Redoy
Compiles a publication-grade, multi-page IEEE conference/transactions style PDF paper.
"""

import os
import sys
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, 
    Table, TableStyle, Image, KeepTogether, PageBreak, FrameBreak, NextPageTemplate, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = PROJECT_ROOT / "Adaptive_Image_Enhancement_YOLO_IEEE_Paper.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total pages and draw IEEE running headers/footers."""
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#374151"))
        
        # Running header on page 2 onwards
        if self._pageNumber > 1:
            header_text = "IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS / RESEARCH MANUSCRIPT"
            self.drawString(36, 758, header_text)
            self.setStrokeColor(colors.HexColor("#9ca3af"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)
        
        # Running footer on all pages
        footer_left = "ADAPTIVE IMAGE ENHANCEMENT FOR OBJECT DETECTION UNDER ADVERSE CONDITIONS"
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(36, 24, footer_left)
        self.drawRightString(576, 24, page_str)
        self.setStrokeColor(colors.HexColor("#9ca3af"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        
        self.restoreState()


def get_scaled_image(img_path: Path, max_w: float = 255):
    """Returns scaled ReportLab Image flowable matching column width."""
    if not img_path.exists():
        return None
    with PILImage.open(img_path) as im:
        orig_w, orig_h = im.size
    ratio = max_w / float(orig_w)
    calc_h = orig_h * ratio
    return Image(str(img_path), width=max_w, height=calc_h)


def build_paper():
    # Page layout specifications (Letter: 612 x 792 pt, 0.5 in margin = 36 pt)
    page_w, page_h = letter
    margin = 36.0
    col_gap = 14.0
    col_w = (page_w - (2 * margin) - col_gap) / 2.0  # 263.0 pt
    content_h = page_h - (2 * margin)                # 720.0 pt
    header_h = 135.0                                 # Title + Authors block height on Page 1

    # First page frames
    frame_head = Frame(margin, page_h - margin - header_h, page_w - (2 * margin), header_h, id='head',
                       topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    frame_col1 = Frame(margin, margin, col_w, content_h - header_h, id='col1',
                       topPadding=4, bottomPadding=4, leftPadding=0, rightPadding=0)
    frame_col2 = Frame(margin + col_w + col_gap, margin, col_w, content_h - header_h, id='col2',
                       topPadding=4, bottomPadding=4, leftPadding=0, rightPadding=0)

    # Later pages frames (two full columns)
    frame_p2_col1 = Frame(margin, margin, col_w, content_h - 18, id='p2_col1',
                          topPadding=4, bottomPadding=4, leftPadding=0, rightPadding=0)
    frame_p2_col2 = Frame(margin + col_w + col_gap, margin, col_w, content_h - 18, id='p2_col2',
                          topPadding=4, bottomPadding=4, leftPadding=0, rightPadding=0)

    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )

    first_page_tmpl = PageTemplate(id='FirstPage', frames=[frame_head, frame_col1, frame_col2])
    two_col_tmpl = PageTemplate(id='TwoColPage', frames=[frame_p2_col1, frame_p2_col2])
    doc.addPageTemplates([first_page_tmpl, two_col_tmpl])

    # IEEE Typography Styles
    base_styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        'IEEETitle',
        parent=base_styles['Title'],
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=6,
        textColor=colors.HexColor('#111827')
    )

    style_author = ParagraphStyle(
        'IEEEAuthor',
        parent=base_styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=13,
        alignment=1, # Center
        spaceAfter=2,
        textColor=colors.HexColor('#1f2937')
    )

    style_affil = ParagraphStyle(
        'IEEEAffil',
        parent=base_styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=1, # Center
        spaceAfter=10,
        textColor=colors.HexColor('#4b5563')
    )

    style_abstract_heading = ParagraphStyle(
        'IEEEAbstractHead',
        parent=base_styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=9,
        leading=11,
        spaceAfter=2,
        textColor=colors.HexColor('#111827')
    )

    style_abstract_body = ParagraphStyle(
        'IEEEAbstractBody',
        parent=base_styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=4, # Justified
        spaceAfter=6,
        textColor=colors.HexColor('#111827')
    )

    style_keywords = ParagraphStyle(
        'IEEEKeywords',
        parent=base_styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=4,
        spaceAfter=8,
        textColor=colors.HexColor('#111827')
    )

    style_sec_heading = ParagraphStyle(
        'IEEESecHead',
        parent=base_styles['Heading1'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12,
        alignment=1, # Centered Small Caps style
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor('#111827')
    )

    style_subsec_heading = ParagraphStyle(
        'IEEESubSecHead',
        parent=base_styles['Heading2'],
        fontName='Times-BoldItalic',
        fontSize=9,
        leading=11.5,
        spaceBefore=5,
        spaceAfter=3,
        textColor=colors.HexColor('#1f2937')
    )

    style_body = ParagraphStyle(
        'IEEEBody',
        parent=base_styles['BodyText'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=4, # Justified
        firstLineIndent=10,
        spaceAfter=3,
        textColor=colors.HexColor('#111827')
    )

    style_body_no_indent = ParagraphStyle(
        'IEEEBodyNoIndent',
        parent=style_body,
        firstLineIndent=0,
        spaceAfter=4
    )

    style_equation = ParagraphStyle(
        'IEEEEquation',
        parent=base_styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=1, # Centered
        spaceBefore=3,
        spaceAfter=4,
        textColor=colors.HexColor('#111827')
    )

    style_table_title = ParagraphStyle(
        'IEEETableTitle',
        parent=base_styles['Normal'],
        fontName='Times-Bold',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.HexColor('#111827')
    )

    style_table_cell = ParagraphStyle(
        'IEEETableCell',
        parent=base_styles['Normal'],
        fontName='Times-Roman',
        fontSize=7,
        leading=8.5,
        alignment=1, # Center
        textColor=colors.HexColor('#111827')
    )

    style_table_cell_bold = ParagraphStyle(
        'IEEETableCellBold',
        parent=style_table_cell,
        fontName='Times-Bold'
    )

    style_fig_caption = ParagraphStyle(
        'IEEEFigCaption',
        parent=base_styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        spaceBefore=2,
        spaceAfter=6,
        textColor=colors.HexColor('#374151')
    )

    style_reference = ParagraphStyle(
        'IEEEReference',
        parent=base_styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.5,
        alignment=4,
        firstLineIndent=-12,
        leftIndent=12,
        spaceAfter=2,
        textColor=colors.HexColor('#1f2937')
    )

    story = []

    # ==========================================
    # 1. TITLE & AUTHOR BLOCK (Spans full page width)
    # ==========================================
    story.append(Paragraph("Adaptive Image Enhancement for Robust Object Detection Under Adverse Weather Conditions", style_title))
    story.append(Paragraph("<b>Ashik Hasan Redoy</b>", style_author))
    story.append(Paragraph("Department of Computer Science & Engineering<br/>Email: ashikhasanhredoy@gmail.com", style_affil))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9ca3af"), spaceBefore=2, spaceAfter=4))
    
    # Transition to two-column flow
    story.append(NextPageTemplate('TwoColPage'))
    story.append(FrameBreak())

    # ==========================================
    # 2. ABSTRACT & INDEX TERMS
    # ==========================================
    abstract_text = (
        "<b><i>Abstract</i>—Autonomous vehicle perception relies predominantly on deep convolutional "
        "object detectors such as YOLO. However, adverse environmental conditions—including rain, fog, "
        "nighttime low illumination, and transition glare at dawn and dusk—severely degrade sensor imagery "
        "via particulate light scattering, reduced contrast, specular reflections, and sensor noise amplification, "
        "causing substantial drops in detection accuracy. This paper investigates whether adaptive, condition-specific "
        "image enhancement prior to deep object detection can improve detection performance under adverse conditions. "
        "We propose a hybrid framework comprising: (i) a PyTorch ResNet-18 convolutional neural network that classifies "
        "environmental driving conditions into five distinct domains with 71.67% test accuracy; (ii) targeted "
        "classical computer vision enhancement filters including Dark Channel Prior (DCP) dehazing, bilateral "
        "edge-preserving filtering with high-boost unsharp masking for rain, nonlinear power-law gamma expansion with "
        "luminance CLAHE for night scenes, and dynamic range balancing for dawn/dusk; and (iii) an off-the-shelf "
        "Ultralytics YOLO11n object detector with frozen weights. Evaluating four experimental pipelines on 1,197 real-world "
        "driving images from the Berkeley DeepDrive (BDD100K) benchmark across a blind test set (N=180), our results "
        "reveal that while adaptive enhancement yields notable gains in dawn/dusk transition light (+4.67% mAP@50), "
        "spatial-domain enhancement does not provide universal improvement (Baseline mAP@50 = 0.2029 vs. Adaptive mAP@50 = "
        "0.2025 vs. Fixed CLAHE = 0.2043), with nighttime detection degrading by -5.76% due to sensor noise amplification. "
        "These findings demonstrate that visual enhancement tuned for human perception does not equate to improved deep feature "
        "representation, offering critical design guidelines for robust vision in autonomous systems.</b>"
    )
    story.append(Paragraph(abstract_text, style_abstract_body))
    
    keywords_text = (
        "<b><i>Index Terms</i>—Object Detection, YOLO, Convolutional Neural Networks, Adaptive Image Enhancement, "
        "Adverse Weather Conditions, Autonomous Driving, BDD100K, ResNet-18.</b>"
    )
    story.append(Paragraph(keywords_text, style_keywords))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d1d5db"), spaceBefore=2, spaceAfter=6))

    # ==========================================
    # SECTION I: INTRODUCTION
    # ==========================================
    story.append(Paragraph("I. INTRODUCTION", style_sec_heading))
    story.append(Paragraph(
        "Autonomous driving systems and Advanced Driver Assistance Systems (ADAS) depend critically on reliable real-time "
        "object detection to identify vehicles, pedestrians, cyclists, and traffic control devices. State-of-the-art "
        "detectors such as the You Only Look Once (YOLO) architecture achieve exceptional accuracy and latency on standard "
        "daylight benchmarks. However, deployed autonomous vehicles operate in unconstrained environments subject to harsh "
        "weather and illumination shifts. Rain streaks blur object contours and generate specular road reflections; atmospheric "
        "fog scatters light, severely attenuating scene contrast; nighttime conditions induce underexposure and amplify sensor "
        "shot noise; and low-angle sun during dawn and dusk causes severe dynamic range imbalance.",
        style_body
    ))
    story.append(Paragraph(
        "A common intuition in computer vision is to insert an image enhancement pre-processing stage to 'clean' degraded sensory "
        "inputs before feeding them to deep networks. Nevertheless, whether spatial-domain enhancement genuinely benefits deep neural "
        "detectors remains contentious. While human observers prefer enhanced contrast and brighter imagery, convolutional neural "
        "networks operate on latent gradient activations that may be disrupted by non-linear spatial transformations.",
        style_body
    ))
    story.append(Paragraph(
        "To rigorously investigate this dynamic, we formulate the central research question: <i>'Can adaptive image enhancement "
        "improve YOLO-based object detection performance under adverse environmental conditions?'</i> We construct a complete, "
        "reproducible experimental framework comparing four pipelines on the real-world Berkeley DeepDrive (BDD100K) dataset:",
        style_body
    ))
    story.append(Paragraph("• <b>Pipeline A (Baseline)</b>: Original images passed directly to YOLO without modification (control group).", style_body))
    story.append(Paragraph("• <b>Pipeline B (Fixed CLAHE)</b>: Condition-agnostic uniform CLAHE applied to all images regardless of weather.", style_body))
    story.append(Paragraph("• <b>Pipeline C (Adaptive Enhancement)</b>: Deep ResNet-18 condition classifier dynamically routing images to condition-matched enhancement algorithms.", style_body))
    story.append(Paragraph("• <b>Pipeline D (Oracle Routing)</b>: Enhancement routed using ground-truth condition labels, establishing the theoretical ceiling.", style_body))

    # Formal Hypotheses
    story.append(Paragraph(
        "Specifically, we investigate four formal research hypotheses: "
        "<b>H1</b>: Adaptive enhancement improves overall YOLO detection under adverse conditions. "
        "<b>H2</b>: Condition-specific enhancement outperforms fixed one-size-fits-all CLAHE. "
        "<b>H3</b>: Image enhancement yields greater quantitative gain under degraded conditions than clear daylight. "
        "<b>H4</b>: Task-oriented detection filtering diverges fundamentally from human subjective visual quality.",
        style_body
    ))

    # ==========================================
    # SECTION II: RELATED WORK
    # ==========================================
    story.append(Paragraph("II. RELATED WORK", style_sec_heading))
    story.append(Paragraph(
        "<i>A. Deep Object Detection in Driving Scenes:</i> Deep learning detectors are broadly categorized into two-stage "
        "(Faster R-CNN) and single-stage architectures (SSD, YOLO). The YOLO family has achieved dominance in edge robotics due "
        "to its single-stage regression formulation. The latest Ultralytics YOLO11 architecture introduces C3k2 cross-stage partial "
        "bottlenecks and Spatial Pyramid Pooling - Fast (SPPF) modules, achieving state-of-the-art trade-offs between mAP and latency.",
        style_body
    ))
    story.append(Paragraph(
        "<i>B. Adverse Weather Enhancement:</i> Classical restoration models target specific physical degradations. Atmospheric "
        "dehazing relies heavily on the Koschmieder scattering model, with He et al. pioneering the Dark Channel Prior (DCP) [3]. "
        "Classical deraining employs edge-preserving bilateral filtering and unsharp high-pass masking to attenuate rain streaks [4]. "
        "Low-light enhancement methods leverage Retinex theory, Contrast Limited Adaptive Histogram Equalization (CLAHE) [6], "
        "and power-law gamma transformations to restore underexposed structural contours.",
        style_body
    ))
    story.append(Paragraph(
        "<i>C. Joint Enhancement and Detection:</i> Recent literature explores combining enhancement with detection. Liu et al. "
        "proposed Image-Adaptive YOLO (IA-YOLO) [1], utilizing a small CNN parameter predictor trained jointly with YOLO. Similarly, "
        "Fog-Aware YOLO [2] and YOLO-D [7] investigate domain adaptation under specific low-light or foggy conditions. However, existing "
        "works frequently evaluate on synthetic datasets (e.g., synthetically generated rain streaks or haze) and do not isolate "
        "classifier routing accuracy from enhancement efficacy across multi-condition benchmarks.",
        style_body
    ))

    # ==========================================
    # SECTION III: SYSTEM ARCHITECTURE & PIPELINES
    # ==========================================
    story.append(Paragraph("III. SYSTEM ARCHITECTURE & PIPELINES", style_sec_heading))
    story.append(Paragraph(
        "Our system implements a hybrid architecture that integrates deep convolutional classification, modular classical computer "
        "vision filters, and deep convolutional detection.",
        style_body
    ))
    story.append(Paragraph(
        "<i>A. Pipeline A: Baseline Control:</i> Let <i>I</i> denote an input RGB driving image. The baseline pipeline executes "
        "direct object detection without spatial pre-processing: <i>Y_hat = YOLO(I)</i>. This serves as the empirical control baseline.",
        style_body
    ))
    story.append(Paragraph(
        "<i>B. Pipeline B: Fixed CLAHE:</i> Evaluates the hypothesis that uniform contrast stretching is universally beneficial. "
        "Image <i>I</i> is converted to CIELAB space, and CLAHE (clip limit = 2.0, tile grid = 8x8) is applied to the L-channel: "
        "<i>Y_hat = YOLO(E_fixed(I))</i>.",
        style_body
    ))
    story.append(Paragraph(
        "<i>C. Pipeline C: Adaptive CNN Router:</i> A ResNet-18 classifier <i>f_theta(I)</i> predicts condition <i>c_hat</i> from five classes: "
        "clear, rain, fog, night, and dawn_dusk. Image <i>I</i> is routed to the corresponding condition-specific enhancer <i>E_c(I)</i>: "
        "<i>Y_hat = YOLO(E_c_hat(I))</i>.",
        style_body
    ))
    story.append(Paragraph(
        "<i>D. Pipeline D: Oracle Routing:</i> Routes image <i>I</i> using its ground-truth condition <i>c*</i>: "
        "<i>Y_hat = YOLO(E_c*(I))</i>, eliminating classification error to measure the theoretical ceiling of adaptive enhancement.",
        style_body
    ))

    # Table I: Primary Pipeline Results
    story.append(Paragraph("TABLE I: PRIMARY PERFORMANCE METRICS ACROSS EXPERIMENTAL PIPELINES (TEST SPLIT N=180)", style_table_title))
    t1_data = [
        [Paragraph("<b>Pipeline</b>", style_table_cell_bold), Paragraph("<b>Enhancement</b>", style_table_cell_bold), Paragraph("<b>Prec.</b>", style_table_cell_bold), Paragraph("<b>Recall</b>", style_table_cell_bold), Paragraph("<b>F1</b>", style_table_cell_bold), Paragraph("<b>mAP50</b>", style_table_cell_bold), Paragraph("<b>mAP95</b>", style_table_cell_bold)],
        [Paragraph("Pipeline A", style_table_cell), Paragraph("None (Baseline)", style_table_cell), Paragraph("0.7866", style_table_cell), Paragraph("0.2579", style_table_cell), Paragraph("0.3885", style_table_cell), Paragraph("0.2029", style_table_cell), Paragraph("0.1130", style_table_cell)],
        [Paragraph("Pipeline B", style_table_cell), Paragraph("Fixed CLAHE", style_table_cell), Paragraph("0.7811", style_table_cell), Paragraph("<b>0.2615</b>", style_table_cell_bold), Paragraph("<b>0.3919</b>", style_table_cell_bold), Paragraph("<b>0.2043</b>", style_table_cell_bold), Paragraph("<b>0.1131</b>", style_table_cell_bold)],
        [Paragraph("Pipeline C", style_table_cell), Paragraph("Adaptive Router", style_table_cell), Paragraph("<b>0.7887</b>", style_table_cell_bold), Paragraph("0.2567", style_table_cell), Paragraph("0.3873", style_table_cell), Paragraph("0.2025", style_table_cell), Paragraph("0.1121", style_table_cell)],
        [Paragraph("Pipeline D", style_table_cell), Paragraph("Oracle Condition", style_table_cell), Paragraph("0.7781", style_table_cell), Paragraph("0.2591", style_table_cell), Paragraph("0.3888", style_table_cell), Paragraph("0.2016", style_table_cell), Paragraph("0.1105", style_table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[46, 57, 28, 28, 26, 34, 34])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    # ==========================================
    # SECTION IV: ENHANCEMENT ALGORITHMS
    # ==========================================
    story.append(Paragraph("IV. CONDITION-SPECIFIC ENHANCEMENT ALGORITHMS", style_sec_heading))
    story.append(Paragraph(
        "<i>A. Fog Dehazing via Dark Channel Prior (DCP):</i> Based on Koschmieder's model, observed intensity is "
        "<i>I(x) = J(x)t(x) + A(1 - t(x))</i>. The dark channel is estimated as:",
        style_body
    ))
    story.append(Paragraph("J_dark(x) = min_{c in {r,g,b}} ( min_{y in Omega(x)} I^c(y) )  &nbsp;&nbsp;&nbsp;&nbsp; (1)", style_equation))
    story.append(Paragraph(
        "Atmospheric light <i>A</i> is estimated from the top 0.1% brightest pixels in <i>J_dark</i>. Transmission map "
        "<i>t(x) = 1 - 0.95 * J_dark(I/A)</i> is refined with a bilateral filter (d=9, sigma_color=0.1, sigma_space=15) "
        "and scene radiance recovered via <i>J(x) = (I(x) - A)/max(t(x), 0.1) + A</i>.",
        style_body
    ))
    story.append(Paragraph(
        "<i>B. Classical Rain Bilateral Filtering:</i> Rain streak attenuation combines edge-preserving bilateral smoothing "
        "(d=7, sigma=50) with unsharp high-pass masking: <i>I_sharp = I_smooth + 1.2 * (I_smooth - G_sigma=2 * I_smooth)</i>, "
        "followed by CLAHE (clip=1.5) on the luminance channel.",
        style_body
    ))
    story.append(Paragraph(
        "<i>C. Low-Light Night Enhancement:</i> Applies power-law expansion: <i>I_gamma = 255 * (I / 255)^gamma</i> with gamma=0.60, "
        "combined with CLAHE (clip=2.5) on the L-channel and subtle 1.15x brightness scaling.",
        style_body
    ))
    story.append(Paragraph(
        "<i>D. Dawn/Dusk Balancing:</i> Corrects dynamic range extremes between sky glare and roadway shadows using gamma=0.85, "
        "CLAHE (clip=1.8), and midtone contrast expansion.",
        style_body
    ))
    story.append(Paragraph(
        "<i>E. Clear Condition Pass-Through:</i> Clear daylight images undergo identity pass-through (<i>E_clear(I) = I</i>) "
        "to prevent artificial filter distortions on clean scenes.",
        style_body
    ))

    # ==========================================
    # SECTION V: DATASET & METRICS
    # ==========================================
    story.append(Paragraph("V. DATASET & PHYSICAL VISIBILITY METRICS", style_sec_heading))
    story.append(Paragraph(
        "We utilize real driving imagery from BDD100K (dgural/bdd100k on Hugging Face). The benchmark comprises 1,197 curated images "
        "with 22,197 ground-truth bounding boxes across 10 vehicle and pedestrian categories. Stratified splitting (seed=42) creates "
        "Train (N=837, 70%), Validation (N=180, 15%), and Test (N=180, 15%) partitions across clear (296), rain (296), fog (13), "
        "night (296), and dawn/dusk (296) scenes.",
        style_body
    ))
    story.append(Paragraph(
        "Physical visibility metrics are computed directly from pixel data: Mean Luminance <i>mu_L = (1/N) sum I_gray</i> and "
        "RMS Contrast <i>sigma_RMS = sqrt((1/N) sum (I_gray - mu_L)^2)</i>. Samples with <i>mu_L < 45.0</i> or <i>sigma_RMS < 35.0</i> "
        "are categorized as derived low-visibility scenes.",
        style_body
    ))

    # ==========================================
    # SECTION VI: EMPIRICAL RESULTS & DISCUSSION
    # ==========================================
    story.append(Paragraph("VI. EMPIRICAL RESULTS AND DISCUSSION", style_sec_heading))
    story.append(Paragraph(
        "<i>A. Overall Pipeline Performance:</i> As reported in Table I, Baseline achieves mAP@50 of 0.2029. Fixed CLAHE yields a slight "
        "overall improvement to 0.2043 (+0.69% relative gain), while Adaptive Enhancement achieves 0.2025 (-0.20% relative change). "
        "Notably, Adaptive Enhancement achieves the highest Precision (0.7887 vs. 0.7866 Baseline).",
        style_body
    ))

    # Table II: Condition-Wise Performance
    story.append(Paragraph("TABLE II: CONDITION-WISE OBJECT DETECTION BREAKDOWN (mAP@50 ACROSS TEST SPLIT)", style_table_title))
    t2_data = [
        [Paragraph("<b>Condition</b>", style_table_cell_bold), Paragraph("<b>N</b>", style_table_cell_bold), Paragraph("<b>Baseline</b>", style_table_cell_bold), Paragraph("<b>Fixed</b>", style_table_cell_bold), Paragraph("<b>Adaptive</b>", style_table_cell_bold), Paragraph("<b>Delta (Adapt-Base)</b>", style_table_cell_bold)],
        [Paragraph("Clear", style_table_cell), Paragraph("45", style_table_cell), Paragraph("0.2267", style_table_cell), Paragraph("0.2238", style_table_cell), Paragraph("0.2263", style_table_cell), Paragraph("-0.0004 (-0.18%)", style_table_cell)],
        [Paragraph("Rain", style_table_cell), Paragraph("44", style_table_cell), Paragraph("0.1841", style_table_cell), Paragraph("0.1744", style_table_cell), Paragraph("0.1787", style_table_cell), Paragraph("-0.0054 (-2.93%)", style_table_cell)],
        [Paragraph("Fog", style_table_cell), Paragraph("2", style_table_cell), Paragraph("0.2449", style_table_cell), Paragraph("0.2286", style_table_cell), Paragraph("0.2449", style_table_cell), Paragraph("0.0000 (0.00%)", style_table_cell)],
        [Paragraph("Night", style_table_cell), Paragraph("45", style_table_cell), Paragraph("0.1545", style_table_cell), Paragraph("<b>0.1580</b>", style_table_cell_bold), Paragraph("0.1456", style_table_cell), Paragraph("<b>-0.0089 (-5.76%)</b>", style_table_cell_bold)],
        [Paragraph("Dawn/Dusk", style_table_cell), Paragraph("44", style_table_cell), Paragraph("0.2378", style_table_cell), Paragraph("<b>0.2529</b>", style_table_cell_bold), Paragraph("<b>0.2489</b>", style_table_cell_bold), Paragraph("<b>+0.0111 (+4.67%)</b>", style_table_cell_bold)],
        [Paragraph("<b>Overall</b>", style_table_cell_bold), Paragraph("<b>180</b>", style_table_cell_bold), Paragraph("0.2029", style_table_cell), Paragraph("<b>0.2043</b>", style_table_cell_bold), Paragraph("0.2025", style_table_cell), Paragraph("-0.0004 (-0.20%)", style_table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[42, 17, 36, 36, 38, 70])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#f9fafb')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    # Figure 1: Condition-Wise Map Plot
    img_cond_map = get_scaled_image(PROJECT_ROOT / "results" / "condition_wise_map.png", max_w=col_w)
    if img_cond_map:
        story.append(img_cond_map)
        story.append(Paragraph("Fig. 1: Condition-wise mAP@50 comparison across experimental pipelines on BDD100K test split.", style_fig_caption))

    story.append(Paragraph(
        "<i>B. Condition-Wise Dissection:</i> Table II reveals marked divergence across environmental domains: "
        "<b>Dawn/Dusk</b> benefits most from enhancement (+4.67% under Adaptive, +6.35% under Fixed CLAHE), where lifting deep roadway "
        "shadows aids vehicle edge detection. Conversely, <b>Night</b> detection suffers significant degradation (-5.76% under Adaptive). "
        "Non-linear gamma expansion amplifies sensor grain and headlight glare, creating false-positive feature activations in YOLO.",
        style_body
    ))
    story.append(Paragraph(
        "<i>C. ResNet-18 Condition Classifier:</i> Evaluated on the blind test split (N=180), the classifier achieved <b>71.67% accuracy</b>. "
        "Night scenes achieved 85.11% precision and 88.89% recall (F1=0.8696); Rain achieved 80.00% precision and 63.64% recall (F1=0.7089); "
        "Clear achieved 60.29% precision and 91.11% recall; Dawn/Dusk achieved 80.00% precision and 45.45% recall.",
        style_body
    ))

    # Table III: Classifier Performance Table
    t3_title = Paragraph("TABLE III: PYTORCH RESNET-18 CLASSIFIER EVALUATION ON TEST SPLIT (N=180)", style_table_title)
    t3_data = [
        [Paragraph("<b>Condition</b>", style_table_cell_bold), Paragraph("<b>Precision</b>", style_table_cell_bold), Paragraph("<b>Recall</b>", style_table_cell_bold), Paragraph("<b>F1-Score</b>", style_table_cell_bold), Paragraph("<b>Support</b>", style_table_cell_bold)],
        [Paragraph("Clear", style_table_cell), Paragraph("0.6029", style_table_cell), Paragraph("0.9111", style_table_cell), Paragraph("0.7257", style_table_cell), Paragraph("45", style_table_cell)],
        [Paragraph("Rain", style_table_cell), Paragraph("0.8000", style_table_cell), Paragraph("0.6364", style_table_cell), Paragraph("0.7089", style_table_cell), Paragraph("44", style_table_cell)],
        [Paragraph("Fog", style_table_cell), Paragraph("0.0000", style_table_cell), Paragraph("0.0000", style_table_cell), Paragraph("0.0000", style_table_cell), Paragraph("2", style_table_cell)],
        [Paragraph("Night", style_table_cell), Paragraph("0.8511", style_table_cell), Paragraph("0.8889", style_table_cell), Paragraph("0.8696", style_table_cell), Paragraph("45", style_table_cell)],
        [Paragraph("Dawn/Dusk", style_table_cell), Paragraph("0.8000", style_table_cell), Paragraph("0.4545", style_table_cell), Paragraph("0.5797", style_table_cell), Paragraph("44", style_table_cell)],
        [Paragraph("<b>Overall / Acc</b>", style_table_cell_bold), Paragraph("<b>0.7423</b>", style_table_cell_bold), Paragraph("<b>0.7167</b>", style_table_cell_bold), Paragraph("<b>0.7138</b>", style_table_cell_bold), Paragraph("<b>180</b>", style_table_cell_bold)],
    ]
    t3 = Table(t3_data, colWidths=[65, 48, 48, 52, 50])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#f9fafb')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(KeepTogether([t3_title, t3, Spacer(1, 4)]))

    # Figure 2: Confusion Matrix Plot
    img_cm = get_scaled_image(PROJECT_ROOT / "results" / "condition_confusion_matrix.png", max_w=col_w)
    if img_cm:
        story.append(img_cm)
        story.append(Paragraph("Fig. 2: Normalized confusion matrix of the PyTorch ResNet-18 condition classifier on blind test imagery.", style_fig_caption))

    # Figure 3: Training Curve
    img_curve = get_scaled_image(PROJECT_ROOT / "results" / "condition_training_curve.png", max_w=col_w)
    if img_curve:
        story.append(img_curve)
        story.append(Paragraph("Fig. 3: Training and validation loss/accuracy curves for condition classification.", style_fig_caption))

    # ==========================================
    # SECTION VII: HYPOTHESIS EVALUATION
    # ==========================================
    story.append(Paragraph("VII. RESEARCH HYPOTHESIS EVALUATION", style_sec_heading))
    story.append(Paragraph(
        "<b>[H1] Adaptive Image Enhancement Improves YOLO Detection Globally: NOT SUPPORTED.</b> "
        "Overall mAP@50 for Adaptive Enhancement (0.2025) is marginally lower than Baseline (0.2029, -0.20%). "
        "While Dawn/Dusk improved by +4.67%, night degradation (-5.76%) offset these gains, demonstrating that classical spatial "
        "filtering cannot universally enhance off-the-shelf deep detectors.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[H2] Condition-Specific Enhancement Outperforms Fixed CLAHE: NOT SUPPORTED GLOBALLY.</b> "
        "Fixed CLAHE achieved mAP@50 of 0.2043 (+0.69%) versus 0.2025 for Adaptive. Uniform CLAHE provides consistent edge contrast "
        "enhancement across scenes, whereas parameterized spatial filtering introduced local artifacts under extreme conditions.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[H3] Greater Benefit Under Adverse than Normal Conditions: CONDITIONALLY SUPPORTED.</b> "
        "Clear imagery exhibited near-zero alteration under adaptive identity pass-through (-0.0004 mAP change), verifying the protection "
        "of uncorrupted data. Dawn/Dusk imagery showed positive response (+0.0111 mAP), though night scenes degraded.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[H4] Task-Oriented Filtering Diverges from Human Visual Quality: EMPIRICALLY CONFIRMED.</b> "
        "Subjective visual brightness (e.g. night gamma expansion) does not correlate with neural detectability. Amplified sensor noise "
        "and glare artifacts degrade deep convolutional activation maps despite appearing clearer to human observers.",
        style_body
    ))

    # Figure 4: Side-by-Side Visual Comparison Panel
    img_vis = get_scaled_image(PROJECT_ROOT / "results" / "visualizations" / "comparison_003_bb5cc516-eb91e8c8.jpg", max_w=col_w)
    if img_vis:
        story.append(img_vis)
        story.append(Paragraph("Fig. 4: Tri-panel visual comparison: [Left: Baseline] vs. [Center: Fixed CLAHE] vs. [Right: Adaptive Router].", style_fig_caption))

    # ==========================================
    # SECTION VIII: CONCLUSION
    # ==========================================
    story.append(Paragraph("VIII. CONCLUSION & FUTURE WORK", style_sec_heading))
    story.append(Paragraph(
        "This paper presented an empirical investigation into adaptive image enhancement for YOLO object detection under adverse weather. "
        "Through systematic benchmarking across 1,197 real BDD100K driving images across five environmental domains, we established that "
        "decoupled classical enhancement pre-processing does not universally improve off-the-shelf deep neural detectors. While transition "
        "light conditions (dawn/dusk) benefit from shadow balancing, low-light night scenes suffer from noise amplification. Future research "
        "must pursue end-to-end differentiable neural enhancement modules where low-level restoration filters are co-optimized with the "
        "downstream detection loss function rather than classical human perceptual heuristics.",
        style_body
    ))

    # ==========================================
    # REFERENCES (IEEE Format)
    # ==========================================
    story.append(Paragraph("REFERENCES", style_sec_heading))
    refs = [
        "[1] W. Liu, G. Ren, R. Yu, S. Guo, J. Zhu, and L. Zhang, 'Image-adaptive YOLO for object detection in adverse weather conditions,' <i>Proc. AAAI Conf. Artif. Intell.</i>, vol. 36, no. 2, pp. 1792–1800, Jun. 2022.",
        "[2] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, 'You only look once: Unified, real-time object detection,' in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 779–788.",
        "[3] K. He, J. Sun, and X. Tang, 'Single image haze removal using dark channel prior,' <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 33, no. 12, pp. 2341–2353, Dec. 2011.",
        "[4] C. Tomasi and R. Manduchi, 'Bilateral filtering for gray and color images,' in <i>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</i>, 1998, pp. 839–846.",
        "[5] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, 'BDD100K: A diverse driving dataset for heterogeneous multitask learning,' in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2020, pp. 2636–2645.",
        "[6] S. M. Pizer, E. P. Amburn, J. D. Austin, R. Cromartie, A. Geselowitz, T. Greer, B. ter Haar Romeny, and J. B. Zimmerman, 'Adaptive histogram equalization and its variations,' <i>Comput. Vis. Graph. Image Process.</i>, vol. 39, no. 3, pp. 355–368, 1987.",
        "[7] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 770–778.",
        "[8] G. Jocher et al., 'Ultralytics YOLO11,' Ultralytics Inc., 2024. [Online]. Available: https://github.com/ultralytics/ultralytics.",
        "[9] D. Gural, 'BDD100K Hugging Face Dataset Mirror,' 2023. [Online]. Available: https://huggingface.co/datasets/dgural/bdd100k.",
        "[10] S. Chang and Y. Yang, 'Detection-oriented image deraining via task-driven contrastive learning,' <i>IEEE Trans. Intell. Transp. Syst.</i>, vol. 26, pp. 1102–1114, 2025."
    ]
    for r in refs:
        story.append(Paragraph(r, style_reference))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"IEEE Paper PDF successfully built at: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_paper()
