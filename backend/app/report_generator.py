import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import Dict, Any, List

def generate_pdf_report(
    media_item: Dict[str, Any], 
    matches: List[Dict[str, Any]], 
    output_path: str,
    case_name: str = "Unassigned Case"
) -> str:
    """
    Generates a dark-themed PDF forensic report for a media item.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Page layout setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Custom styles matching cyber intelligence theme
    styles = getSampleStyleSheet()
    
    # Base text styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=colors.HexColor('#00E5FF'), # Neon Cyan
        spaceAfter=10,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.HexColor('#00FF9D'), # Neon Green
        spaceAfter=30,
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#00E5FF'),
        spaceBefore=15,
        spaceAfter=10,
        borderColor=colors.HexColor('#7C3AED'), # Violet border
        borderWidth=1,
        borderPadding=6,
        borderRadius=4
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#E5E5E5'), # Off-white
        spaceAfter=8,
        leading=14
    )
    
    body_bold_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    code_style = ParagraphStyle(
        'ReportCode',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        textColor=colors.HexColor('#00FF9D'),
        backColor=colors.HexColor('#1A1A1A'),
        borderPadding=6,
        spaceAfter=8
    )
    
    story = []
    
    # --- PAGE 1: COVER PAGE ---
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("TRACELENS AI", title_style))
    story.append(Paragraph("MEDIA DNA & FORENSIC INTELLIGENCE ENGINE", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Cover metadata table
    meta_data = [
        [Paragraph("REPORT TYPE:", body_bold_style), Paragraph("Forensic Media Intelligence Log", body_style)],
        [Paragraph("INVESTIGATION CASE:", body_bold_style), Paragraph(case_name, body_style)],
        [Paragraph("TARGET FILE:", body_bold_style), Paragraph(media_item.get('filename', 'Unknown'), body_style)],
        [Paragraph("FILE FORMAT / TYPE:", body_bold_style), Paragraph(f"{media_item.get('mime_type', 'Unknown')} ({media_item.get('resolution', 'N/A')})", body_style)],
        [Paragraph("SHA256 CHECKSUM:", body_bold_style), Paragraph(media_item.get('sha256', 'N/A'), code_style)],
        [Paragraph("GENERATED DATE:", body_bold_style), Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)]
    ]
    
    meta_table = Table(meta_data, colWidths=[2.0*inch, 5.0*inch])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
    ]))
    
    story.append(meta_table)
    story.append(PageBreak())
    
    # --- PAGE 2: EXECUTIVE SUMMARY & RISK ASSESSMENT ---
    story.append(Paragraph("EXECUTIVE SUMMARY", h1_style))
    story.append(Paragraph(
        f"This forensic report details the structural, perceptual, and semantic validation check for file <b>{media_item.get('filename')}</b>. "
        f"TraceLens AI has cross-matched the fingerprint signatures against the global repository index containing related records. "
        f"The integrity validation pipeline estimates the origin source and lists modifications applied to the asset.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Scores Grid
    integrity = media_item.get('integrity_score', 100)
    risk = media_item.get('risk_score', 0)
    
    score_data = [
        [
            Paragraph(f"<font color='#00FF9D' size='22'><b>{integrity} / 100</b></font><br/>DNA INTEGRITY SCORE", body_bold_style),
            Paragraph(f"<font color='#FF3366' size='22'><b>{risk} / 100</b></font><br/>RISK SCORING ASSESSMENT", body_bold_style)
        ]
    ]
    score_table = Table(score_data, colWidths=[3.5*inch, 3.5*inch])
    score_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#121212')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 15),
        ('BOTTOMPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 15))
    
    # DNA HASHE TABLE
    story.append(Paragraph("MEDIA DNA PROFILE", h1_style))
    dna_data = [
        [Paragraph("DNA Hash Component", body_bold_style), Paragraph("Fingerprint Hex / Binary Signature", body_bold_style)],
        [Paragraph("SHA-256", body_style), Paragraph(media_item.get('sha256', 'N/A'), code_style)],
        [Paragraph("Perceptual Hash (pHash)", body_style), Paragraph(media_item.get('phash', 'N/A'), code_style)],
        [Paragraph("Difference Hash (dHash)", body_style), Paragraph(media_item.get('dhash', 'N/A'), code_style)],
        [Paragraph("Average Hash (aHash)", body_style), Paragraph(media_item.get('ahash', 'N/A'), code_style)]
    ]
    
    if media_item.get('audio_fingerprint') and media_item['audio_fingerprint'].get('has_audio'):
        chroma = media_item['audio_fingerprint'].get('mean_chroma', [])
        chroma_str = ", ".join([f"{v:.2f}" for v in chroma[:6]]) + "..."
        dna_data.append([Paragraph("Audio Chroma Fingerprint", body_style), Paragraph(chroma_str, code_style)])
        
    dna_table = Table(dna_data, colWidths=[2.2*inch, 4.8*inch])
    dna_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00E5FF')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(dna_table)
    
    # --- PAGE 3: BLIND FORENSIC INVESTIGATION REPORT ---
    story.append(PageBreak())
    story.append(Paragraph("FORENSIC INVESTIGATION REPORT", h1_style))
    
    # 1. Executive Summary
    mod_report = media_item.get('modification_report', {})
    exec_sum = mod_report.get("executive_summary", {})
    findings_txt = exec_sum.get("findings", "No standalone findings recorded.")
    conclusions = exec_sum.get("conclusions", [])
    overall_conf = exec_sum.get("confidence_score", 50)
    
    story.append(Paragraph("<b>Executive Summary:</b>", body_bold_style))
    story.append(Paragraph(findings_txt, body_style))
    story.append(Spacer(1, 6))
    
    if conclusions:
        story.append(Paragraph("<b>Key Conclusions:</b>", body_bold_style))
        for c in conclusions:
            story.append(Paragraph(f"• {c}", body_style))
        story.append(Spacer(1, 10))
        
    # Confidence Score & Factors Table
    conf_factors = exec_sum.get("confidence_factors", {})
    factors_rows = [
        [Paragraph("<b>Confidence Metric</b>", body_bold_style), Paragraph("<b>Score</b>", body_bold_style),
         Paragraph("<b>Confidence Metric</b>", body_bold_style), Paragraph("<b>Score</b>", body_bold_style)]
    ]
    
    # Let's pair them up for a 2-column key-value layout
    factor_items = list(conf_factors.items())
    for idx in range(0, len(factor_items), 2):
        row = []
        name1 = factor_items[idx][0].replace('_', ' ').title()
        val1 = f"{factor_items[idx][1]}%"
        row.extend([Paragraph(name1, body_style), Paragraph(f"<font color='#00FF9D'><b>{val1}</b></font>", body_bold_style)])
        
        if idx + 1 < len(factor_items):
            name2 = factor_items[idx+1][0].replace('_', ' ').title()
            val2 = f"{factor_items[idx+1][1]}%"
            row.extend([Paragraph(name2, body_style), Paragraph(f"<font color='#00FF9D'><b>{val2}</b></font>", body_bold_style)])
        else:
            row.extend([Paragraph("", body_style), Paragraph("", body_style)])
        factors_rows.append(row)
        
    factors_table = Table(factors_rows, colWidths=[2.2*inch, 1.2*inch, 2.2*inch, 1.2*inch])
    factors_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(Paragraph(f"<b>Overall Investigation Confidence: <font color='#00E5FF'>{overall_conf}%</font></b>", body_bold_style))
    story.append(Spacer(1, 6))
    story.append(factors_table)
    story.append(Spacer(1, 15))
    
    # 2. Technical Profile
    tech_prof = mod_report.get("technical_profile", {})
    comp_ind = tech_prof.get("compression_indicators", {}) or {}
    jpeg_q = comp_ind.get("jpeg_quality")
    block_idx = comp_ind.get("blockiness", 1.0)
    
    jpeg_q_str = f"{jpeg_q}%" if jpeg_q is not None else "N/A"
    
    story.append(Paragraph("TECHNICAL PROFILE", h1_style))
    tech_data = [
        [Paragraph("<b>Property</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style),
         Paragraph("<b>Property</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style)],
        [Paragraph("Resolution", body_style), Paragraph(tech_prof.get("resolution", "N/A"), body_style),
         Paragraph("EXIF Status", body_style), Paragraph(tech_prof.get("exif_status", "N/A"), body_style)],
        [Paragraph("Format / MIME", body_style), Paragraph(tech_prof.get("format", "N/A"), body_style),
         Paragraph("JPEG Quality (Est.)", body_style), Paragraph(jpeg_q_str, body_style)],
        [Paragraph("File Size", body_style), Paragraph(f"{tech_prof.get('file_size', 0):,} bytes", body_style),
         Paragraph("Blockiness Index", body_style), Paragraph(f"{block_idx:.2f}", body_style)]
    ]
    tech_table = Table(tech_data, colWidths=[1.8*inch, 1.7*inch, 1.8*inch, 1.7*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(tech_table)
    
    # --- PAGE 4: DETAILED FORENSIC FINDINGS & INSIGHTS ---
    story.append(PageBreak())
    story.append(Paragraph("DETAILED FORENSIC FINDINGS", h1_style))
    
    # 3. Forensic Findings
    forensic_finds = mod_report.get("forensic_findings", [])
    finds_rows = [
        [Paragraph("<b>Forensic Check</b>", body_bold_style), Paragraph("<b>Status</b>", body_bold_style),
         Paragraph("<b>Conf.</b>", body_bold_style), Paragraph("<b>Evidence / Rationale</b>", body_bold_style)]
    ]
    
    for f in forensic_finds:
        status = f.get("status", "Inconclusive")
        if status == "Detected":
            status_html = "<font color='#FF3366'><b>DETECTED</b></font>"
        elif status == "Not Detected":
            status_html = "<font color='#00FF9D'><b>NOT DETECTED</b></font>"
        else:
            status_html = "<font color='#FFCC00'><b>INCONCLUSIVE</b></font>"
            
        evidence_list = f.get("evidence", [])
        evidence_html = "<br/>".join([f"• {e}" for e in evidence_list])
        
        finds_rows.append([
            Paragraph(f.get("finding", "Unknown Check"), body_bold_style),
            Paragraph(status_html, body_style),
            Paragraph(f"{f.get('confidence', 50)}%", body_style),
            Paragraph(evidence_html, body_style)
        ])
        
    finds_table = Table(finds_rows, colWidths=[1.8*inch, 1.2*inch, 0.7*inch, 3.3*inch])
    finds_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(finds_table)
    story.append(Spacer(1, 15))
    
    # 4. Relationship Analysis
    story.append(Paragraph("RELATIONSHIP ANALYSIS", h1_style))
    rel_analysis = mod_report.get("relationship_analysis", {})
    
    origin_conf = rel_analysis.get("origin_confidence")
    origin_prob = rel_analysis.get("origin_probability")
    origin_undet = rel_analysis.get("origin_undetermined")
    origin_status = "Undetermined / Probabilistic" if origin_undet else "Determined Baseline"
    
    rel_data = [
        [Paragraph("<b>Metric</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style),
         Paragraph("<b>Metric</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style)],
        [Paragraph("Related Assets", body_style), Paragraph(str(rel_analysis.get("related_assets_count", 0)), body_style),
         Paragraph("Classification Type", body_style), Paragraph(rel_analysis.get("relationship_type", "N/A"), body_style)],
        [Paragraph("Probable Origin File", body_style), Paragraph(rel_analysis.get("probable_origin_asset", "N/A"), body_style),
         Paragraph("Origin Sim Confidence", body_style), Paragraph(f"{rel_analysis.get('confidence_score', 0)}%", body_style)]
    ]
    
    if origin_conf is not None:
        rel_data.extend([
            [Paragraph("Origin Confidence", body_style), Paragraph(f"{origin_conf}%", body_style),
             Paragraph("Origin Probability", body_style), Paragraph(f"{origin_prob}%", body_style)],
            [Paragraph("Origin Status", body_style), Paragraph(origin_status, body_style),
             Paragraph("", body_style), Paragraph("", body_style)]
        ])
        
    rel_table = Table(rel_data, colWidths=[1.8*inch, 1.7*inch, 1.8*inch, 1.7*inch])
    rel_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(rel_table)
    story.append(Spacer(1, 15))
    
    # 5. Investigation Insights
    story.append(Paragraph("INVESTIGATION INSIGHTS", h1_style))
    insights = mod_report.get("investigation_insights", {})
    
    insights_list = [
        ("Platform Redistribution Risk", insights.get("possible_redistribution", "N/A")),
        ("Social Media Repost Signatures", insights.get("possible_social_media_repost", "N/A")),
        ("Messaging App Recompression Indicators", insights.get("possible_messaging_app_recompression", "N/A")),
        ("Content Frame Stability", insights.get("content_stability_assessment", "N/A"))
    ]
    
    for title, desc in insights_list:
        story.append(Paragraph(f"<b>{title}:</b>", body_bold_style))
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 4))
        
    story.append(Spacer(1, 15))
    
    # SIMILARITY LOG
    story.append(Paragraph("CROSS-PLATFORM SIMILARITY MATCHES", h1_style))
    story.append(Paragraph(
        f"The similarity matching algorithm mapped this asset against similar database media profiles. "
        f"A total of <b>{len(matches)}</b> matching assets were identified.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    match_rows = [
        [Paragraph("ID", body_bold_style), Paragraph("Filename", body_bold_style), Paragraph("Relation Type", body_bold_style), Paragraph("Similarity Score", body_bold_style)]
    ]
    
    for match in matches:
        match_rows.append([
            Paragraph(str(match.get('id', 'N/A')), body_style),
            Paragraph(match.get('filename', 'Unknown'), body_style),
            Paragraph(match.get('relationship_type', 'Variant'), body_style),
            Paragraph(f"<font color='#00FF9D'><b>{int(match.get('combined_score', 0) * 100)}%</b></font>", body_bold_style)
        ])
        
    if len(match_rows) == 1:
        match_rows.append([Paragraph("No similar variations registered in matching case indices.", body_style), Paragraph("", body_style), Paragraph("", body_style), Paragraph("", body_style)])
        
    match_table = Table(match_rows, colWidths=[0.8*inch, 3.2*inch, 1.8*inch, 1.2*inch])
    match_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(match_table)
    
    # --- PAGE 5: LOCALIZED AI EDITING ANALYSIS ---
    ai_edit_json = media_item.get("ai_edit_analysis_json")
    if isinstance(ai_edit_json, str):
        import json
        try:
            ai_edit_json = json.loads(ai_edit_json)
        except Exception:
            ai_edit_json = None
            
    if ai_edit_json:
        story.append(PageBreak())
        story.append(Paragraph("LOCALIZED AI EDITING ANALYSIS", h1_style))
        story.append(Paragraph(
            "This section details localized AI-assisted editing/inpainting forensics (e.g. object removals, content fills). "
            "The analysis scans for regional texture smoothness, ELA variances, noise residual consistency, "
            "periodic FFT frequency grids, and JPEG block boundary disruptions.",
            body_style
        ))
        story.append(Spacer(1, 10))
        
        prob = ai_edit_json.get("editing_probability", 0)
        conf = ai_edit_json.get("confidence", "Low")
        
        edit_score_data = [
            [
                Paragraph(f"<font color='#FFCC00' size='18'><b>{prob}%</b></font><br/>AI EDITED PROBABILITY", body_bold_style),
                Paragraph(f"<font color='#00E5FF' size='18'><b>{conf}</b></font><br/>ANALYSIS CONFIDENCE", body_bold_style)
            ]
        ]
        edit_score_table = Table(edit_score_data, colWidths=[3.5*inch, 3.5*inch])
        edit_score_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#121212')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#2A2A2A')),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(edit_score_table)
        story.append(Spacer(1, 15))
        
        # Suspicious Regions List
        story.append(Paragraph("SUSPICIOUS DETECTED REGIONS", body_bold_style))
        regions = ai_edit_json.get("suspicious_regions", [])
        if regions:
            reg_rows = [
                [Paragraph("<b>Region Box (X, Y, W, H)</b>", body_bold_style), 
                 Paragraph("<b>Conf.</b>", body_bold_style), 
                 Paragraph("<b>Forensic Signals / Reason</b>", body_bold_style)]
            ]
            for r in regions:
                box_str = f"{r.get('x_pct')}% x {r.get('y_pct')}% ({r.get('width_pct')}% x {r.get('height_pct')}%)"
                reg_rows.append([
                    Paragraph(box_str, body_style),
                    Paragraph(f"<font color='#FFCC00'><b>{r.get('confidence')}</b></font> ({r.get('score')}%)", body_style),
                    Paragraph(r.get("reason", "Forensic anomaly"), body_style)
                ])
            reg_table = Table(reg_rows, colWidths=[2.3*inch, 1.5*inch, 3.2*inch])
            reg_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(reg_table)
        else:
            story.append(Paragraph("• No localized suspicious regions or inpainting boundaries detected. Image structure is locally consistent.", body_style))
            
        story.append(Spacer(1, 15))
        
        # Image-wide signals table
        signals = ai_edit_json.get("signals", {})
        story.append(Paragraph("CONTRIBUTING FORENSIC SIGNS", body_bold_style))
        sig_data = [
            [Paragraph("<b>Forensic Metric</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style),
             Paragraph("<b>Forensic Metric</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style)],
            [Paragraph("ELA Inconsistency", body_style), Paragraph(f"{signals.get('ela_inconsistency', 0):.4f}", body_style),
             Paragraph("Noise Residual Variance", body_style), Paragraph(f"{signals.get('noise_residual_var', 0):.4f}", body_style)],
            [Paragraph("JPEG Block Inconsistency", body_style), Paragraph(f"{signals.get('jpeg_block_inconsistency', 0):.4f}", body_style),
             Paragraph("Average Laplacian Variance", body_style), Paragraph(f"{signals.get('laplacian_variance_avg', 0):.2f}", body_style)],
            [Paragraph("Average Local Entropy", body_style), Paragraph(f"{signals.get('local_entropy_avg', 0):.2f}", body_style),
             Paragraph("", body_style), Paragraph("", body_style)]
        ]
        sig_table = Table(sig_data, colWidths=[2.0*inch, 1.5*inch, 2.0*inch, 1.5*inch])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E1E2F')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2A2A2A')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(sig_table)
    
    # Canvas background painter function for dark mode
    def make_dark_bg(canvas, doc):
        canvas.saveState()
        # Dark charcoal (#0A0A0A)
        canvas.setFillColor(colors.HexColor('#0A0A0A'))
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=True, stroke=False)
        
        # Draw header bar in neon cyan
        canvas.setFillColor(colors.HexColor('#00E5FF'))
        canvas.rect(0, doc.pagesize[1] - 4, doc.pagesize[0], 4, fill=True, stroke=False)
        
        # Footer text
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(0.5*inch, 0.4*inch, "TraceLens AI - Forensic Intelligence Database Log. UNCLASSIFIED // OSINT EXPORT.")
        canvas.drawRightString(doc.pagesize[0] - 0.5*inch, 0.4*inch, f"Page {doc.page}")
        canvas.restoreState()
        
    doc.build(story, onFirstPage=make_dark_bg, onLaterPages=make_dark_bg)
    return output_path
