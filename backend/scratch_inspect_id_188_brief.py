import sqlite3
import json

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, filepath, integrity_score, risk_score, modification_report FROM media_items WHERE id = 188')
row = cursor.fetchone()
if row:
    media_id, filename, filepath, integrity, risk, report_json = row
    print("ID:", media_id)
    print("Filename:", filename)
    print("Filepath:", filepath)
    print("Integrity:", integrity)
    print("Risk:", risk)
    if report_json:
        report = json.loads(report_json)
        summary = report.get("investigation_summary", {})
        print("AI Artifacts (adjusted):", summary.get("adjusted_ai_artifact_score"))
        print("RF Probability:", summary.get("ml_tampering_probability"))
        print("Screenshot Probability:", summary.get("screenshot_probability"))
        print("Metadata Trust:", report.get("metadata_intelligence", {}).get("metadata_trust_score"))
        print("CASIA Probability:", summary.get("casia_tampering_probability"))
        print("Stego Suspicion:", summary.get("steganography_suspicion"))
        print("ELA blockiness:", report.get("metadata", {}).get("blockiness"))
        
        # Let's see if ELA blockiness is in there
        # Let's inspect the whole metadata dictionary keys
        print("Metadata keys:", report.get("metadata", {}).keys())
        print("blockiness:", report.get("metadata", {}).get("blockiness"))
        print("fft:", report.get("metadata", {}).get("fft"))
        print("fft_triggered:", report.get("metadata", {}).get("fft_triggered"))
else:
    print("Not found")
conn.close()
