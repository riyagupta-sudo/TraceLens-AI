import sqlite3
import json

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, modification_report, integrity_score, risk_score FROM media_items')
rows = cursor.fetchall()
found = False
for r in rows:
    media_id, filename, report_json, integrity, risk = r
    if report_json:
        try:
            report = json.loads(report_json)
        except Exception:
            continue
        
        # Check signal breakdown or investigation summary values
        summary = report.get('investigation_summary', {})
        consensus = summary.get('consensus', {})
        signals = consensus.get('signal_breakdown', {})
        
        # Look for values close to target: AI=79, RF=22, SS=10, Meta=100
        ai = signals.get('ai_score', summary.get('ai_generation_probability', 0))
        rf = int(summary.get('ml_tampering_probability', 0) * 100)
        ss = signals.get('screenshot_prob', summary.get('screenshot_probability', 0))
        meta = signals.get('metadata_trust', report.get('metadata_intelligence', {}).get('metadata_trust_score', 0))
        
        if ai == 79 or rf == 22 or ss == 10 or meta == 100:
            print(f"ID: {media_id} | File: {filename} | Integrity: {integrity} | Risk: {risk}")
            print(f"  AI Score: {ai}% | RF: {rf}% | Screenshot: {ss}% | Metadata Trust: {meta}%")
            found = True

if not found:
    print("No direct database matches. Listing all items with Integrity=100 and Risk=0:")
    for r in rows:
        media_id, filename, report_json, integrity, risk = r
        if integrity == 100 and risk == 0:
            print(f"ID: {media_id} | File: {filename}")
conn.close()
