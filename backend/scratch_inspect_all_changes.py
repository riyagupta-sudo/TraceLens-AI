import sqlite3
import json
import os
import sys

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
app_dir = os.path.join(backend_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from app.dna_engine import calculate_integrity_and_risk

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, filepath, integrity_score, risk_score, metadata_sig, parent_id, modification_report FROM media_items')
rows = cursor.fetchall()

print(f"{'ID':<4} | {'Filename':<30} | {'Before Int/Risk':<16} | {'After Int/Risk':<16} | {'CASIA':<5} | {'RF':<3} | {'Stego':<5} | {'Consensus (Bf -> Af)'}")
print("-" * 125)

for r in rows:
    media_id, filename, filepath, integrity, risk, metadata_json, parent_id, report_json = r
    meta = json.loads(metadata_json) if metadata_json else {}
    report = json.loads(report_json) if report_json else {}
    
    local_path = filepath
    if filepath.startswith("/media/uploads/"):
        local_path = os.path.join(backend_dir, "app", "uploads", os.path.basename(filepath))
        
    if not os.path.exists(local_path):
        continue
        
    # Get parent context
    parent_meta = None
    if parent_id:
        cursor.execute('SELECT metadata_sig, integrity_score, modification_report, filepath FROM media_items WHERE id = ?', (parent_id,))
        p_row = cursor.fetchone()
        if p_row:
            p_meta_json, p_int, p_rep_json, p_filepath = p_row
            parent_meta = json.loads(p_meta_json) if p_meta_json else {}
            parent_meta['integrity_score'] = p_int
            p_rep = json.loads(p_rep_json) if p_rep_json else {}
            parent_meta['phash'] = p_rep.get('phash', '')
            parent_meta['dhash'] = p_rep.get('dhash', '')
            parent_meta['ahash'] = p_rep.get('ahash', '')
            parent_meta['embedding'] = parent_meta.get('embedding', [])
            
            # Map parent filepath to local path
            if p_filepath.startswith("/media/uploads/"):
                parent_meta['filepath'] = os.path.join(backend_dir, "app", "uploads", os.path.basename(p_filepath))
            else:
                parent_meta['filepath'] = p_filepath
            
    int_after, risk_after, fore_after = calculate_integrity_and_risk(
        local_path,
        meta.copy(),
        "image/png" if filename.lower().endswith(".png") else "image/jpeg",
        report.get('phash', ''),
        parent_metadata=parent_meta
    )
    
    casia = report.get('investigation_summary', {}).get('casia_tampering_probability', 0)
    rf = int(report.get('investigation_summary', {}).get('ml_tampering_probability', 0) * 100)
    stego = report.get('investigation_summary', {}).get('steganography_suspicion', 0)
    
    bf_state = report.get('investigation_summary', {}).get('consensus', {}).get('state', 'N/A')
    af_state = fore_after.get('consensus', {}).get('state', 'N/A')
    
    if (integrity != int_after) or (risk != risk_after):
        print(f"{media_id:<4} | {filename:<30} | {integrity:<3}/{risk:<3}            | {int_after:<3}/{risk_after:<3}            | {casia:<5} | {rf:<3} | {stego:<5} | {bf_state} -> {af_state}")

conn.close()
