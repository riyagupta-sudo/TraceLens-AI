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
from PIL import Image

def load_item_data(cursor, item_id):
    cursor.execute('SELECT id, filename, filepath, integrity_score, risk_score, metadata_sig, parent_id, modification_report FROM media_items WHERE id = ?', (item_id,))
    row = cursor.fetchone()
    if not row:
        return None
    
    media_id, filename, filepath, integrity, risk, metadata_json, parent_id, report_json = row
    meta = json.loads(metadata_json) if metadata_json else {}
    report = json.loads(report_json) if report_json else {}
    
    # Map back filepath from relative URL /media/uploads/ to local disk
    local_path = filepath
    if filepath.startswith("/media/uploads/"):
        local_path = os.path.join(backend_dir, "app", "uploads", os.path.basename(filepath))
        
    return {
        "id": media_id,
        "filename": filename,
        "filepath": local_path,
        "integrity": integrity,
        "risk": risk,
        "metadata": meta,
        "parent_id": parent_id,
        "report": report
    }

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()

# Load ID 188
item = load_item_data(cursor, 188)
if item:
    print(f"Loaded Item {item['filename']} (ID: {item['id']})")
    
    # Load parent metadata if any
    parent_meta = None
    if item['parent_id']:
        parent = load_item_data(cursor, item['parent_id'])
        if parent:
            parent_meta = parent['metadata']
            parent_meta['integrity_score'] = parent['integrity']
            parent_meta['phash'] = parent['report'].get('phash', '')
            parent_meta['dhash'] = parent['report'].get('dhash', '')
            parent_meta['ahash'] = parent['report'].get('ahash', '')
            parent_meta['embedding'] = parent['metadata'].get('embedding', [])
            parent_meta['filepath'] = parent['filepath']
            print(f"Loaded parent context: {parent['filename']} (ID: {item['parent_id']})")

    # Let's run standalone (no parent) first to see the pure score changes
    print("\n--- Running STANDALONE ---")
    int_std, risk_std, fore_std = calculate_integrity_and_risk(
        item['filepath'],
        item['metadata'].copy(),
        "image/jpeg",
        item['report'].get('phash', ''),
        parent_metadata=None
    )
    
    # Let's run with parent context to match the database state
    print("\n--- Running with PARENT CONTEXT ---")
    int_parent, risk_parent, fore_parent = calculate_integrity_and_risk(
        item['filepath'],
        item['metadata'].copy(),
        "image/jpeg",
        item['report'].get('phash', ''),
        parent_metadata=parent_meta
    )
    
    print("\n==============================================================")
    print("BEFORE / AFTER REPAIR SCORES FOR ID 188 (STANDALONE VS PARENT)")
    print("==============================================================")
    
    # Original values from report JSON before repair
    orig_summary = item['report'].get('investigation_summary', {})
    orig_consensus = orig_summary.get('consensus', {})
    
    print(f"1. STANDALONE (No parent context):")
    print(f"   Before Repair: Integrity = 100 | Risk = 0")
    print(f"   After Repair:  Integrity = {int_std} | Risk = {risk_std}")
    print(f"   Consensus State: {fore_std.get('consensus', {}).get('state')}")
    
    print(f"\n2. WITH PARENT (Database state):")
    print(f"   Before Repair: Integrity = {item['integrity']} | Risk = {item['risk']}")
    print(f"   After Repair:  Integrity = {int_parent} | Risk = {risk_parent}")
    print(f"   Consensus State: {fore_parent.get('consensus', {}).get('state')}")
    
else:
    print("Item 188 not found")

conn.close()
