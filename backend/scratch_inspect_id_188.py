import sqlite3
import json

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, filepath, modification_report, integrity_score, risk_score FROM media_items WHERE id = 188')
row = cursor.fetchone()
if row:
    media_id, filename, filepath, report_json, integrity, risk = row
    print(f"ID: {media_id}")
    print(f"Filename: {filename}")
    print(f"Filepath: {filepath}")
    print(f"Integrity Score: {integrity}")
    print(f"Risk Score: {risk}")
    if report_json:
        report = json.loads(report_json)
        print("Report JSON:")
        print(json.dumps(report, indent=2))
else:
    print("Record 188 not found")
conn.close()
