import sqlite3
import json

conn = sqlite3.connect('tracelens.db')
cursor = conn.cursor()
cursor.execute('SELECT modification_report FROM media_items WHERE id = 188')
row = cursor.fetchone()
if row and row[0]:
    report = json.loads(row[0])
    print("Report Keys:", list(report.keys()))
    if "metadata" in report:
        print("metadata:", report["metadata"])
    else:
        # Check where blockiness, fft are stored in other sub-dicts
        # Let's do a recursive search for blockiness, fft, stego, casia, etc.
        def recurse_find(d, indent=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    if any(x in k.lower() for x in ["blockiness", "fft", "anomaly", "ela", "stego", "casia", "rf"]):
                        print(f"{indent}{k}: {type(v)} -> {v}")
                    if isinstance(v, (dict, list)):
                        recurse_find(v, indent + "  ")
            elif isinstance(d, list):
                for item in d:
                    recurse_find(item, indent)
        recurse_find(report)
conn.close()
