import sqlite3

for db_file in ['tracelens.db', 'tracelens_backup.db']:
    print(f"\n--- Checking {db_file} ---")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, filename, filepath, modification_report FROM media_items")
        rows = cursor.fetchall()
        for r in rows:
            report_str = str(r[3])
            if '1000111612' in report_str or '1612' in report_str:
                print(f"MATCH: ID: {r[0]}, Filename: {r[1]}, Filepath: {r[2]}")
                print(f"Report: {report_str[:500]}...")
    except Exception as e:
        print("Error:", e)
    conn.close()
