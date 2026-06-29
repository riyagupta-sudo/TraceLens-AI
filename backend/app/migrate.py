import sqlite3
import os

def run_migration():
    print("Running database migrations...")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tracelens.db")
    print(f"Connecting to database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Add cluster_id column to media_items if missing
    cursor.execute("PRAGMA table_info(media_items)")
    columns = [col[1] for col in cursor.fetchall()]
    if "cluster_id" not in columns:
        print("Adding column 'cluster_id' to table 'media_items'...")
        cursor.execute("ALTER TABLE media_items ADD COLUMN cluster_id VARCHAR")
        conn.commit()
        print("Column 'cluster_id' added successfully.")
    else:
        print("Column 'cluster_id' already exists in 'media_items'.")
        
    # 2. Create cluster_merge_recommendations table if missing
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_merge_recommendations'")
    table_exists = cursor.fetchone()
    if not table_exists:
        print("Creating table 'cluster_merge_recommendations'...")
        cursor.execute("""
            CREATE TABLE cluster_merge_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                source_cluster_id VARCHAR NOT NULL,
                target_cluster_id VARCHAR NOT NULL,
                confidence FLOAT NOT NULL,
                status VARCHAR DEFAULT 'Pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        print("Table 'cluster_merge_recommendations' created successfully.")
    else:
        print("Table 'cluster_merge_recommendations' already exists.")
        
    # 3. Add source_type column to osint_results if missing
    cursor.execute("PRAGMA table_info(osint_results)")
    columns = [col[1] for col in cursor.fetchall()]
    if "source_type" not in columns:
        print("Adding column 'source_type' to table 'osint_results'...")
        cursor.execute("ALTER TABLE osint_results ADD COLUMN source_type VARCHAR")
        conn.commit()
        print("Column 'source_type' added successfully.")
    else:
        print("Column 'source_type' already exists in 'osint_results'.")
        
    # 4. Add localized AI editing analysis columns to media_items if missing
    cursor.execute("PRAGMA table_info(media_items)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "ai_edit_analysis_version" not in columns:
        print("Adding column 'ai_edit_analysis_version' to table 'media_items'...")
        cursor.execute("ALTER TABLE media_items ADD COLUMN ai_edit_analysis_version VARCHAR")
        conn.commit()
        print("Column 'ai_edit_analysis_version' added.")
        
    if "ai_edit_analysis_timestamp" not in columns:
        print("Adding column 'ai_edit_analysis_timestamp' to table 'media_items'...")
        cursor.execute("ALTER TABLE media_items ADD COLUMN ai_edit_analysis_timestamp TIMESTAMP")
        conn.commit()
        print("Column 'ai_edit_analysis_timestamp' added.")
        
    if "ai_edit_analysis_json" not in columns:
        print("Adding column 'ai_edit_analysis_json' to table 'media_items'...")
        cursor.execute("ALTER TABLE media_items ADD COLUMN ai_edit_analysis_json JSON")
        conn.commit()
        print("Column 'ai_edit_analysis_json' added.")
        
    conn.close()
    print("Migrations completed successfully.")

if __name__ == "__main__":
    run_migration()
