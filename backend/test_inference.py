import sys
import os

# Ensure backend and backend/app are in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
app_dir = os.path.join(backend_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from app.database import SessionLocal
from app.models import MediaItem
from app.dna_engine import calculate_integrity_and_risk

db = SessionLocal()
try:
    # Get any item that was seed/uploaded
    item = db.query(MediaItem).order_by(MediaItem.id.desc()).first()
    if item:
        print(f"Testing with item {item.id}: {item.filename}")
        metadata = item.metadata_sig or {}
        # Map relative web path to absolute file path
        rel_path = item.filepath.replace("/media/uploads/", "")
        abs_path = os.path.join(backend_dir, "app", "uploads", rel_path)
        # Simulate calculate_integrity_and_risk
        # We need parent metadata if it was derived, but we can pass None for testing
        print("Calling calculate_integrity_and_risk...")
        integrity, risk, forensics = calculate_integrity_and_risk(
            abs_path,
            metadata,
            item.mime_type,
            item.phash or "",
            None
        )
        print("Results:")
        print("  Integrity:", integrity)
        print("  Risk:", risk)
        print("  ml_tampering_probability:", forensics.get("ml_tampering_probability"))
        print("  ml_classification:", forensics.get("ml_classification"))
        if "investigation_summary" in forensics:
            print("  investigation_summary ml_tampering_probability:", forensics["investigation_summary"].get("ml_tampering_probability"))
            print("  investigation_summary ml_classification:", forensics["investigation_summary"].get("ml_classification"))
    else:
        print("No items found in database.")
finally:
    db.close()
