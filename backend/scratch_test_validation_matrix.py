import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
app_dir = os.path.join(backend_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from app.dna_engine import resolve_forensic_consensus

print("Running Synthetic Consensus Validation Matrix...")

cases = [
    {
        "name": "Case A",
        "inputs": {"ai_score": 98, "rf_prob": 15, "metadata_trust": 15, "screenshot_prob": 10, "stego_susp": 10, "casia_prob": 13},
        "expected": "MIXED_SIGNALS"
    },
    {
        "name": "Case B",
        "inputs": {"ai_score": 98, "rf_prob": 80, "metadata_trust": 15, "screenshot_prob": 10, "stego_susp": 10, "casia_prob": 10},
        "expected": "HIGH_CONFIDENCE_AI_GENERATED"
    },
    {
        "name": "Case C",
        "inputs": {"ai_score": 15, "rf_prob": 25, "metadata_trust": 85, "screenshot_prob": 5, "stego_susp": 5, "casia_prob": 10},
        "expected": "LIKELY_AUTHENTIC"
    },
    {
        "name": "Case D",
        "inputs": {"ai_score": 65, "rf_prob": 55, "metadata_trust": 70, "screenshot_prob": 5, "stego_susp": 5, "casia_prob": 10},
        "expected": "INVESTIGATE_FURTHER"
    }
]

print(f"{'Case':<8} | {'AI':<3} | {'RF':<3} | {'Meta':<4} | {'SS':<3} | {'Stego':<5} | {'Expected':<30} | {'Actual':<30} | {'Status'}")
print("-" * 115)

for c in cases:
    inp = c["inputs"]
    metadata_stripped_possible = (
        inp["metadata_trust"] <= 20
        and inp["rf_prob"] < 40
        and inp["stego_susp"] < 20
        and inp["casia_prob"] < 20
    )
    res = resolve_forensic_consensus(
        ai_score=inp["ai_score"],
        rf_prob=inp["rf_prob"],
        metadata_trust=inp["metadata_trust"],
        screenshot_prob=inp["screenshot_prob"],
        stego_susp=inp["stego_susp"],
        casia_prob=inp["casia_prob"],
        metadata_stripped_possible=metadata_stripped_possible
    )
    actual = res["state"]
    status = "PASS" if actual == c["expected"] else "FAIL"
    print(f"{c['name']:<8} | {inp['ai_score']:<3} | {inp['rf_prob']:<3} | {inp['metadata_trust']:<4} | {inp['screenshot_prob']:<3} | {inp['stego_susp']:<5} | {c['expected']:<30} | {actual:<30} | {status}")
