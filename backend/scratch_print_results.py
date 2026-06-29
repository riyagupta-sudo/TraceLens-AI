import json

with open("backend/ml/v2/forensic_analysis_results.json", "r") as f:
    data = json.load(f)

with open("backend/ml/v2/forensic_report_raw.txt", "w") as out:
    out.write("TOP 50 FALSE POSITIVES (REAL flagged as FAKE):\n")
    for i, x in enumerate(data["top_50_false_positives"][:50]):
        out.write(f"{i+1:2d}. Filename: {x['filename']:<45} | Prob FAKE: {x['prob_fake']:.4f} | Source: {x['source']}\n")

    out.write("\nTOP 50 FALSE NEGATIVES (FAKE flagged as REAL):\n")
    for i, x in enumerate(data["top_50_false_negatives"][:50]):
        out.write(f"{i+1:2d}. Filename: {x['filename']:<45} | Prob FAKE: {x['prob_fake']:.4f} | Source: {x['source']}\n")

print("Done! Formatted report saved to backend/ml/v2/forensic_report_raw.txt")
