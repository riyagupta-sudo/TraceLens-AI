import numpy as np
import json

data = {
  "birds.jpg": {
    "Manipulation Risk": 25,
    "Screenshot Probability": 30,
    "AI Generation Probability": 15,
    "Metadata Trust Score": 85,
    "Stego Suspicion": 30,
    "Investigation Confidence": 69
  },
  "spider.jpg": {
    "Manipulation Risk": 50,
    "Screenshot Probability": 30,
    "AI Generation Probability": 15,
    "Metadata Trust Score": 85,
    "Stego Suspicion": 90,
    "Investigation Confidence": 66
  },
  "random.jpg": {
    "Manipulation Risk": 50,
    "Screenshot Probability": 30,
    "AI Generation Probability": 15,
    "Metadata Trust Score": 85,
    "Stego Suspicion": 90,
    "Investigation Confidence": 66
  },
  "Screenshot 2026-06-09.png": {
    "Manipulation Risk": 25,
    "Screenshot Probability": 75,
    "AI Generation Probability": 15,
    "Metadata Trust Score": 85,
    "Stego Suspicion": 0,
    "Investigation Confidence": 76
  }
}

metrics = [
    "Manipulation Risk",
    "Screenshot Probability",
    "AI Generation Probability",
    "Metadata Trust Score",
    "Stego Suspicion",
    "Investigation Confidence"
]

results = {}
for m in metrics:
    values = [data[img][m] for img in data]
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=0)
    var_val = np.var(values, ddof=0)
    
    results[m] = {
        "values": values,
        "mean": float(round(mean_val, 2)),
        "std_dev": float(round(std_val, 2)),
        "variance": float(round(var_val, 2)),
        "std_dev_under_10pct": bool(std_val < 10.0),
        "coef_var": float(round(std_val / mean_val, 4)) if mean_val > 0 else 0
    }

print(json.dumps(results, indent=2))
