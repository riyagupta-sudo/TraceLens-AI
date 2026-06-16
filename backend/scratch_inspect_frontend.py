import re

with open('../frontend/src/app/media/[id]/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

keywords = ["investigation_summary", "risk_score", "integrity_score", "stego", "ai_res", "ai_detection", "probability"]
for kw in keywords:
    matches = [m.start() for m in re.finditer(kw, content)]
    print(f"Keyword '{kw}': {len(matches)} matches")
    for idx in matches[:2]:
        start = max(0, idx - 100)
        end = min(len(content), idx + 150)
        print(f"  Snippet: {content[start:end]}\n")
