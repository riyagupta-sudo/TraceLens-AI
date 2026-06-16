with open('app/schemas.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'class Media' in line or 'class OSINT' in line:
        print(f"Line {i+1}: {line.strip()}")
