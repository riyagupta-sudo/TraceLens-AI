import re
import os

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
app_dir = os.path.join(project_root, "backend", "app")

def inspect_file(filename):
    filepath = os.path.join(app_dir, filename)
    if not os.path.exists(filepath):
        print(f"{filename} does not exist.")
        return
        
    print(f"\n==========================================")
    print(f"File: {filename}")
    print(f"==========================================")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find all function definitions and lines they start on
    lines = content.split("\n")
    func_regex = re.compile(r"^\s*def\s+(\w+)\s*\(")
    class_regex = re.compile(r"^\s*class\s+(\w+)")
    
    for idx, line in enumerate(lines):
        fm = func_regex.match(line)
        cm = class_regex.match(line)
        if fm:
            print(f"Line {idx+1:4d}: def {fm.group(1)}")
        elif cm:
            print(f"Line {idx+1:4d}: class {cm.group(1)}")

def main():
    inspect_file("dna_engine.py")
    inspect_file("similarity_engine.py")
    inspect_file("report_generator.py")
    inspect_file("video_analyzer.py")

if __name__ == "__main__":
    main()
