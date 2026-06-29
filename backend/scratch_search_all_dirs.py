import os
import win32com.client # In case pythoncom/pywin32 is installed, else we can read binary or search manually

def get_shortcut_target(shortcut_path):
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        return shortcut.Targetpath
    except Exception as e:
        print("WScript.Shell failed, trying binary read of lnk:", e)
        try:
            with open(shortcut_path, 'rb') as f:
                content = f.read()
                # A very simple parser for lnk to extract path: look for ':\'
                idx = content.find(b':\\')
                if idx != -1:
                    start = idx - 1
                    end = content.find(b'\x00', start)
                    path_str = content[start:end].decode('utf-16le', errors='ignore')
                    # Let's clean up printable chars
                    clean_path = []
                    for c in content[start:start+200]:
                        if 32 <= c < 127:
                            clean_path.append(chr(c))
                        elif c == 0:
                            break
                    target = "".join(clean_path)
                    print("Extracted path from lnk:", target)
                    return target
        except Exception as ex:
            print("Binary read failed:", ex)
    return None

print("Starting search for 1000111612.jpg...")

# Check the shortcut target
shortcut = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset - Shortcut.lnk"
if os.path.exists(shortcut):
    target = get_shortcut_target(shortcut)
    if target and os.path.exists(target):
        print(f"Shortcut target: {target}")
        for root, dirs, files in os.walk(target):
            for file in files:
                if '1000111612' in file:
                    print(f"FOUND IN SHORTCUT TARGET: {os.path.join(root, file)}")

# Search OneDrive Desktop
desktop = r"c:\Users\riya2\OneDrive\Desktop"
for root, dirs, files in os.walk(desktop):
    for file in files:
        if '1000111612' in file:
            print(f"FOUND IN DESKTOP: {os.path.join(root, file)}")

# Search App Data
app_data = r"C:\Users\riya2\.gemini\antigravity-ide"
for root, dirs, files in os.walk(app_data):
    for file in files:
        if '1000111612' in file:
            print(f"FOUND IN APP DATA: {os.path.join(root, file)}")

print("Search completed.")
