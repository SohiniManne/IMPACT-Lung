import os

# Define the root root
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

print(f"🕵️  Sherlock Holmes Path Debugger")
print(f"-------------------------------")
print(f"Working Directory: {BASE_DIR}")

# 1. Check where the PNGs are hiding
locations_to_check = [
    os.path.join(DATA_DIR, 'imaging'),
    os.path.join(DATA_DIR, 'imaging', 'images'),
]

found_path = None

for path in locations_to_check:
    if os.path.exists(path):
        # Count PNGs in this folder
        files = [f for f in os.listdir(path) if f.endswith('.png')]
        count = len(files)
        print(f"Checking: {path} -> Found {count} PNGs")
        if count > 0:
            found_path = path
    else:
        print(f"Checking: {path} -> ❌ Folder does not exist")

print(f"-------------------------------")

if found_path:
    print(f"✅ FOUND THEM! Your images are located at:")
    print(f"   {found_path}")
    print(f"\n👇 ACTION: Open 'src/config.py' and ensure IMG_DIR matches this structure.")
    
    if found_path.endswith('images'):
        print("   Set: IMG_DIR = os.path.join(DATA_DIR, 'imaging', 'images')")
    else:
        print("   Set: IMG_DIR = os.path.join(DATA_DIR, 'imaging')")
else:
    print("❌ CRITICAL: Could not find any .png files in /data/imaging or /data/imaging/images.")
    print("   Did you move them?")