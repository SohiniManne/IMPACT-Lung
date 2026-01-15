import os

BASE_DIR = os.getcwd()
CT_IMG_DIR = os.path.join(BASE_DIR, 'data', 'ct_scans', 'images')
CT_MASK_DIR = os.path.join(BASE_DIR, 'data', 'ct_scans', 'masks')

print(f"🕵️  CT SCAN DEBUGGER")
print(f"--------------------")
print(f"Checking Images at: {CT_IMG_DIR}")
print(f"Checking Masks at:  {CT_MASK_DIR}")

if not os.path.exists(CT_IMG_DIR):
    print("❌ ERROR: Image folder does not exist!")
else:
    images = sorted(os.listdir(CT_IMG_DIR))
    print(f"✅ Found {len(images)} files in Images folder.")
    if len(images) > 0:
        print(f"   First file: '{images[0]}'")

if not os.path.exists(CT_MASK_DIR):
    print("❌ ERROR: Mask folder does not exist!")
else:
    masks = sorted(os.listdir(CT_MASK_DIR))
    print(f"✅ Found {len(masks)} files in Masks folder.")
    if len(masks) > 0:
        print(f"   First file: '{masks[0]}'")

print(f"--------------------")
# Check for matches
if os.path.exists(CT_IMG_DIR) and os.path.exists(CT_MASK_DIR):
    img_set = set(images)
    mask_set = set(masks)
    common = img_set.intersection(mask_set)
    print(f"🔍 MATCH REPORT: Found {len(common)} exact filename matches.")
    
    if len(common) == 0 and len(images) > 0:
        print("\n⚠️  POSSIBLE MISMATCH:")
        print(f"   Image Name: '{images[0]}'")
        if len(masks) > 0:
            print(f"   Mask Name:  '{masks[0]}'")
        print("   (Are the extensions different? Is one .png and the other .jpg?)")