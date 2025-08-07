import os
import math

root_folder = "./ingredients_detection"
num_classes = 54
subfolders = ['train', 'valid', 'test']

def is_valid_float(x):
    try:
        val = float(x)
        return not (math.isnan(val) or math.isinf(val))
    except:
        return False

def scan_numeric_errors():
    for sub in subfolders:
        label_dir = os.path.join(root_folder, sub, 'labels')
        if not os.path.exists(label_dir):
            continue

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue
            path = os.path.join(label_dir, file)
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = parts[0]
                    coords = parts[1:]
                    if not class_id.isdigit() or not all(is_valid_float(c) for c in coords):
                        print(f"❌ Format angka invalid di {path}, baris {i+1}: {line.strip()}")

if __name__ == "__main__":
    scan_numeric_errors()
