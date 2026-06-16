import os

LABELS_DIR = "./dataset/train/labels"
counts = {0: 0, 1: 0, 2: 0}

for file in os.listdir(LABELS_DIR):
    if file.endswith('.txt'):
        with open(os.path.join(LABELS_DIR, file), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in counts:
                        counts[cls_id] += 1

print(f"📊 Dataset Class Count:\n🔴 Red Lights (0): {counts[0]}\n🟡 Yellow Lights (1): {counts[1]}\n🟢 Green Lights (2): {counts[2]}")