import os
import yaml
import gc
import torch
from ultralytics import YOLO

# ——————————————————————————————————————————————
# Fungsi TRAINING YOLOv8 dengan augmentasi dan HSV
# ——————————————————————————————————————————————
def train_yolo():
    yaml_file = "./ingredients_detection/data.yaml"
    if not os.path.exists(yaml_file):
        print(f"Error: File YAML {yaml_file} tidak ditemukan!")
        return

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    train_dir, val_dir = data["train"], data["val"]

    # Cek CUDA
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️ Menggunakan device: {device}")

    print("\n🚀 Mulai training YOLOv8…")
    model = YOLO("yolov8s.pt")

    model.train(
        data=os.path.abspath(yaml_file),
        epochs=50,
        imgsz=512,
        batch=8,
        device=device,
        patience=10,

        # — Augmentasi bawaan: mixup & mosaic
        augment=True,

        # — HSV augmentation
        hsv_h=0.015,  # hue shift
        hsv_s=0.7,    # saturation shift
        hsv_v=0.4,    # value shift

        # opsi lain
        name="ingredients_detection",
        half=(device != "cpu"),
        cache=False,
        verbose=False,
        save=True,
        exist_ok=True,
        workers=16,
    )

    # Validasi & metrik
    metrics = model.val()
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50   : {metrics.box.map50:.4f}")
    print(f"mAP@75   : {metrics.box.map75:.4f}")

    # Eksport ONNX
    export_path = model.export(format="onnx")
    print(f"✅ Model diekspor ke {export_path}")
    print("✅ Training selesai! Weight terbaik berada di runs/detect/ingredients_detection/weights/best.pt")


# ——————————————————————————————————————————————
# Fungsi INFERENSI dengan threshold confidence
# ——————————————————————————————————————————————
from typing import List, Set
from PIL import Image

def detect_objects(img_path: str, model: YOLO, conf: float = 0.6):
    """
    Jalankan deteksi, hanya hasil dengan confidence >= conf yang diambil.
    """
    res = model.predict(
        source=img_path,
        device="cpu",
        save=False,
        conf=conf,
        verbose=False
    )
    foods: List[str] = []
    for r in res:
        if r.boxes is not None:
            ids = r.boxes.cls.cpu().numpy().astype(int)
            foods.extend(model.names.get(i, f"class_{i}") for i in ids)
    return sorted(set(foods)), res[0].plot()


if __name__ == "__main__":
    # Bersihkan cache GPU
    torch.cuda.empty_cache()
    gc.collect()

    # Training
    train_yolo()

    # Contoh inferensi setelah training:
    # model = YOLO("runs/detect/ingredients_detection/weights/best.pt")
    # foods, viz = detect_objects("contoh_gambar.jpg", model)
    # print("Terdeteksi:", foods)
