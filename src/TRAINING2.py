import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.amp as amp
import torchvision.models as models
import torchvision.transforms as T  # Menggunakan v1 secara konsisten sesuai isi fungsi
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import timm
from transformers import CvtForImageClassification, CvtConfig


# ============================================================
# CONFIGURATION & TRANSPORMS
# ============================================================
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_WORKERS = 2
GLOBAL_LR = 5e-5
HEAD_LR = 1e-4
GLOBAL_WEIGHT_DECAY = 1e-3
STUDENT_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 6
WARMUP_EPOCHS = 3
COOLDOWN_SECONDS = 300

CLASS_TO_IDX = {"real": 0, "inpainting": 1, "text2image": 2}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SIZE = 224

class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        import io
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

def get_train_transforms(target_size=TARGET_SIZE):
    return T.Compose([
        # 1. PAKSA SEMUA GAMBAR (Besar maupun kecil) MENGALAMI PENYUSUTAN SKALA YANG ADIL
        # Kita perkecil dahulu sisi terpendek gambar menjadi 256px (Rasio asli tetap terjaga, tanpa padding hitam)
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        
        # 2. POTONG AREA 224x224 SECARA ACAK DARI GAMBAR YANG SUDAH SELEVEL
        # Ini menghancurkan bias padding hitam karena tidak ada canvas kosong di sekeliling gambar!
        T.RandomCrop(target_size, pad_if_needed=True, padding_mode="reflect"),
        
        # 3. TURUNKAN KUALITAS SECARA ACAK (Menghancurkan bias perbedaan ukuran KB file)
        RandomJPEGCompression(quality_range=(50, 95), p=0.4),
        
        # 4. AUGMENTASI GEOMETRIS ALAMI (Aman untuk artefak frekuensi tinggi)
        T.RandomHorizontalFlip(p=0.5),
        
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms(target_size=TARGET_SIZE):
    return T.Compose([
        # Proses validasi harus identik secara skala agar evaluasi bersifat objektif
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ============================================================
# ARCHITECTURES, DATASET & UTILS (Sama Seperti Kode Anda)
# ============================================================
class StandaloneEfficientNetV2S(nn.Module):
    """Model EfficientNetV2-S Murni (~24 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Diperbarui menjadi "tf_efficientnetv2_s" agar pretrained weights resmi dapat diunduh
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x): 
        return self.model(x)

class StandaloneCoAtNet0(nn.Module):
    """Model CoAtNet Hybrid (~28 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Diubah ke 'coatnet_nano_rw_224' yang registrasi unduhannya dijamin aman dan stabil di semua versi timm
        self.model = timm.create_model("coatnet_nano_rw_224", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x): 
        return self.model(x)

class StandaloneSwinTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.model(x)

class StandaloneConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.model(x)

class StandaloneMaxViTTiny(nn.Module):
    """Model MaxViT Tiny Hybrid (~31 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Menggunakan arsitektur maxvit_tiny_tf_224 dari pustaka timm
        self.model = timm.create_model("maxvit_tiny_tf_224", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class StandaloneCvT21(nn.Module):
    """Model CvT-21 Hybrid (~32 Juta Parameter) via HuggingFace"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Menggunakan CvtForImageClassification (v kecil)
        if pretrained:
            self.model = CvtForImageClassification.from_pretrained(
                "microsoft/cvt-21", 
                num_labels=num_classes, 
                ignore_mismatched_sizes=True
            )
        else:
            config = CvtConfig.from_pretrained("microsoft/cvt-21")
            config.num_labels = num_classes
            self.model = CvtForImageClassification(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits


class MappedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_folder = self.root_dir / cls_name
            if not cls_folder.exists(): continue
            for img_path in sorted(cls_folder.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTS:
                    self.samples.append((img_path, cls_idx))
        if not self.samples: raise RuntimeError(f"Tidak ada gambar ditemukan di {self.root_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as image: img = image.convert("RGB")
        if self.transform is not None: img = self.transform(img)
        return img, label

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_root_dir():
    current_file = Path(__file__).resolve()
    return current_file.parent.parent if current_file.parent.name == "src" else current_file.parent

def build_optimizer(model):
    head_params, backbone_params = [], []
    head_keywords = {'classifier', 'fc', 'head'}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        parts = name.split('.')
        if any(keyword in parts for keyword in head_keywords): head_params.append(param)
        else: backbone_params.append(param)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": GLOBAL_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=GLOBAL_WEIGHT_DECAY)

def trigger_cooldown():
    print(f"\n[💤 COOLDOWN] Mengistirahatkan perangkat selama {COOLDOWN_SECONDS} detik...")
    time.sleep(COOLDOWN_SECONDS)
    print("[⚡ READY] Melanjutkan eksperimen berikutnya!\n")

@torch.no_grad()
def evaluate_model(model, loader, device, criterion):
    model.eval()
    use_amp = device.type == "cuda"
    preds, targets, loss_sum = [], [], 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        preds.extend(out.argmax(1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, f1, loss_sum / max(1, len(loader.dataset))

# ============================================================
# STANDALONE TRAINING ENGINE (SUDAH DIPERBAIKI)
# ============================================================
def run_train_student_standalone(name, model_factory, train_loader, val_loader, device, root_dir):
    print("\n" + "="*70)
    print(f"RUNNING STANDALONE STUDENT TRAINING: {name.upper()}")
    print("="*70)
    
    models_dir = root_dir / "models"
    results_dir = root_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model = model_factory().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Parameter Aktif {name}: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
        ], milestones=[WARMUP_EPOCHS]
    )
    
    use_amp = device.type == "cuda"
    # PERBAIKAN: Memanggil GradScaler yang aman lintas versi PyTorch modern
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    best_acc, patience_counter, history = -1.0, 0, []
    ckpt_path = models_dir / f"standalone_student_{name}.pth"
    
    for epoch in range(1, STUDENT_EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, preds, targets = 0.0, [], []
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                loss = criterion(out, labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_sum += loss.item() * images.size(0)
            preds.extend(out.argmax(1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())
            
        train_loss = loss_sum / len(train_loader.dataset)
        train_acc = accuracy_score(targets, preds)
        val_acc, val_f1, val_loss = evaluate_model(model, val_loader, device, criterion)
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{STUDENT_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | F1: {val_f1:.4f} | {time.time()-t0:.0f}s")
        # Menyimpan history per epoch
        history.append({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "val_loss": val_loss, 
            "val_acc": val_acc, 
            "val_f1": val_f1
        })
        
        # Logika checkpoint untuk akurasi validasi terbaik
        if val_acc > best_acc:
            best_acc, patience_counter = val_acc, 0
            torch.save({"model_state_dict": model.state_dict(), "best_val_acc": best_acc}, ckpt_path)
            print(f"   [+] Checkpoint {name} disimpan: Val Acc {best_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"   [-] Early stopping {name} dipicu.")
                break
                
    # Penyimpanan hasil training ke file CSV
    csv_out_path = results_dir / f"standalone_history_{name}.csv"
    try:
        pd.DataFrame(history).to_csv(csv_out_path, index=False)
        print(f"[✓] Histori pelatihan berhasil disimpan di: {csv_out_path}")
    except PermissionError:
        backup_path = results_dir / f"standalone_history_{name}_backup_{int(time.time())}.csv"
        pd.DataFrame(history).to_csv(backup_path, index=False)
        print(f"[!] Warning: File utama sedang dibuka Excel. Histori disimpan di: {backup_path}")

    # Pembersihan memori model saat ini
    del model
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    return ckpt_path


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Pipeline Standalone Student Aktif Pada: {device}")
    
    root_dir = get_root_dir()
    splits_dir = root_dir / "dataset" / "splits"
    pin_mem = (device.type == "cuda")

    # Daftar eksperimen diperluas menjadi 6 model secara berurutan
    experiments = [
        # Ganti name menjadi tf_efficientnetv2_s
        {"name": "tf_efficientnetv2_s", "factory": lambda: StandaloneEfficientNetV2S(pretrained=True)},
        {"name": "coatnet_nano_rw_224", "factory": lambda: StandaloneCoAtNet0(pretrained=True)},
        {"name": "swin_tiny", "factory": lambda: StandaloneSwinTiny(pretrained=True)},
        {"name": "convnext_tiny", "factory": lambda: StandaloneConvNeXtTiny(pretrained=True)},
        {"name": "maxvit_tiny", "factory": lambda: StandaloneMaxViTTiny(pretrained=True)},
        {"name": "cvt_21", "factory": lambda: StandaloneCvT21(pretrained=True)}
    ]


    # Melakukan perulangan untuk mengeksekusi training tiap model secara berurutan
    for idx, exp in enumerate(experiments, 1):
        name = exp["name"]
        model_factory = exp["factory"]
        print(f"\n[PREPARE {idx}/{len(experiments)}] Mengonfigurasi Dataset & Loader untuk {name.upper()} (224x224)...")
        
        # Membuat instance dataset dan dataloader baru khusus untuk model yang aktif
        train_dataset = MappedImageDataset(splits_dir / "train", transform=get_train_transforms(target_size=224))
        val_dataset = MappedImageDataset(splits_dir / "validation", transform=get_val_transforms(target_size=224))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_mem)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)
        
        # Memulai fungsi training standalone
        run_train_student_standalone(
            name=name, 
            model_factory=model_factory, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device, 
            root_dir=root_dir
        )
        
        # Pembersihan RAM dan VRAM setelah training model selesai agar tidak menumpuk
        del train_dataset, val_dataset, train_loader, val_loader
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
            
        # Memberikan waktu jeda cooldown (istirahat GPU) sebelum berlanjut ke model berikutnya
        if idx < len(experiments): 
            trigger_cooldown()

    print("\n" + "="*70)
    print("[✓✓✓] PROSES LATIHAN SELURUH MODEL BARU (6 MODEL) SELESAI!")
    print("="*70)

if __name__ == "__main__":
    main()

