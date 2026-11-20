import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader, Dataset
import psutil
import timm
from transformers import CvtForImageClassification, CvtConfig

# ============================================================
# CENTRAL CONFIGURATION (Sinkron dengan Skrip Latihan Murni)
# ============================================================
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_WORKERS = 2
WARMUP_BATCHES = 3

CLASS_TO_IDX = {"real": 0, "inpainting": 1, "text2image": 2}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================
# SINKRONISASI ARSITEKTUR MODEL BARU (DARI TRAININ2.PY)
# ============================================================

class StandaloneEfficientNetV2S(nn.Module):
    """Model EfficientNetV2-S Murni (~24 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x): 
        return self.model(x)

class StandaloneCoAtNet0(nn.Module):
    """Model CoAtNet Hybrid (~28 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model("coatnet_nano_rw_224", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x): 
        return self.model(x)

class StandaloneSwinTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.model(x)

class StandaloneConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    def forward(self, x): return self.model(x)

class StandaloneMaxViTTiny(nn.Module):
    """Model MaxViT Tiny Hybrid (~31 Juta Parameter)"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model("maxvit_tiny_tf_224", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class StandaloneCvT21(nn.Module):
    """Model CvT-21 Hybrid (~32 Juta Parameter) via HuggingFace"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
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

# ============================================================
# DATA TRANSFORMS & DATASET MAPPER (FIXED SINKRON VALIDASI)
# ============================================================
TARGET_SIZE = 224

def get_val_transforms(target_size=TARGET_SIZE):
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

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
        if not self.samples:
            raise RuntimeError(f"Tidak ada gambar ditemukan di {self.root_dir}")

    def __len__(self): 
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as image: 
            img = image.convert("RGB")
        if self.transform is not None: 
            img = self.transform(img)
        return img, label

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# PERFORMANCE HARDWARE PROFILER
# ============================================================
class RAMProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.reset()
    def reset(self):
        import gc
        gc.collect()
        self.baseline_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = self.baseline_ram_mb
    def update(self):
        current_ram = self.process.memory_info().rss / (1024 * 1024)
        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram
    def get_summary(self):
        return self.baseline_ram_mb, self.peak_ram_mb, max(0.0, self.peak_ram_mb - self.baseline_ram_mb)

def estimate_flops(model, input_size):
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        x = torch.randn(input_size).to(device)
        return FlopCountAnalysis(model, x).total() / 1e9
    except Exception:
        return (sum(p.numel() for p in model.parameters() if p.requires_grad) * 2) / 1e9

def show_best_epoch_from_history(results_dir, model_name):
    history_file = results_dir / f"standalone_history_{model_name}.csv"
    if history_file.exists():
        try:
            df = pd.read_csv(history_file)
            if not df.empty:
                best_row = df.sort_values(by=["val_acc", "val_loss"], ascending=[False, True]).iloc[0]
                print(f"   [*] Info Log Histori:")
                print(f"       - Terakhir di-save pada Epoch : Ke-{int(best_row['epoch'])}")
                print(f"       - Akurasi Validasi Tertinggi  : {best_row['val_acc']*100:.2f}%")
                return int(best_row['epoch'])
        except Exception as e:
            print(f"   [!] Gagal membaca histori training log: {str(e)}")
    return "N/A"

# ============================================================
# EVALUATOR ENGINE
# ============================================================
def evaluate_model(model_name, model, test_loader, device, results_dir, target_size):
    print(f"\n" + "=" * 75)
    print(f"EVALUASI METRIK KINERJA & PROFILING FISIK (BARU): {model_name.upper()}")
    print("=" * 75)
    
    best_epoch = show_best_epoch_from_history(results_dir, model_name)
    print("-" * 75)
    
    model = model.to(device)
    model.eval()
    
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            if i >= WARMUP_BATCHES: break
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _ = model(imgs)
            
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ram_profiler = RAMProfiler()
    ram_profiler.reset()
    
    all_targets, all_probs = [], []
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            ram_profiler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time_sec = time.perf_counter() - start_time
    baseline_ram, peak_ram, peak_ram_delta = ram_profiler.get_summary()

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        all_probs = np.array(all_probs, dtype=np.float64)
        all_probs = all_probs / np.sum(all_probs, axis=1, keepdims=True)
        
        all_preds_3cls = np.argmax(all_probs, axis=1)
        total_images = len(all_targets)
    else:
        print("[!] Dataset evaluasi kosong.")
        return None, None

    latency_ms = (total_time_sec / max(1, total_images)) * 1000.0
    throughput = total_images / max(1e-6, total_time_sec)

    # === PERHITUNGAN METRIK MULTIKELAS (3-CLASSES) ===
    acc_3cls = accuracy_score(all_targets, all_preds_3cls)
    prec_3cls = precision_score(all_targets, all_preds_3cls, average="macro", zero_division=0)
    rec_3cls = recall_score(all_targets, all_preds_3cls, average="macro", zero_division=0)
    f1_3cls = f1_score(all_targets, all_preds_3cls, average="macro", zero_division=0)
    
    try:
        targets_one_hot = np.eye(NUM_CLASSES)[all_targets]
        auc_3cls = roc_auc_score(targets_one_hot, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc_3cls = 0.0
    cm_3cls = confusion_matrix(all_targets, all_preds_3cls).tolist()

    # === PERHITUNGAN METRIK BINER (REAL VS AI) ===
    binary_targets = np.where(all_targets == 0, 0, 1)
    binary_probs_ai = all_probs[:, 1] + all_probs[:, 2]
    binary_preds = np.where(binary_probs_ai >= 0.5, 1, 0)

    acc_bin = accuracy_score(binary_targets, binary_preds)
    prec_bin = precision_score(binary_targets, binary_preds, zero_division=0)
    rec_bin = recall_score(binary_targets, binary_preds, zero_division=0)
    f1_bin = f1_score(binary_targets, binary_preds, zero_division=0)
    try:
        auc_bin = roc_auc_score(binary_targets, binary_probs_ai)
    except Exception:
        auc_bin = 0.0
    cm_bin = confusion_matrix(binary_targets, binary_preds).tolist()

    # === HITUNG PARAMETER & FLOPS ===
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = estimate_flops(model, input_size=(1, 3, target_size, target_size))

    # === CETAK HASIL EVALUASI KE KONSOL ===
    print(f"   [+] Multikelas (3-Cls) -> Akurasi: {acc_3cls*100:.2f}% | F1-Macro: {f1_3cls*100:.2f}% | AUC: {auc_3cls:.4f}")
    print(f"   [+] Klasifikasi Biner  -> Akurasi: {acc_bin*100:.2f}% | F1-Biner: {f1_bin*100:.2f}% | AUC: {auc_bin:.4f}")
    print(f"   [+] Latensi & Speed    -> Latensi: {latency_ms:.2f} ms/img | Throughput: {throughput:.2f} img/s")
    print(f"   [+] Konsumsi Hardware  -> Peak VRAM: {peak_vram_mb:.2f} MB | Delta RAM: {peak_ram_delta:.2f} MB")

    # === PENYUSUNAN RINGKASAN DATA ROW ===
    eval_summary_row = {
        "model": model_name,
        "condition": "test_in_domain",
        "best_training_epoch": best_epoch,
        "total_images": total_images,
        "acc_3class": round(acc_3cls, 4),
        "prec_3class": round(prec_3cls, 4),
        "rec_3class": round(rec_3cls, 4),
        "f1_3class": round(f1_3cls, 4),
        "roc_auc_3class": round(float(auc_3cls), 4),
        "confusion_matrix_3class": str(cm_3cls),
        "acc_binary": round(acc_bin, 4),
        "prec_binary": round(prec_bin, 4),
        "rec_binary": round(rec_bin, 4),
        "f1_binary": round(f1_bin, 4),
        "roc_auc_binary": round(auc_bin, 4),
        "confusion_matrix_binary": str(cm_bin),
    }

    efficiency_summary_row = {
        "model": model_name,
        "condition": "test_in_domain",
        "total_parameters": total_params,
        "gflops": round(gflops, 3),
        "total_inference_time_sec": round(total_time_sec, 4),
        "latency_ms_per_image": round(latency_ms, 2),
        "throughput_images_per_sec": round(throughput, 2),
        "peak_gpu_vram_mb": round(peak_vram_mb, 2),
        "baseline_ram_mb": round(baseline_ram, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "peak_ram_delta_mb": round(peak_ram_delta, 2)
    }

    return eval_summary_row, efficiency_summary_row

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Pipeline Evaluasi 6 Model Baru Aktif Pada: {device}")

    current_file = Path(__file__).resolve()
    ROOT_DIR = current_file.parent.parent if current_file.parent.name == "src" else current_file.parent
    splits_dir = ROOT_DIR / "dataset" / "splits"
    models_dir = ROOT_DIR / "models"
    results_dir = ROOT_DIR / "results"

    test_path = splits_dir / "test"
    if not test_path.exists():
        test_path = splits_dir / "validation"
        print(f"[!] Warning: Folder 'test' tidak ditemukan. Menggunakan: {test_path}")

    # SINKRONISASI CHECKPOINT MODEL DARI KODE TRAININ2.PY (SUDAH FIXED 6 MODEL)
    models_config = {
        "tf_efficientnetv2_s": (lambda: StandaloneEfficientNetV2S(pretrained=False), "standalone_student_tf_efficientnetv2_s.pth", 224),
        "coatnet_nano_rw_224": (lambda: StandaloneCoAtNet0(pretrained=False), "standalone_student_coatnet_nano_rw_224.pth", 224),
        "swin_tiny": (lambda: StandaloneSwinTiny(pretrained=False), "standalone_student_swin_tiny.pth", 224),
        "convnext_tiny": (lambda: StandaloneConvNeXtTiny(pretrained=False), "standalone_student_convnext_tiny.pth", 224),
        "maxvit_tiny": (lambda: StandaloneMaxViTTiny(pretrained=False), "standalone_student_maxvit_tiny.pth", 224),
        "cvt_21": (lambda: StandaloneCvT21(pretrained=False), "standalone_student_cvt_21.pth", 224)
    }

    eval_summaries, efficiency_summaries = [], []

    for model_name, (model_factory, file_name, target_size) in models_config.items():
        ckpt_path = models_dir / file_name
        
        if ckpt_path.exists():
            print(f"\n[*] Berhasil Menemukan Checkpoint Baru: {file_name}")
            
            try:
                test_dataset = MappedImageDataset(test_path, transform=get_val_transforms(target_size=target_size))
                test_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
                )
            except Exception as e:
                print(f"   [!] Gagal memuat dataset untuk {model_name}: {str(e)}")
                continue

            # Load Bobot Model
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            
            model = model_factory()
            model.load_state_dict(state_dict)
                
            # Jalankan Evaluasi Utama
            eval_row, eff_row = evaluate_model(model_name, model, test_loader, device, results_dir, target_size)
            if eval_row is not None:
                eval_summaries.append(eval_row)
                efficiency_summaries.append(eff_row)
            
            # Bersihkan Memori Sekuensial VRAM
            del model, test_dataset, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
        else:
            print(f"[-] Checkpoint murni '{file_name}' absen di /models/, skipping {model_name}.")

    # Menyimpan Output CSV Kolektif Baru
    if eval_summaries:
        results_dir.mkdir(parents=True, exist_ok=True)
        eval_out_path = results_dir / "evaluation_summary_model_baru.csv"
        eff_out_path = results_dir / "efficiency_summary_model_baru.csv"
        
        pd.DataFrame(eval_summaries).to_csv(eval_out_path, index=False)
        pd.DataFrame(efficiency_summaries).to_csv(eff_out_path, index=False)
        
        print("\n" + "=" * 75)
        print("[✓✓✓] EVALUASI KOMPARATIF MODEL BARU SELESAI! File Tersimpan Pada:")
        print(f"      1. {eval_out_path}")
        print(f"      2. {eff_out_path}")
        print("=" * 75)
    else:
        print("\n[!] Gagal evaluasi. Pastikan berkas .pth hasil latihan baru sudah ada di folder /models/.")

if __name__ == "__main__":
    main()

