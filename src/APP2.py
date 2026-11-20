import io
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

# ============================================================
# 1. KONFIGURASI MODEL (SINKRON DENGAN TRAINING & EVALUASI)
# ============================================================
NUM_CLASSES = 3
TARGET_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IDX_TO_CLASS = {
    0: "real",
    1: "inpainting",
    2: "text2image"
}

LABEL_DESCRIPTIONS = {
    "real": "Gambar Asli (Kamera / Non-AI)",
    "inpainting": "Editan AI (Inpainting / Manipulasi Sebagian)",
    "text2image": "Buatan AI Sepenuhnya (Text-to-Image)"
}

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "src" else CURRENT_FILE.parent
MODEL_PATH = BASE_DIR / "models" / "standalone_student_swin_tiny.pth"

# ============================================================
# 2. KODE FRONTEND (HTML + CSS + JS) DIGABUNG DI SINI
# ============================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detektor Gambar AI — Swin Tiny</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body {
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            max-width: 680px;
            width: 100%;
            padding: 32px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        }
        .header { text-align: center; margin-bottom: 24px; }
        .header h1 { font-size: 1.6rem; margin-bottom: 6px; }
        .header p { color: var(--muted); font-size: 0.9rem; }
        
        .upload-area {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(15, 23, 42, 0.4);
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.08);
        }
        .upload-area p { color: var(--muted); margin-top: 8px; font-size: 0.85rem; }
        input[type="file"] { display: none; }

        .preview-box {
            display: none;
            margin-top: 20px;
            text-align: center;
        }
        .preview-box img {
            max-height: 260px;
            max-width: 100%;
            border-radius: 10px;
            border: 1px solid var(--border);
            object-fit: contain;
        }
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 16px;
            transition: background 0.2s;
        }
        .btn:hover { background: var(--primary-hover); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }

        .result-card {
            display: none;
            margin-top: 24px;
            padding: 20px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
        }
        .verdict-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 8px;
        }
        .verdict-real { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
        .verdict-ai { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
        
        .sub-desc { color: var(--muted); font-size: 0.9rem; margin-bottom: 18px; }
        
        .bar-group { margin-bottom: 12px; }
        .bar-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
        .bar-bg { background: #334155; height: 10px; border-radius: 6px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Detektor Keaslian Gambar AI</h1>
        <p>Didukung Arsitektur <strong>Swin-Tiny</strong> (Akurasi Uji: 97,67%)</p>
    </div>

    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <strong>Klik untuk memilih gambar</strong> atau seret file ke area ini
        <p>Mendukung format JPG, PNG, WEBP, BMP</p>
        <input type="file" id="fileInput" accept="image/*">
    </div>

    <div class="preview-box" id="previewBox">
        <img id="imagePreview" src="" alt="Pratinjau Gambar">
        <button class="btn" id="analyzeBtn" onclick="analyzeImage()">Analisis Gambar Sekarang</button>
    </div>

    <div class="result-card" id="resultCard">
        <div id="verdictBadge" class="verdict-badge"></div>
        <div id="subDesc" class="sub-desc"></div>

        <div class="bar-group">
            <div class="bar-label"><span>Real (Gambar Asli)</span><span id="valReal">0%</span></div>
            <div class="bar-bg"><div class="bar-fill" id="barReal" style="width: 0%; background: var(--success);"></div></div>
        </div>
        <div class="bar-group">
            <div class="bar-label"><span>Inpainting (Editan AI Sebagian)</span><span id="valInpaint">0%</span></div>
            <div class="bar-bg"><div class="bar-fill" id="barInpaint" style="width: 0%; background: var(--warning);"></div></div>
        </div>
        <div class="bar-group">
            <div class="bar-label"><span>Text-to-Image (AI Sepenuhnya)</span><span id="valT2I">0%</span></div>
            <div class="bar-bg"><div class="bar-fill" id="barT2I" style="width: 0%; background: var(--danger);"></div></div>
        </div>
    </div>
</div>

<script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewBox = document.getElementById('previewBox');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultCard = document.getElementById('resultCard');
    let selectedFile = null;

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Harap pilih file gambar yang valid.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewBox.style.display = 'block';
            resultCard.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async function analyzeImage() {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerText = 'Sedang Menganalisis...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) throw new Error(data.detail || 'Terjadi kesalahan pada server.');

            const badge = document.getElementById('verdictBadge');
            badge.innerText = `${data.binary_result.verdict} (${data.binary_result.is_ai ? data.binary_result.ai_confidence : data.binary_result.real_confidence}%)`;
            badge.className = 'verdict-badge ' + (data.binary_result.is_ai ? 'verdict-ai' : 'verdict-real');

            document.getElementById('subDesc').innerText = `Kategori Spesifik: ${data.multiclass_result.description}`;

            const probs = data.multiclass_result.probabilities;
            updateBar('Real', probs.real);
            updateBar('Inpaint', probs.inpainting);
            updateBar('T2I', probs.text2image);

            resultCard.style.display = 'block';
        } catch (err) {
            alert(err.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = 'Analisis Gambar Sekarang';
        }
    }

    function updateBar(idSuffix, value) {
        document.getElementById('val' + idSuffix).innerText = value + '%';
        document.getElementById('bar' + idSuffix).style.width = value + '%';
    }
</script>

</body>
</html>
"""

# ============================================================
# 3. DEFINISI ARSITEKTUR SWIN TINY & TRANSFORMASI
# ============================================================
class StandaloneSwinTiny(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224", 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

def get_val_transforms(target_size=TARGET_SIZE):
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ============================================================
# 4. INISIALISASI SERVER & PEMUATAN MODEL
# ============================================================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Memuat model Swin Tiny pada perangkat: {device}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}")

    model = StandaloneSwinTiny(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    ml_models["swin_tiny"] = model
    ml_models["device"] = device
    ml_models["transform"] = get_val_transforms()
    print("[✓] Model Swin Tiny siap menerima permintaan!")
    
    yield
    ml_models.clear()

app = FastAPI(title="AI Image Detector - Swin Tiny", lifespan=lifespan)

# ============================================================
# 5. ENDPOINT HALAMAN UTAMA & PREDIKSI API
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Gambar rusak atau format tidak didukung.")

    device = ml_models["device"]
    model = ml_models["swin_tiny"]
    transform = ml_models["transform"]

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # Hasil 3-Kelas
    pred_idx = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]
    
    # Hasil Biner (Real vs AI Gabungan)
    prob_real = float(probs[0])
    prob_inpainting = float(probs[1])
    prob_t2i = float(probs[2])
    prob_ai_total = prob_inpainting + prob_t2i
    is_ai = prob_ai_total >= 0.5

    return {
        "status": "success",
        "filename": file.filename,
        "binary_result": {
            "verdict": "TERDETEKSI AI" if is_ai else "GAMBAR ASLI (REAL)",
            "is_ai": is_ai,
            "ai_confidence": round(prob_ai_total * 100, 2),
            "real_confidence": round(prob_real * 100, 2)
        },
        "multiclass_result": {
            "predicted_class": pred_class,
            "description": LABEL_DESCRIPTIONS[pred_class],
            "probabilities": {
                "real": round(prob_real * 100, 2),
                "inpainting": round(prob_inpainting * 100, 2),
                "text2image": round(prob_t2i * 100, 2)
            }
        }
    }

# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
