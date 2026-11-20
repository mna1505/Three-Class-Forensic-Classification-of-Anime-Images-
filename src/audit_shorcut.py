import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Cari folder utama projek secara otomatis (naik 1 folder dari folder 'src')
BASE_DIR = Path(__file__).resolve().parent.parent

def scan(split):
    rows = []
    # Gabungkan jalur dasar dengan folder split target
    search_path = BASE_DIR / "dataset" / "splits" / split
    
    for cls in ["real", "inpainting", "text2image"]:
        folder = search_path / cls
        if not folder.exists():
            continue
            
        for p in folder.rglob("*"):
            if p.suffix.lower() in EXT:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                    rows.append(dict(cls=cls, ext=p.suffix.lower(), w=w, h=h,
                                     ratio=w / h, kb=p.stat().st_size / 1024,
                                     bpp=p.stat().st_size * 8 / (w * h)))
                except Exception:
                    # Lewati jika ada file gambar yang rusak/corrupt
                    continue
                    
    df = pd.DataFrame(rows)
    
    # Pengaman: Jika dataframe kosong, beri tahu foldernya di mana agar mudah dicek
    if df.empty:
        raise ValueError(f"Tidak ditemukan gambar di jalur: {search_path}. Pastikan dataset sudah diletakkan di folder tersebut.")
        
    return df

# Proses scanning
tr, va = scan("train"), scan("validation")

feats = ["w", "h", "ratio", "kb", "bpp"]
Xtr = pd.concat([tr[feats], pd.get_dummies(tr["ext"]).astype(int)], axis=1)
Xva = pd.concat([va[feats], pd.get_dummies(va["ext"]).astype(int)], axis=1).reindex(columns=Xtr.columns, fill_value=0)

clf = DecisionTreeClassifier(max_depth=6).fit(Xtr, tr["cls"])
print("Akurasi hanya dari metadata:", clf.score(Xva, va["cls"]))
print(tr.groupby("cls")[["w", "h", "ratio", "kb"]].describe().T)
