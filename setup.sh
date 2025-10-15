#!/bin/bash

echo "========================================="
echo "🚀 RGDORA Full Setup Script"
echo "========================================="

# --- PYTHON DEPENDENCIES INSTALLATION ---
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets evaluate
pip install pandas openpyxl pillow numpy scikit-learn tqdm sentencepiece protobuf rouge_score nltk bert_score

# --- SAFE FAISS INSTALLER (GPU → CPU fallback) ---
echo ""
echo "⚙️ Installing FAISS (GPU preferred, CPU fallback)..."

python - <<'PYCODE'
import subprocess, sys

def try_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        import faiss
        print(f"✅ Successfully installed {pkg}")
        return True
    except Exception as e:
        print(f"❌ Failed to install {pkg}: {e}")
        return False

try:
    import faiss
    print("✅ FAISS already installed")
except ImportError:
    print("📦 Installing FAISS...")
    for pkg in ["faiss-gpu-cu121", "faiss-gpu-cu118", "faiss-gpu"]:
        if try_install(pkg):
            break
    else:
        if try_install("faiss-cpu"):
            print("✅ Installed faiss-cpu (CPU fallback)")
        else:
            print("❌ Could not install FAISS at all. Please install manually.")
PYCODE

# --- DOWNLOAD DATASET ---
echo ""
if [ ! -d "BHM" ]; then
    echo "📥 Downloading BHM dataset..."
    pip install gdown
    gdown 'https://docs.google.com/uc?export=download&id=1uftdZXcf00X2MQVb8UUcRMaIor7Pfd1w' -O bhm_dataset.zip
    unzip bhm_dataset.zip >/dev/null 2>&1
    rm bhm_dataset.zip
    echo "✅ Dataset downloaded and extracted"
else
    echo "✅ BHM dataset already exists"
fi

# --- VERIFY DATASET STRUCTURE ---
echo ""
echo "📊 Verifying dataset structure..."
python - <<'PYCODE'
import os
required_paths = [
    'BHM/Files/train_task1.xlsx',
    'BHM/Files/valid_task1.xlsx',
    'BHM/Files/test_task1.xlsx',
    'BHM/Files/train_task2.xlsx',
    'BHM/Files/valid_task2.xlsx',
    'BHM/Files/test_task2.xlsx',
    'BHM/Memes'
]
all_exist = all(os.path.exists(p) for p in required_paths)
if all_exist:
    print("✅ All required files found!")
else:
    print("❌ Some required files are missing:")
    for p in required_paths:
        status = "✅" if os.path.exists(p) else "❌"
        print(f"  {status} {p}")
PYCODE

# --- DONE ---
echo ""
echo "========================================="
echo "✅ RGDORA Setup Complete!"
echo "========================================="
echo ""
echo "👉 To train:"
echo "  python main.py --task task1    # Binary classification"
echo "  python main.py --task task2    # Multi-class classification"
echo "  python main.py --task both     # Train both tasks"
echo ""
