
![alt RGDORA](https://github.com/NahinAlam001/meme/blob/main/RGDORA.png?raw=true)
---
# 🌏 RGDORA: Retrieval-Guided Domain Robust Architecture
### Bengali Hateful Meme Detection System 🇧🇩

> **RGDORA** is a domain-robust multimodal architecture for **Bengali hateful meme detection**, integrating retrieval-guided contrastive learning, class imbalance handling, and rationale generation.

---

## 🔥 Features

- ✅ Full **checkpoint resumability**
- ✅ **Early stopping** with validation monitoring
- ✅ Advanced **class imbalance handling** (Focal Loss, Class-Balanced Loss, Balanced Sampling)
- ✅ **Mixed precision** training (AMP)
- ✅ **FAISS-based retrieval-guided** contrastive learning
- ✅ **Rationale generation** via BART
- ✅ Comprehensive **metrics & evaluation** suite

---

## 📁 Project Structure
```
.
├── config.py              # Configuration management
├── dataset.py             # Dataset loader for BHM
├── models.py              # RGDORA and NegativeGenerator
├── trainer.py             # Main training system
├── metrics.py             # Metrics tracking
├── early_stopping.py      # Early stopping handler
├── checkpoint_manager.py  # Checkpoint management
├── faiss_manager.py       # FAISS index management
├── imbalance_handler.py   # Loss functions and samplers
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── setup.sh               # Installation script
└── README.md              # This file

```
---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
chmod +x setup.sh
./setup.sh
```

Installs all dependencies and downloads the **Bengali Hateful Meme (BHM)** dataset automatically.

---

### 2️⃣ Train Models

Train **Task 1 (Binary: Hate vs Non-Hate)**

```bash
python main.py --task task1
```

Train **Task 2 (Multi-class: TI, TC, TO, TS)**

```bash
python main.py --task task2
```

Train **both tasks sequentially**

```bash
python main.py --task both
```

---

### 3️⃣ Resume Training

Automatically resumes from the latest checkpoint:

```bash
python main.py --task task1
```

---

## 📊 Dataset: Bengali Hateful Meme (BHM)

### **Task 1: Binary Classification**

| Class     | Label | Count     | Ratio    |
| :-------- | :---- | :-------- | :------- |
| Hate      | 1     | 3,418     | 59.4%    |
| Non-Hate  | 0     | 2,340     | 40.6%    |
| **Total** | —     | **5,758** | 1.72 : 1 |

### **Task 2: Multi-class Classification**

| Class                    | Label | Count     | Ratio  |
| :----------------------- | :---- | :-------- | :----- |
| Target Individual (TI)   | 0     | 1,623     | 76.7%  |
| Target Community (TC)    | 1     | 249       | 11.8%  |
| Target Organization (TO) | 2     | 160       | 7.6%   |
| Target Society (TS)      | 3     | 85        | 4.0%   |
| **Total**                | —     | **2,117** | 19 : 1 |

---

## ⚙️ Configuration

You can edit `config.py` or override parameters via CLI.

```python
config = RGDORAConfig(
    task="task1",                # "task1" or "task2"
    batch_size=8,
    epochs=20,
    learning_rate=2e-5,
    loss_type="focal",           # "focal", "class_balanced", "weighted_ce"
    use_balanced_sampler=True,
    patience=5,                  # Early stopping patience
    monitor_metric="val_macro_f1"
)
```

---

## 🧩 Key Components

### 🎯 Class Imbalance Handling

* **Focal Loss:** Down-weights easy examples
* **Class-Balanced Loss:** Uses effective sample numbers
* **Balanced Batch Sampler:** Ensures equal representation
* **Weighted Random Sampler:** Oversamples minorities

### ⚖️ Training Stability

* Gradient clipping
* Mixed precision (AMP)
* NaN-safe batch skipping
* Warmup + linear LR scheduling

### 📈 Evaluation Metrics

* Accuracy & Balanced Accuracy
* Macro / Weighted F1, Precision, Recall
* Matthews Correlation Coefficient (MCC)
* Cohen’s Kappa
* ROC-AUC & PR-AUC
* Per-class metrics

---

## 🔬 Model Architecture

**RGDORA Components**

| Component             | Description                           |
| --------------------- | ------------------------------------- |
| 🖼️ Visual Encoder    | CLIP (ViT-B/32)                       |
| 🔤 Text Encoder       | XGLM-564M (Multilingual)              |
| ⚙️ Fusion             | Feature Interaction Matrix (FIM)      |
| 🧠 Classifier         | 2-layer MLP + LayerNorm               |
| 🧾 Rationale Decoder  | BART-base                             |
| 💣 Negative Generator | 3-layer MLP for adversarial negatives |

**Training Objectives:**

* Classification Loss (Focal or Class-Balanced)
* Retrieval-Guided Contrastive Loss (RGCL)
* Optional Rationale Generation Loss

---

## 🧠 Performance Tips

### 🟦 Task 1 (Binary)

* `loss_type="focal"` with `focal_alpha=0.25`
* `sampler_alpha=0.7` (moderate balancing)
* Monitor `val_macro_f1`

### 🟥 Task 2 (Multi-class)

* `loss_type="class_balanced"` with `cb_beta=0.9999`
* `sampler_alpha=0.9` (aggressive balancing)
* `patience=7`, `learning_rate=1.5e-5`

---

## 📂 Outputs

All logs, checkpoints, and metrics are automatically saved.

| Task   | Logs Directory | Checkpoints Directory | Best Model      |
| ------ | -------------- | --------------------- | --------------- |
| Task 1 | `logs_task1/`  | `checkpoints_task1/`  | `best_model.pt` |
| Task 2 | `logs_task2/`  | `checkpoints_task2/`  | `best_model.pt` |

Monitor training live:

```bash
tail -f logs_task1/training.log
```

---

## 🧾 Citation

If you use this code, please cite:

```bibtex
@misc{rgdora2025,
  title={RGDORA: Retrieval-Guided Domain Robust Architecture for Bengali Hateful Meme Detection},
  author={Md. Nahin Alam},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request.

---

## 📜 License

**MIT License** — see [`LICENSE`](./LICENSE) for details.

---

## 🧩 Usage Summary

```bash
# 1. Setup (one-time)
chmod +x setup.sh
./setup.sh

# 2. Train Task 1 (Binary)
python main.py --task task1

# 3. Train Task 2 (Multi-class)
python main.py --task task2

# 4. Train both tasks
python main.py --task both

# 5. Monitor training
tail -f logs_task1/training.log

# 6. Resume interrupted training (auto)
python main.py --task task1
```

---

> 💡 **Tip:** RGDORA is fully modular — plug in any dataset or encoder by extending `dataset.py` and `models.py`.

---

<p align="center">
  <b>Developed with ❤️ by <a href="https://github.com/nahinalam">Md. Nahin Alam</a></b><br>
  <i>For robust Bengali multimodal hate speech understanding.</i>
</p>
