# PawPrint AI 🐾

**PawPrint AI** identifies animal species from footprint images and provides auxiliary traits such as estimated age group, wildness, and human risk level. The project includes a training script (EfficientNetV2-S), a Flask backend API that serves an interactive frontend, and utility scripts for data augmentation and preprocessing.

---

## ✅ Key Features

- Species classification from footprint images (multi-class)
- Age-group estimation using footprint area (Juvenile / Sub-Adult / Adult)
- Wildness & human-risk severity estimation per species
- Web UI for quick inference (upload an image → get result)
- Trainable pipeline using EfficientNetV2-S
- Lightweight augmentation scripts to balance datasets

---

## 📂 Project Structure

- `train_model.py` — Training script (EfficientNetV2-S, mixed precision enabled)
- `backend/` — Flask API + model artifacts
  - `app.py` — API server (serves frontend + `/predict` endpoint)
  - `model/` — Saved model & `class_indices.json`
  - `uploads/` — temporary upload folder
  - `utils/` — preprocessing, segmentation, age estimation, severity mapping
- `frontend/` — Static UI (`index.html`, `script.js`, `style.css`)
- `dataset/` — Classification dataset (train / val / test folders)
- `scripts/` — Data augmentation helpers
- `requirements.txt` — Python dependencies
- `steps_to_run.txt` — quick run notes

---

## ⚙️ Prerequisites

- Python 3.10+ recommended
- Virtual environment (venv or conda)
- macOS (Apple Silicon): consider `tensorflow-macos` + `tensorflow-metal` for best performance

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quickstart — Run the API + Frontend

1. Activate your venv and install dependencies (see above).
2. Start the backend server:

```bash
cd backend
python app.py
```

3. Open the frontend in your browser:

```
http://127.0.0.1:5000/
```

The UI lets you upload a footprint image and will display species, confidence, age group, wildness, and human risk.

---

## 🧪 Inference — API Usage

POST a file to `/predict` (multipart/form-data):

```bash
curl -X POST -F "file=@/path/to/footprint.jpg" http://127.0.0.1:5000/predict
```

Example response JSON:

```json
{
  "animal": "elephant",
  "confidence": 92.5,
  "wildness_level": "Very High",
  "human_risk_level": "Extreme",
  "estimated_age_group": "Adult"
}
```

> Note: `backend/app.py` requires a trained model file at `backend/model/footprint_cnn_final.keras` or `backend/model/footprint_cnn.h5` and `backend/model/class_indices.json` to map class indices to labels. The training script saves `class_indices.json` automatically.

---

## 🏋️ Training

Place your dataset under `dataset/classification` with `train/`, `val/`, and `test/` subfolders where each class is a subdirectory (standard Keras `flow_from_directory` layout).

Run training:

```bash
python train_model.py
```

Important notes:
- Image size: 224×224
- Training uses mixed precision (`mixed_float16`) to improve performance on Apple Silicon / GPUs
- Final model is saved to `backend/model/footprint_cnn_final.keras` and `class_indices.json` is written to the same folder

---

## 🔧 Utility Scripts

- `scripts/augment_oversample.py <class_name> <target_count>` — light augmentation-based oversampling to balance classes
- `scripts/elephant_targeted_aug.py <num_to_add>` — targeted augmentations for `elephant`

---

## 🧩 Implementation Details

- Model: EfficientNetV2-S (pretrained on ImageNet)
- Input preprocessing: `tensorflow.keras.applications.efficientnet_v2.preprocess_input` (used in both training and inference)
- Segmentation & area estimation: small OpenCV-based functions provide footprint area for age estimation

---

## 🛠 Troubleshooting

- Missing model errors: ensure `backend/model/footprint_cnn_final.keras` (or `.h5`) and `class_indices.json` exist.
- OpenCV read failures: confirm valid image path and supported formats (`jpg`, `jpeg`, `png`)
- On Apple Silicon, use `tensorflow-macos` and `tensorflow-metal` to leverage the GPU for faster training/inference

---

## 🤝 Contributing

Contributions are welcome! Open issues for bugs or feature requests and send PRs for fixes and improvements. Keep changes focused and include tests/documentation when possible.

---

## 📜 License

This repository is unlicensed in-tree; consider adding a license file (e.g., `LICENSE` with MIT) if you want to permit reuse. If you want, I can add a license for you.

---

## 🙏 Acknowledgements

Built with TensorFlow / Keras, OpenCV, and inspiration from EfficientNet V2.

---
