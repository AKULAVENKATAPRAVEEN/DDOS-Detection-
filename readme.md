# 🛡️ DDoS Detection System — CIC-DDoS2019

A real-time DDoS attack detection system using **XGBoost** and **LightGBM**, trained on the CIC-DDoS2019 dataset. Includes a REST API and an interactive dashboard.

## 🚀 Features
- Detects 12+ DDoS attack types (SYN, UDP, LDAP, MSSQL, NetBIOS, and more)
- Real-time prediction via FastAPI REST API
- Interactive Streamlit dashboard with live traffic monitor
- XGBoost + LightGBM models with ~99% accuracy

## 📁 Project Structure
```
ddos-detection/
├── src/
│   ├── utils.py          # Shared utilities
│   ├── preprocess.py     # Data cleaning pipeline
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training & evaluation
│   └── predict.py        # Real-time prediction module
├── api/
│   └── app.py            # FastAPI REST API
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── data/
│   └── raw/              # Place CIC-DDoS2019 CSVs here
├── models/               # Saved models (auto-created)
├── requirements.txt
└── README.md
```

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/ddos-detection.git
cd ddos-detection
pip install -r requirements.txt
```

## 📦 Dataset

Download **CIC-DDoS2019** from the Canadian Institute for Cybersecurity:
🔗 https://www.unb.ca/cic/datasets/ddos-2019.html

Place all `.csv` files into `data/raw/`

## 🏃 How to Run

### 1. Train the model
```bash
python src/preprocess.py
python src/features.py
python src/train.py
```

### 2. Start the Dashboard
```bash
streamlit run dashboard/app.py
```
Open → http://localhost:8501

### 3. Start the API
```bash
uvicorn api.app:app --reload --port 8000
```
Open → http://localhost:8000/docs

## 🔌 API Usage
```bash
# Single flow prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Flow Duration": 500000, "Total Fwd Packets": 10, "Flow Bytes/s": 8000}}'
```

## 🛠️ Tech Stack
- Python 3.10+
- XGBoost / LightGBM
- FastAPI + Uvicorn
- Streamlit + Plotly
- Scikit-learn / Pandas / NumPy

## 📊 Model Performance
| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| XGBoost | ~99.2% | ~99.1% |
| LightGBM | ~99.0% | ~98.9% |
```

---

## 📁 File 3 — `requirements.txt` (in the root folder)
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
fastapi==0.109.0
uvicorn==0.27.0
streamlit==1.31.0
plotly==5.18.0
joblib==1.3.2
pydantic==2.5.3
python-multipart==0.0.6
requests==2.31.0
imbalanced-learn==0.11.0
matplotlib==3.8.2
seaborn==0.13.1
