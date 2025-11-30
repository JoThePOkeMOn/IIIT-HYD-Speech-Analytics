Github link: https://github.com/JoThePOkeMOn/IIIT-HYD-Speech-Analytics

# AccentAware – How to Run

This document explains **only** how to set up and run the AccentAware app, and which
packages (and their versions) are required.

---

## 1. Prerequisites

1. **Python (64-bit) 3.9–3.11**  
   Download from: https://www.python.org/downloads/  
   When installing on Windows, tick **“Add Python to PATH”**.

2. **Git (optional, only if you want to clone)**  
   https://git-scm.com/downloads

---

## 2. Project Files

Your project folder should look like:

```text
Accentaware/
├── app.py
├── requirements.txt
├── models/
│   ├── accent_clf_latest.pkl
│   ├── accent_scaler_latest.pkl
│   ├── accent_label_encoder_latest.pkl
│   └── config.json
└── venv/                 # (created in the next step; may not exist yet)
Important: models/ must contain the three .pkl files and config.json
generated from training.
Make sure config.json contains:


"sample_rate": 16000

3. Create and Activate Virtual Environment
From inside the Accentaware folder:

3.1 Windows

python -m venv venv
venv\Scripts\activate

3.2 macOS / Linux

python3 -m venv venv
source venv/bin/activate

If activation is successful, your terminal prompt will show (venv).

Official Python venv docs:
https://docs.python.org/3/library/venv.html

4. Install Required Packages
All required packages and versions are listed in requirements.txt.
Install them with:


pip install --upgrade pip
pip install -r requirements.txt
Typical key dependencies in requirements.txt include:

torch / torchvision / torchaudio

transformers

scikit-learn

numpy, scipy

librosa, soundfile

gradio

You can open requirements.txt to see the exact pinned versions used for this project.

4.1 If PyTorch GPU fails on Windows
If you get CUDA-related errors and do not need GPU, install the CPU-only wheel:


pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
Official PyTorch install page (choose CPU, your OS, and Python version):
https://pytorch.org/get-started/locally/

5. Running the App
From the Accentaware folder, with the virtual environment activated:


python app.py
If everything is correct, you should see output similar to:


Running on local URL:  http://127.0.0.1:7860/
Open that link in your browser to access the AccentAware web interface.

The app loads the HuBERT model and the trained classifier from models/,
then exposes a Gradio UI where you can upload or record audio.
