# Probability AI Assistant — Railway Ready

This package is the Railway-ready version of the Flask Probability AI Assistant.

## Local run
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py

## Railway run
Build command:
pip install -r requirements.txt

Start command:
gunicorn app:app

## SQLite persistence
Attach a Railway Volume and mount it to:
`/app/data`

This project supports:
- DATA_DIR
- RAILWAY_VOLUME_MOUNT_PATH
- fallback to local data
