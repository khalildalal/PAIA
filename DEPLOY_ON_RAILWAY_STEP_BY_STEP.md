# Railway Deployment Guide (SQLite + Persistent Volume)

## 1. Test locally
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py

## 2. Push to GitHub
git init
git add .
git commit -m "Railway-ready app"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main

## 3. Create the project on Railway
- New Project
- Deploy from GitHub repo
- Select your repository

## 4. Add a Volume
Mount path:
`/app/data`

## 5. Add variables
OPENAI_API_KEY=your_openai_api_key_here
FLASK_SECRET_KEY=replace_with_a_long_random_secret
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

Optional:
DATA_DIR=/app/data

## 6. Build and start
Build:
pip install -r requirements.txt

Start:
gunicorn app:app

## 7. Generate a public domain
Use Railway's networking/domain settings.

## 8. Test persistence
Restart or redeploy, then confirm users and quiz data still exist.
