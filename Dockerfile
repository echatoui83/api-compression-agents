# Utiliser une version légère de Linux avec Python 3.10
FROM python:3.10-slim

# Installer Tesseract-OCR (pour ton Agent Analyseur)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail sur le serveur
WORKDIR /app

# Copier tous tes fichiers (api.py, agents/, requirements.txt) dans le serveur
COPY . /app

# Installer tes bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Lancer ton API sur le port par défaut de Render
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:10000"]