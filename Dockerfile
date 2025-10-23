# Image Python légère
FROM python:3.12-slim

# Paquets système minimaux (git requis par torch.hub pour Dinov2)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

# Copie du code
COPY backend.py .
COPY frontend.html .
COPY admin.html .
COPY approve.html .
COPY auth.py .
COPY user_manager.py .

# Optionnel : répertoire cache torch/hub (évite d'encombrer /root)
ENV TORCH_HOME=/app/.cache/torch

# Railway fournit $PORT. On expose un port par défaut pour la lisibilité.
EXPOSE 8080

# Commande de démarrage (Gunicorn, threads pour I/O, port Railway)
# NB: forme "sh -c" pour l'expansion des variables d'env.
CMD ["sh", "-c", "gunicorn -w ${WEB_CONCURRENCY:-2} -k gthread --threads ${GUNICORN_THREADS:-8} -t 180 -b 0.0.0.0:${PORT:-8080} backend:app"]
