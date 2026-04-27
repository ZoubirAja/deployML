FROM python:3.12-slim

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code
COPY . .

# Lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]