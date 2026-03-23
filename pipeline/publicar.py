"""
Publica o model.pkl no Hugging Face Hub.

Uso:
    python pipeline/publicar.py

Requer:
    model.pkl  (gerado por pipeline/treinar.py)
    .env       com HF_TOKEN e HF_REPO_ID definidos

Exemplo de .env:
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    HF_REPO_ID=seu-usuario/bella-tavola-model
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN não encontrado. Defina no arquivo .env.")
if not HF_REPO_ID:
    raise ValueError("HF_REPO_ID não encontrado. Defina no arquivo .env.")

login(token=HF_TOKEN)

api = HfApi()

# Cria o repositório se não existir
api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
print(f"Repositório: {HF_REPO_ID}")

# Faz upload do modelo
api.upload_file(
    path_or_fileobj=str(ROOT / "model.pkl"),
    path_in_repo="model.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

print(f"Modelo publicado em: https://huggingface.co/{HF_REPO_ID}")
