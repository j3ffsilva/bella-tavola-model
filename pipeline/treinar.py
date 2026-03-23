"""
Treinamento do modelo de risco de cancelamento.

Uso:
    python pipeline/treinar.py

Requer:
    data/pedidos.csv  (gerado por pipeline/gerar_dados.py)

Saída:
    model.pkl
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent

# --- Carrega dados ---
df = pd.read_csv(ROOT / "data" / "pedidos.csv")

FEATURES = [
    "valor_pedido",
    "hora_pedido",
    "num_itens",
    "historico_cancelamentos",
    "distancia_entrega",
]
TARGET = "cancelamento"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Treina modelo ---
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

# --- Avalia ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["normal", "cancelamento"]))

# --- Salva modelo ---
model_path = ROOT / "model.pkl"
joblib.dump(model, model_path)
print(f"Modelo salvo em: {model_path}")
