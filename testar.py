"""
Testa o model.pkl localmente com exemplos representativos.

Uso:
    python testar.py

Requer:
    model.pkl  (gerado por pipeline/treinar.py)
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent

FEATURES = [
    "valor_pedido",
    "hora_pedido",
    "num_itens",
    "historico_cancelamentos",
    "distancia_entrega",
]

# Casos de teste: (descrição, [valor_pedido, hora_pedido, num_itens, historico_cancelamentos, distancia_entrega])
CASOS = [
    ("Pedido normal — baixo risco",       [95.0,  12, 3, 0, 3.0]),
    ("Horário tardio — risco elevado",    [80.0,  23, 2, 1, 5.0]),
    ("Histórico alto — risco elevado",    [120.0, 14, 4, 6, 4.0]),
    ("Distância alta + poucos itens",     [60.0,  18, 1, 0, 16.0]),
    ("Valor baixo — leve risco",          [18.0,  11, 1, 0, 2.0]),
    ("Combinação de fatores de risco",    [25.0,  23, 1, 5, 18.0]),
]


def main():
    model = joblib.load(ROOT / "model.pkl")
    print(f"Modelo carregado: {type(model).__name__}\n")
    print(f"{'Caso':<40} {'Pred':>6} {'Prob cancel':>12}  Label")
    print("-" * 72)

    for descricao, valores in CASOS:
        X = pd.DataFrame([valores], columns=FEATURES)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        label = "cancelamento provavel" if pred == 1 else "pedido normal"
        print(f"{descricao:<40} {pred:>6} {prob:>11.1%}  {label}")


if __name__ == "__main__":
    main()
