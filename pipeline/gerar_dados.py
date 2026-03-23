"""
Geração do dataset sintético para o modelo de risco de cancelamento.

Regras de negócio embutidas na geração:
- historico_cancelamentos alto → maior probabilidade de cancelamento
- hora_pedido > 22 → maior risco (horário tardio)
- distancia_entrega alta + num_itens baixo → maior risco
- valor_pedido muito baixo (< 20) → leve aumento de risco

Uso:
    python pipeline/gerar_dados.py

Saída:
    data/pedidos.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SEED = 42
N = 2000

rng = np.random.default_rng(SEED)

# --- Features ---
valor_pedido = rng.uniform(15, 300, N)
hora_pedido = rng.integers(0, 24, N)
num_itens = rng.integers(1, 10, N)
historico_cancelamentos = rng.integers(0, 8, N)
distancia_entrega = rng.uniform(0.5, 20, N)

# --- Score de risco (antes do threshold) ---
risco = np.zeros(N)

# historico_cancelamentos: principal preditor
risco += historico_cancelamentos * 0.45

# hora tardia (> 22)
risco += np.where(hora_pedido > 22, 2.0, 0)

# distancia alta com poucos itens
risco += np.where((distancia_entrega > 12) & (num_itens <= 2), 1.5, 0)
risco += np.where(distancia_entrega > 15, 0.8, 0)

# valor muito baixo
risco += np.where(valor_pedido < 20, 0.6, 0)

# ruído
risco += rng.normal(0, 0.5, N)

# Converte score em probabilidade via sigmoid e define threshold
prob_cancelamento = 1 / (1 + np.exp(-risco + 1.5))
cancelamento = (prob_cancelamento > 0.5).astype(int)

df = pd.DataFrame({
    "valor_pedido": np.round(valor_pedido, 2),
    "hora_pedido": hora_pedido,
    "num_itens": num_itens,
    "historico_cancelamentos": historico_cancelamentos,
    "distancia_entrega": np.round(distancia_entrega, 2),
    "cancelamento": cancelamento,
})

data_dir = ROOT / "data"
data_dir.mkdir(exist_ok=True)
df.to_csv(data_dir / "pedidos.csv", index=False)

print(f"Dataset gerado: {len(df)} amostras")
print(f"Cancelamentos: {cancelamento.sum()} ({cancelamento.mean():.1%})")
print(f"Salvo em: {data_dir / 'pedidos.csv'}")
