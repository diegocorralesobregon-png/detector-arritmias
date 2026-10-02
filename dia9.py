import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                            recall_score, precision_recall_fscore_support)

torch.set_num_threads(4)

DS1 = ['101','106','108','109','112','114','115','116','118','119','122',
    '124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210',
    '212','213','214','219','221','222','228','231','232','233','234']
CLASES = ['N', 'S', 'V', 'F']
RR_FEATS = ['rr_pre_norm', 'rr_post_norm', 'rr_ratio', 'rr_pre_local']

# ============ 1. Datos: forma de onda + ritmo ============
df = pd.read_csv('caracteristicas.csv', dtype={'registro': str})
X = np.load('ondas.npz')['X']
mascara = df.clase.isin(CLASES).values
X, df = X[mascara], df[mascara].reset_index(drop=True)
df['rr_pre_local'] = df.rr_pre / df.rr_local

# Forma: normalizar cada latido (igual que el día 7)
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

tr = df.registro.isin(DS1).values
te = df.registro.isin(DS2).values

# Ritmo: estandarizar con media y desviación SOLO de DS1
R = df[RR_FEATS].values.astype(np.float32)
media, desv = R[tr].mean(axis=0), R[tr].std(axis=0) + 1e-6
R = (R - media) / desv

idx_clase = {c: i for i, c in enumerate(CLASES)}
y_num = df.clase.map(idx_clase).values

Xtr, Rtr, ytr = torch.tensor(X[tr]).unsqueeze(1), torch.tensor(R[tr]), torch.tensor(y_num[tr])
Xte, Rte = torch.tensor(X[te]).unsqueeze(1), torch.tensor(R[te])

# ============ 2. La red de dos ramas ============
class Hibrido(nn.Module):
    def __init__(self, n_rr=4, n_clases=4):
        super().__init__()
        # Rama de FORMA (la misma CNN del día 7)
        self.forma = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),          # → 512 números
        )
        # Rama de RITMO
        self.ritmo = nn.Sequential(nn.Linear(n_rr, 16), nn.ReLU())   # → 16 números
        # Decisión con las dos juntas
        self.cabeza = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 8 + 16, 64), nn.ReLU(),
            nn.Linear(64, n_clases),
        )

    def forward(self, x, r):
        juntas = torch.cat([self.forma(x), self.ritmo(r)], dim=1)
        return self.cabeza(juntas)

torch.manual_seed(0)
modelo = Hibrido()

cuenta = np.bincount(y_num[tr], minlength=4)
pesos = torch.tensor(np.sqrt(cuenta.sum() / cuenta), dtype=torch.float32)
pesos = pesos / pesos.mean()
criterio = nn.CrossEntropyLoss(weight=pesos)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
cargador = DataLoader(TensorDataset(Xtr, Rtr, ytr), batch_size=256, shuffle=True)

# ============ 3. Entrenar (sin mirar DS2) ============
EPOCAS = 10
for ep in range(EPOCAS):
    t0 = time.time()
    modelo.train()
    total = 0
    for xb, rb, yb in cargador:
        optimizador.zero_grad()
        perdida = criterio(modelo(xb, rb), yb)
        perdida.backward()
        optimizador.step()
        total += perdida.item() * len(xb)
    print(f"Época {ep + 1}/{EPOCAS}  pérdida: {total / len(Xtr):.4f}  ({time.time() - t0:.0f} s)", flush=True)

# ============ 4. Evaluar en pacientes nuevos ============
modelo.eval()
with torch.no_grad():
    logits = torch.cat([modelo(Xte[i:i + 4096], Rte[i:i + 4096]) for i in range(0, len(Xte), 4096)])
pred = np.array(CLASES)[logits.argmax(1).numpy()]
real = df.clase.values[te]
reg = df.registro.values[te]

print("\n" + "=" * 60)
print("HÍBRIDO (forma + ritmo): DIVISIÓN ENTRE PACIENTES")
print("=" * 60)
print(classification_report(real, pred, labels=CLASES, digits=3, zero_division=0))

# Comparación con los días 6 y 7
_, r_h, _, _ = precision_recall_fscore_support(real, pred, labels=CLASES, zero_division=0)
comp = pd.DataFrame({'RF': [89.2, 17.4, 88.5, 10.3],
                    'CNN': [96.4, 6.3, 90.3, 0.5]}, index=CLASES)
comp['Híbrido'] = (100 * r_h).round(1)
print("Sensibilidad (%) por clase:")
print(comp)

# Los casos difíciles del día 8
sin232 = reg != '232'
se_s = 100 * (pred[sin232 & (real == 'S')] == 'S').mean()
print(f"\nSe S sin el 232: {se_s:.1f} %   (RF: 67,0 % | CNN: 25,1 %)")
for r_, rf_, cnn_ in [('105', 97.6, 17.1), ('213', 49.5, 84.1), ('228', 99.4, 73.8), ('233', 78.8, 96.6)]:
    m = (reg == r_) & (real == 'V')
    print(f"Se V registro {r_}: {100 * (pred[m] == 'V').mean():.1f} %   (RF: {rf_} | CNN: {cnn_})")

np.save('pred_hibrido.npy', pred)
torch.save(modelo.state_dict(), 'hibrido.pt')

fig, a = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay.from_predictions(real, pred, labels=CLASES, normalize='true',
                                        values_format='.1%', ax=a, colorbar=False, cmap='Greens')
a.set_title('Híbrido: matriz de confusión (DS2)')
a.set_xlabel('Predicho'); a.set_ylabel('Real')
plt.tight_layout()
plt.savefig('figuras/matriz_hibrido.png', dpi=150)
plt.show()