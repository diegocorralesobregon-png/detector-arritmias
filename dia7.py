import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score
from preprocesado import filtrar_ecg

DS1 = ['101','106','108','109','112','114','115','116','118','119','122',
       '124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']
CLASES = ['N', 'S', 'V', 'F']
ANTES, DESPUES = 90, 144        # 250 ms antes y 400 ms después del R (a 360 Hz)

# ============ 1. Recortar la forma de onda de cada latido (solo la 1ª vez) ============
df = pd.read_csv('caracteristicas.csv', dtype={'registro': str})
if not os.path.exists('ondas.npz'):
    X = np.zeros((len(df), ANTES + DESPUES), dtype=np.float32)
    for reg, grupo in df.groupby('registro'):
        rec = wfdb.rdrecord(f'mitdb/{reg}')
        canal = rec.sig_name.index('MLII')
        y = filtrar_ecg(rec.p_signal[:, canal], rec.fs)
        for idx, r in zip(grupo.index, grupo.muestra):
            X[idx] = y[r - ANTES: r + DESPUES]
        print(f"{reg} ✓")
    np.savez_compressed('ondas.npz', X=X)
X = np.load('ondas.npz')['X']

mascara = df.clase.isin(CLASES).values
X, df = X[mascara], df[mascara].reset_index(drop=True)

# Normalizar cada latido (cada paciente/electrodo tiene amplitudes distintas)
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

idx_clase = {c: i for i, c in enumerate(CLASES)}
y_num = df.clase.map(idx_clase).values
tr = df.registro.isin(DS1).values
te = df.registro.isin(DS2).values

Xtr = torch.tensor(X[tr]).unsqueeze(1)    # forma (n, 1, 234)
ytr = torch.tensor(y_num[tr])
Xte = torch.tensor(X[te]).unsqueeze(1)

# ============ 2. La red ============
class CNN(nn.Module):
    def __init__(self, n_clases=4):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 8, n_clases),
        )
    def forward(self, x):
        return self.red(x)

torch.manual_seed(0)
modelo = CNN()

cuenta = np.bincount(y_num[tr], minlength=4)
pesos = torch.tensor(np.sqrt(cuenta.sum() / cuenta), dtype=torch.float32)
pesos = pesos / pesos.mean()
print("Latidos de entrenamiento por clase:", dict(zip(CLASES, cuenta)))
print("Pesos de clase:", dict(zip(CLASES, pesos.numpy().round(2))))

criterio = nn.CrossEntropyLoss(weight=pesos)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
cargador = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

# ============ 3. Entrenar (sin mirar DS2) ============
EPOCAS = 10
for ep in range(EPOCAS):
    modelo.train()
    total = 0
    for xb, yb in cargador:
        optimizador.zero_grad()
        perdida = criterio(modelo(xb), yb)
        perdida.backward()
        optimizador.step()
        total += perdida.item() * len(xb)
    print(f"Época {ep + 1}/{EPOCAS}  pérdida: {total / len(Xtr):.4f}")

# ============ 4. Evaluar en pacientes nuevos (DS2) ============
modelo.eval()
with torch.no_grad():
    logits = torch.cat([modelo(Xte[i:i + 4096]) for i in range(0, len(Xte), 4096)])
pred = np.array(CLASES)[logits.argmax(1).numpy()]
real = df.clase.values[te]

print("\n" + "=" * 60)
print("CNN 1D: DIVISIÓN ENTRE PACIENTES (DS1 → DS2)")
print("=" * 60)
print(classification_report(real, pred, labels=CLASES, digits=3, zero_division=0))
se_v = recall_score(real, pred, labels=['V'], average=None)[0] * 100
print(f"🎯 Sensibilidad V: {se_v:.1f} %  (RF: 88,5 %)")

np.save('pred_cnn.npy', pred)               # para comparar mañana
torch.save(modelo.state_dict(), 'cnn.pt')

# ============ 5. Gráficos ============
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(real, pred, labels=CLASES,
                                        ax=ax[0], colorbar=False, cmap='Purples')
ax[0].set_title('CNN: matriz de confusión (DS2)')
ax[0].set_xlabel('Predicho'); ax[0].set_ylabel('Real')

t = (np.arange(ANTES + DESPUES) - ANTES) / 360 * 1000
colores = {'N': 'tab:blue', 'S': 'tab:orange', 'V': 'tab:red', 'F': 'tab:green'}
for c in CLASES:
    ax[1].plot(t, X[tr & (df.clase.values == c)].mean(axis=0), label=c, color=colores[c], lw=2)
ax[1].axvline(0, ls='--', color='gray')
ax[1].set_title('Latido medio de cada clase (lo que "ve" la red)')
ax[1].set_xlabel('Tiempo respecto al pico R (ms)')
ax[1].legend()
plt.tight_layout()
plt.show()