import numpy as np
import matplotlib.pyplot as plt
import wfdb
from preprocesado import filtrar_ecg, detectar_picos_r

LATIDOS = set('NLRejAaJSVEF/fQ')   # símbolos que son latidos

def evaluar(det, ref, tol):
    """Cuántos latidos reales (ref) tienen una detección a menos de 'tol' muestras."""
    tp = 0
    idx = np.searchsorted(det, ref)
    for r, i in zip(ref, idx):
        dist = [abs(det[j] - r) for j in (i - 1, i) if 0 <= j < len(det)]
        if dist and min(dist) <= tol:
            tp += 1
    fn = len(ref) - tp
    fp = len(det) - tp
    se = 100 * tp / (tp + fn)          # sensibilidad: latidos reales encontrados
    pp = 100 * tp / (tp + fp)          # predictividad positiva: detecciones correctas
    return se, pp, fn, fp

# --- Parte A: evaluar el detector en varios registros ---
print(f"{'Reg':>5} {'Latidos':>8} {'Se %':>7} {'+P %':>7} {'FN':>5} {'FP':>5}")
for reg in ['100', '105', '108', '119', '203', '208']:
    rec = wfdb.rdrecord(f'mitdb/{reg}')
    ann = wfdb.rdann(f'mitdb/{reg}', 'atr')
    fs = rec.fs
    y = filtrar_ecg(rec.p_signal[:, 0], fs)
    det = detectar_picos_r(y, fs)
    ref = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in LATIDOS])
    se, pp, fn, fp = evaluar(det, ref, tol=int(0.075 * fs))
    print(f"{reg:>5} {len(ref):>8} {se:>7.2f} {pp:>7.2f} {fn:>5} {fp:>5}")

# --- Parte B: ver los picos y el tacograma RR del registro 119 ---
REG = '119'
rec = wfdb.rdrecord(f'mitdb/{REG}')
fs = rec.fs
y = filtrar_ecg(rec.p_signal[:, 0], fs)
picos = detectar_picos_r(y, fs)
rr = np.diff(picos) / fs * 1000      # intervalos RR en ms

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
n = 10 * fs
t = np.arange(n) / fs
ax[0].plot(t, y[:n])
p = picos[picos < n]
ax[0].plot(p / fs, y[p], 'rx', markersize=10)
ax[0].set_title(f'Registro {REG}: picos R detectados')
ax[0].set_xlabel('Tiempo (s)')

ax[1].plot(picos[1:] / fs / 60, rr, '.', markersize=3)
ax[1].set_title('Tacograma: intervalo RR de cada latido')
ax[1].set_xlabel('Tiempo (min)')
ax[1].set_ylabel('RR (ms)')
plt.tight_layout()
plt.show()