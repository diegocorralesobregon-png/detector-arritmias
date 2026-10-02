import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from preprocesado import filtrar_ecg

AAMI = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q',
}
EXCLUIR = {'102', '104', '107', '217'}   # registros con marcapasos

registros = sorted(os.path.basename(f)[:-4] for f in glob.glob('mitdb/*.hea'))
registros = [r for r in registros if r not in EXCLUIR]
print(f"{len(registros)} registros")

def ms(t, fs):
    """Convierte milisegundos a número de muestras."""
    return int(t * fs / 1000)

filas = []
for reg in registros:
    rec = wfdb.rdrecord(f'mitdb/{reg}')
    ann = wfdb.rdann(f'mitdb/{reg}', 'atr')
    fs = rec.fs
    canal = rec.sig_name.index('MLII')          # en algún registro MLII no es el canal 0
    y = filtrar_ecg(rec.p_signal[:, canal], fs)

    # Solo anotaciones que son latidos
    pos = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI])
    sim = [sym for sym in ann.symbol if sym in AAMI]
    rr = np.diff(pos) / fs                       # en segundos
    rr_medio = np.median(rr)                     # RR típico del paciente

    for i in range(1, len(pos) - 1):             # sin el primero ni el último
        r = pos[i]
        if r - ms(250, fs) < 0 or r + ms(450, fs) >= len(y):
            continue

        rr_pre, rr_post = rr[i - 1], rr[i]
        rr_local = np.mean(rr[max(0, i - 10):i])

        # QRS: ventana de ±100 ms
        qrs = y[r - ms(100, fs): r + ms(100, fs)]
        principal = qrs[np.argmax(np.abs(qrs))]  # deflexión principal (con signo)
        ancho = np.sum(np.abs(qrs) > 0.5 * abs(principal)) / fs * 1000

        seg_t = y[r + ms(150, fs): r + ms(400, fs)]

        filas.append({
            'registro': reg,
            'muestra': r,
            'simbolo': sim[i],
            'clase': AAMI[sim[i]],
            'rr_pre': rr_pre,
            'rr_post': rr_post,
            'rr_local': rr_local,
            'rr_pre_norm': rr_pre / rr_medio,
            'rr_post_norm': rr_post / rr_medio,
            'rr_ratio': rr_post / rr_pre,
            'ancho_qrs_ms': ancho,
            'amp_r': principal,
            'amp_q': y[r - ms(50, fs): r].min(),
            'amp_s': y[r: r + ms(80, fs)].min(),
            'amp_t': seg_t[np.argmax(np.abs(seg_t))],
            'energia_qrs': np.sum(qrs ** 2),
        })
    print(f"{reg} ✓")

df = pd.DataFrame(filas)
df.to_csv('caracteristicas.csv', index=False)

print(f"\nTotal de latidos: {len(df)}")
print(df['clase'].value_counts())

print("\nValor medio de cada característica por clase:")
cols = ['rr_pre_norm', 'rr_post_norm', 'rr_ratio', 'ancho_qrs_ms', 'amp_r']
print(df.groupby('clase')[cols].mean().round(2))

# --- Gráfico: ¿se separan las clases con solo 2 características? ---
muestra = pd.concat([
    df[df.clase == 'N'].sample(3000, random_state=0),   # submuestreo de N para que no tape lo demás
    df[df.clase.isin(['S', 'V', 'F'])],
])
colores = {'N': 'tab:blue', 'S': 'tab:orange', 'V': 'tab:red', 'F': 'tab:green'}
plt.figure(figsize=(10, 7))
for c in ['N', 'S', 'V', 'F']:
    d = muestra[muestra.clase == c]
    plt.scatter(d.rr_pre_norm, d.ancho_qrs_ms, s=4, alpha=0.4, c=colores[c], label=c)
plt.xlim(0.2, 2)
plt.xlabel('RR anterior normalizado (1 = ritmo típico del paciente)')
plt.ylabel('Anchura del QRS (ms)')
plt.title('Cada punto es un latido')
plt.legend(markerscale=4)
plt.show()