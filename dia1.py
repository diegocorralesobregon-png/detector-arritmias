import wfdb
import matplotlib.pyplot as plt

# Registro 100 de la base MIT-BIH (se descarga de PhysioNet)
rec = wfdb.rdrecord('100', pn_dir='mitdb')
ann = wfdb.rdann('100', 'atr', pn_dir='mitdb')

fs = rec.fs                      # frecuencia de muestreo
senal = rec.p_signal[:, 0]       # derivación MLII
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración: {len(senal)/fs/60:.1f} min")
print(f"Latidos anotados: {len(ann.sample)}")

# Primeros 10 segundos con las anotaciones
n = 10 * fs
t = [i / fs for i in range(n)]
plt.figure(figsize=(14, 4))
plt.plot(t, senal[:n])
for s, sym in zip(ann.sample, ann.symbol):
    if s < n:
        plt.annotate(sym, (s / fs, senal[s]), color='red')
plt.xlabel('Tiempo (s)'); plt.ylabel('mV')
plt.title('MIT-BIH registro 100: primeros 10 s')
plt.show()