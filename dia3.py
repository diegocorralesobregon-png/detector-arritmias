import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import welch
from preprocesado import filtrar_ecg

# Descarga local de la base (solo la primera vez, unos minutos)
if not os.path.exists('mitdb'):
    print("Descargando MIT-BIH...")
    wfdb.dl_database('mitdb', 'mitdb')

REG = '105'   # registro con bastante ruido
rec = wfdb.rdrecord(f'mitdb/{REG}')
fs = rec.fs
x = rec.p_signal[:, 0]        # original
y = filtrar_ecg(x, fs)        # filtrada

n = 10 * fs
t = np.arange(n) / fs

fig, ax = plt.subplots(3, 1, figsize=(14, 9))

ax[0].plot(t, x[:n])
ax[0].set_title(f'Registro {REG}: señal original')

ax[1].plot(t, y[:n], color='green')
ax[1].set_title('Señal filtrada (0,5–40 Hz)')
ax[1].set_xlabel('Tiempo (s)')

# Espectro: cuánta energía hay en cada frecuencia
f1, P1 = welch(x, fs, nperseg=4096)
f2, P2 = welch(y, fs, nperseg=4096)
ax[2].semilogy(f1, P1, label='original')
ax[2].semilogy(f2, P2, label='filtrada', color='green')
ax[2].axvline(0.5, ls='--', color='gray')
ax[2].axvline(40, ls='--', color='gray')
ax[2].set_xlim(0, 100)
ax[2].set_xlabel('Frecuencia (Hz)')
ax[2].set_title('Espectro de potencia')
ax[2].legend()

plt.tight_layout()
plt.show()