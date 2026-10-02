import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def filtrar_ecg(senal, fs, low=0.5, high=40, orden=3):
    """Filtro paso banda Butterworth de fase cero."""
    nyq = fs / 2
    b, a = butter(orden, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, senal)

def detectar_picos_r(senal, fs):
    """Detector de QRS tipo Pan-Tompkins simplificado.
    'senal' debe estar ya filtrada."""
    # 1. Derivada
    d = np.diff(senal, prepend=senal[0])
    # 2. Cuadrado
    c = d ** 2
    # 3. Media móvil de 150 ms
    w = int(0.15 * fs)
    energia = np.convolve(c, np.ones(w) / w, mode='same')
    # 4. Picos de energía separados al menos 250 ms
    umbral = 0.15 * np.percentile(energia, 99)
    candidatos, _ = find_peaks(energia, height=umbral, distance=int(0.25 * fs))
    # 5. Afinar: máximo de |señal| en ±100 ms alrededor de cada candidato
    r = int(0.1 * fs)
    picos = []
    for p in candidatos:
        ini, fin = max(0, p - r), min(len(senal), p + r)
        picos.append(ini + np.argmax(np.abs(senal[ini:fin])))
    return np.unique(np.array(picos))