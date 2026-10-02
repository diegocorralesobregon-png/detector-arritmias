import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay

DS1 = ['101','106','108','109','112','114','115','116','118','119','122',
       '124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']
CLASES = ['N', 'S', 'V', 'F']
FEATS = ['rr_pre_norm', 'rr_post_norm', 'rr_ratio', 'rr_pre_local',
         'ancho_qrs_ms', 'amp_r', 'amp_q', 'amp_s', 'amp_t', 'energia_qrs']
os.makedirs('figuras', exist_ok=True)

# ============ 1. Predicciones de los dos modelos ============
df = pd.read_csv('caracteristicas.csv', dtype={'registro': str})
df = df[df.clase.isin(CLASES)].reset_index(drop=True)
df['rr_pre_local'] = df.rr_pre / df.rr_local
train = df[df.registro.isin(DS1)]
test = df[df.registro.isin(DS2)]

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                            min_samples_leaf=5, n_jobs=-1, random_state=0)
rf.fit(train[FEATS], train.clase)

res = pd.DataFrame({
    'registro': test.registro.values,
    'real': test.clase.values,
    'rf': rf.predict(test[FEATS]),
    'cnn': np.load('pred_cnn.npy', allow_pickle=True),
})
assert len(res) == len(np.load('pred_cnn.npy', allow_pickle=True)), "Los latidos no coinciden"

# ============ 2. Tabla de resultados ============
filas = []
for modelo in ['rf', 'cnn']:
    p, r, _, _ = precision_recall_fscore_support(res.real, res[modelo], labels=CLASES, zero_division=0)
    for i, c in enumerate(CLASES):
        filas.append({'modelo': modelo.upper(), 'clase': c,
                      'Se %': round(100 * r[i], 1), '+P %': round(100 * p[i], 1)})
tabla = pd.DataFrame(filas).pivot(index='clase', columns='modelo', values=['Se %', '+P %']).loc[CLASES]
print("=" * 50)
print("RESULTADOS EN PACIENTES NUEVOS (DS2)")
print("=" * 50)
print(tabla)
for m in ['rf', 'cnn']:
    print(f"Accuracy {m.upper()}: {100 * (res.real == res[m]).mean():.1f} %")
tabla.to_csv('resultados.csv')

# ============ 3. Paciente por paciente ============
filas = []
for reg, g in res.groupby('registro'):
    fila = {'registro': reg, 'latidos': len(g)}
    for c in ['S', 'V']:
        es_c = g.real == c
        fila[f'n_{c}'] = int(es_c.sum())
        for m in ['rf', 'cnn']:
            fila[f'Se{c}_{m}'] = round(100 * (g.loc[es_c, m] == c).mean(), 1) if es_c.sum() > 0 else '-'
    filas.append(fila)
por_paciente = pd.DataFrame(filas)
print("\nSENSIBILIDAD S y V POR PACIENTE")
print(por_paciente.to_string(index=False))

total_s = (res.real == 'S').sum()
s_232 = ((res.real == 'S') & (res.registro == '232')).sum()
print(f"\nEl registro 232 aporta {s_232} de los {total_s} latidos S de DS2 ({100 * s_232 / total_s:.0f} %)")

# Se de S SIN el 232, para ver cuánto pesa
sin232 = res[res.registro != '232']
for m in ['rf', 'cnn']:
    se = 100 * (sin232.loc[sin232.real == 'S', m] == 'S').mean()
    print(f"Se S {m.upper()} sin el 232: {se:.1f} %")

# ============ 4. Figuras para el README ============
# Figura 1: matrices de confusión normalizadas (% por fila)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, m, cmap, nombre in [(ax[0], 'rf', 'Blues', 'Random Forest'), (ax[1], 'cnn', 'Purples', 'CNN 1D')]:
    ConfusionMatrixDisplay.from_predictions(res.real, res[m], labels=CLASES, normalize='true',
                                            values_format='.1%', ax=a, colorbar=False, cmap=cmap)
    a.set_title(f'{nombre}: matriz de confusión (DS2)')
    a.set_xlabel('Predicho'); a.set_ylabel('Real')
plt.tight_layout()
plt.savefig('figuras/matrices_confusion.png', dpi=150)

# Figura 2: sensibilidad por clase
se = tabla['Se %']
x = np.arange(len(CLASES))
fig, a = plt.subplots(figsize=(9, 5))
b1 = a.bar(x - 0.2, se['RF'], 0.4, label='Random Forest', color='tab:blue')
b2 = a.bar(x + 0.2, se['CNN'], 0.4, label='CNN 1D', color='tab:purple')
a.bar_label(b1, fmt='%.1f'); a.bar_label(b2, fmt='%.1f')
a.axhline(85, ls='--', color='red', lw=1, label='Objetivo V (85 %)')
a.set_xticks(x, CLASES); a.set_ylim(0, 105)
a.set_ylabel('Sensibilidad (%)')
a.set_title('Sensibilidad por clase en pacientes nuevos (DS2)')
a.legend()
plt.tight_layout()
plt.savefig('figuras/sensibilidad_por_clase.png', dpi=150)

plt.show()
print("\nFiguras guardadas en la carpeta 'figuras/'")