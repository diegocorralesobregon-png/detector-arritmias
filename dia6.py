import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score

# División estándar entre pacientes (de Chazal et al., 2004)
DS1 = ['101','106','108','109','112','114','115','116','118','119','122',
    '124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210',
    '212','213','214','219','221','222','228','231','232','233','234']

CLASES = ['N', 'S', 'V', 'F']

df = pd.read_csv('caracteristicas.csv', dtype={'registro': str})
df = df[df.clase.isin(CLASES)]                      # Q casi no existe sin marcapasos
df['rr_pre_local'] = df.rr_pre / df.rr_local        # prematuridad respecto al ritmo reciente

# Solo características RELATIVAS o de forma (los RR absolutos no generalizan entre pacientes)
FEATS = ['rr_pre_norm', 'rr_post_norm', 'rr_ratio', 'rr_pre_local',
        'ancho_qrs_ms', 'amp_r', 'amp_q', 'amp_s', 'amp_t', 'energia_qrs']

def nuevo_modelo():
    return RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                min_samples_leaf=5, n_jobs=-1, random_state=0)

# ================= A) La TRAMPA: división aleatoria de latidos =================
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df.clase, test_size=0.5,
                                    stratify=df.clase, random_state=0)
m_trampa = nuevo_modelo().fit(Xtr, ytr)
print("=" * 60)
print("A) DIVISIÓN ALEATORIA (latidos del mismo paciente en ambos lados)")
print("=" * 60)
print(classification_report(yte, m_trampa.predict(Xte), labels=CLASES, digits=3, zero_division=0))

# ================= B) Lo HONESTO: división entre pacientes =================
train = df[df.registro.isin(DS1)]
test = df[df.registro.isin(DS2)]
modelo = nuevo_modelo().fit(train[FEATS], train.clase)
pred = modelo.predict(test[FEATS])

print("=" * 60)
print("B) DIVISIÓN ENTRE PACIENTES (DS1 → DS2)")
print("=" * 60)
print(f"Entrenamiento: {len(train)} latidos | Prueba: {len(test)} latidos")
print(classification_report(test.clase, pred, labels=CLASES, digits=3, zero_division=0))

se_v = recall_score(test.clase, pred, labels=['V'], average=None)[0] * 100
print(f"🎯 Sensibilidad V: {se_v:.1f} %  (objetivo ≥ 85 %)  {'✅' if se_v >= 85 else '❌'}")

# ================= Gráficos =================
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(test.clase, pred, labels=CLASES,
                                        ax=ax[0], colorbar=False, cmap='Blues')
ax[0].set_title('Matriz de confusión (DS2, pacientes nuevos)')
ax[0].set_xlabel('Predicho por el modelo')
ax[0].set_ylabel('Real (cardiólogos)')

imp = pd.Series(modelo.feature_importances_, index=FEATS).sort_values()
imp.plot.barh(ax=ax[1], color='tab:green')
ax[1].set_title('Importancia de cada característica')
plt.tight_layout()
plt.show()