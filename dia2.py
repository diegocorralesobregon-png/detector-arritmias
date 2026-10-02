import wfdb
import pandas as pd
from collections import Counter

AAMI = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q',
}

registros = wfdb.get_record_list('mitdb')
filas = []
for r in registros:
    ann = wfdb.rdann(r, 'atr', pn_dir='mitdb')
    # Nos quedamos solo con los símbolos que son latidos (ignora '+', '~', etc.)
    c = Counter(AAMI[s] for s in ann.symbol if s in AAMI)
    filas.append({'registro': r, **c})
    print(f"{r} ✓")

df = pd.DataFrame(filas).set_index('registro').fillna(0).astype(int)
df = df[['N', 'S', 'V', 'F', 'Q']]

total = df.sum()
print("\nTotal de latidos por clase:")
print(total)
print("\nPorcentaje:")
print((100 * total / total.sum()).round(2))

print("\nRegistros con más latidos V:")
print(df.sort_values('V', ascending=False).head(5))

df.to_csv('distribucion_clases.csv')