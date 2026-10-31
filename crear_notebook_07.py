import nbformat as nbf
from pathlib import Path

p = Path("notebooks/nuevo_experimento/07_auditoria_dataset.ipynb")

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# 07. Auditoría y preparación del dataset cardiovascular

**Objetivo:** auditar el conjunto de datos original, establecer criterios de consistencia y preparar una muestra reproducible para el entrenamiento posterior."""),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("../../data/nuevo_dataset/cardio_train.csv")
OUT_DIR = Path("../../resultados/nuevo_experimento")
FIG_DIR = OUT_DIR / "figuras"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

print("Dimensiones:", df.shape)
display(df.head())"""),

    nbf.v4.new_markdown_cell("""## 1. Auditoría inicial

Se verifica la estructura del dataset, valores faltantes, duplicados y distribución de la variable objetivo."""),
    
    nbf.v4.new_code_cell("""print("Registros:", len(df))
print("Columnas:", len(df.columns))

print("\\nColumnas:")
print(df.columns.tolist())

print("\\nValores nulos:")
print(df.isnull().sum())
print("Total de nulos:", df.isnull().sum().sum())

print("\\nDuplicados exactos:", df.duplicated().sum())

print("\\nTipos de datos:")
print(df.dtypes)

print("\\nDistribución de cardio:")
print(df["cardio"].value_counts().sort_index())

print("\\nPorcentajes:")
print((df["cardio"].value_counts(normalize=True).sort_index() * 100).round(2))"""),

    nbf.v4.new_markdown_cell("""## 2. Auditoría de valores clínicos

Se identifican valores potencialmente inconsistentes en edad, altura, peso y presión arterial. Los criterios utilizados son reglas operacionales para este experimento y no representan criterios médicos universales."""),
    
    nbf.v4.new_code_cell("""age_years = df["age"] / 365.25

rules = {
    "edad": ~age_years.between(18, 100),
    "altura": ~df["height"].between(120, 220),
    "peso": ~df["weight"].between(30, 250),
    "ap_hi": ~df["ap_hi"].between(70, 250),
    "ap_lo": ~df["ap_lo"].between(40, 200),
    "presion_relacion": ~(df["ap_hi"] > df["ap_lo"])
}

rules_df = pd.DataFrame(rules)

print("REGISTROS QUE INCUMPLEN CADA CRITERIO")
print("-" * 45)

for nombre, regla in rules.items():
    print(f"{nombre}: {int(regla.sum())}")

invalid = rules_df.any(axis=1)

print()
print("Incumplen al menos un criterio:", int(invalid.sum()))
print("Incumplen dos o más:", int(rules_df.sum(axis=1).ge(2).sum()))"""),

    nbf.v4.new_markdown_cell("""## 3. Conjunto limpio

Se genera una copia del dataset excluyendo los registros que incumplen al menos uno de los criterios definidos. El archivo original permanece sin modificaciones."""),
    
    nbf.v4.new_code_cell("""df_clean = df.loc[~invalid].copy()

print("Registros originales:", len(df))
print("Registros válidos:", len(df_clean))
print("Registros eliminados:", len(df) - len(df_clean))
print("Porcentaje conservado:", round(len(df_clean) / len(df) * 100, 2), "%")

print("\\nDistribución de cardio:")
print(df_clean["cardio"].value_counts().sort_index())

clean_path = OUT_DIR / "cardio_limpio.csv"

df_clean.to_csv(
    clean_path,
    index=False,
    sep=";"
)

print("\\nArchivo generado:")
print(clean_path)"""),

    nbf.v4.new_markdown_cell("""## 4. Selección de 15 000 registros

Se seleccionará una muestra estratificada respecto de `cardio`, utilizando `random_state=42` para garantizar la reproducibilidad."""),
    
    nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

sample_15000, _ = train_test_split(
    df_clean,
    train_size=15000,
    random_state=42,
    stratify=df_clean["cardio"]
)

sample_15000 = sample_15000.reset_index(drop=True)

print("Registros seleccionados:", len(sample_15000))

print("\\nDistribución:")
print(sample_15000["cardio"].value_counts().sort_index())

print("\\nPorcentajes:")
print(
    (sample_15000["cardio"].value_counts(normalize=True).sort_index() * 100).round(2)
)

sample_path = OUT_DIR / "cardio_15000.csv"

sample_15000.to_csv(
    sample_path,
    index=False,
    sep=";"
)

print("\\nArchivo generado:")
print(sample_path)"""),

    nbf.v4.new_markdown_cell("""## 5. Variables para el modelo

`id` no se utilizará como predictor. `cardio` corresponde a la variable objetivo. La edad original está expresada en días y se transformará a años para el modelado."""),
    
    nbf.v4.new_code_cell("""features = [
    c for c in sample_15000.columns
    if c not in ["id", "cardio"]
]

print("Variables predictoras:", len(features))
print(features)

model_df = sample_15000.copy()
model_df["age"] = model_df["age"] / 365.25

print("\\nRango de edad:")
print(
    round(model_df["age"].min(), 2),
    "a",
    round(model_df["age"].max(), 2),
    "años"
)"""),

    nbf.v4.new_markdown_cell("""## 6. Gráficos de auditoría"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()

counts = df["cardio"].value_counts().sort_index()

ax.bar(["0", "1"], counts.values)

ax.set_title("Distribución de cardio — dataset original")
ax.set_xlabel("Clase")
ax.set_ylabel("Registros")

fig.tight_layout()
fig.savefig(FIG_DIR / "cardio_original.png", dpi=300)

plt.show()"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()

counts = df_clean["cardio"].value_counts().sort_index()

ax.bar(["0", "1"], counts.values)

ax.set_title("Distribución de cardio — dataset limpio")
ax.set_xlabel("Clase")
ax.set_ylabel("Registros")

fig.tight_layout()
fig.savefig(FIG_DIR / "cardio_limpio.png", dpi=300)

plt.show()"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()

counts = sample_15000["cardio"].value_counts().sort_index()

ax.bar(["0", "1"], counts.values)

ax.set_title("Distribución de cardio — muestra de 15 000")
ax.set_xlabel("Clase")
ax.set_ylabel("Registros")

fig.tight_layout()
fig.savefig(FIG_DIR / "cardio_15000.png", dpi=300)

plt.show()"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()

ax.hist(model_df["age"], bins=30)

ax.set_title("Distribución de edad")
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Frecuencia")

fig.tight_layout()
fig.savefig(FIG_DIR / "distribucion_edad.png", dpi=300)

plt.show()"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()

ax.hist(
    model_df["ap_hi"],
    bins=30,
    alpha=0.7,
    label="ap_hi"
)

ax.hist(
    model_df["ap_lo"],
    bins=30,
    alpha=0.7,
    label="ap_lo"
)

ax.set_title("Distribución de presión arterial")
ax.set_xlabel("Presión")
ax.set_ylabel("Frecuencia")

ax.legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "distribucion_presion.png", dpi=300)

plt.show()"""),

    nbf.v4.new_markdown_cell("""## 7. Verificaciones finales"""),
    
    nbf.v4.new_code_cell("""print("cardio_limpio:", len(df_clean))
print("cardio_15000:", len(sample_15000))

print("Nulos en limpio:", int(df_clean.isnull().sum().sum()))
print("Duplicados en limpio:", int(df_clean.duplicated().sum()))

print("Nulos en muestra:", int(sample_15000.isnull().sum().sum()))
print("Duplicados en muestra:", int(sample_15000.duplicated().sum()))

assert len(sample_15000) == 15000
assert df_clean.isnull().sum().sum() == 0
assert sample_15000.isnull().sum().sum() == 0

print("\\nTodas las verificaciones principales fueron satisfactorias.")"""),

    nbf.v4.new_markdown_cell("""## 8. Conclusión

La auditoría permite caracterizar el dataset cardiovascular y establecer un procedimiento reproducible de preparación. El conjunto resultante será utilizado en las siguientes etapas para desarrollar y evaluar una arquitectura 1D-CNN modificada, con un número de parámetros reducido respecto del modelo de referencia.""")
]

nbf.write(nb, p)

print("Notebook creado correctamente:")
print(p)
print("Número de celdas:", len(nb.cells))
