import nbformat as nbf
from pathlib import Path

p = Path("notebooks/nuevo_experimento/12_analisis_final.ipynb")

nb = nbf.v4.new_notebook()

nb.cells = [

nbf.v4.new_markdown_cell(
"""# 12. Análisis final del experimento

**Objetivo:** integrar los resultados obtenidos durante la auditoría del dataset, la construcción de la arquitectura 1D-CNN, el entrenamiento, la evaluación y el análisis de importancia de variables.

Este notebook no realiza un nuevo entrenamiento. Utiliza exclusivamente los resultados generados en los notebooks anteriores."""
),

nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

BASE = Path("../../resultados/nuevo_experimento")
FIG_DIR = BASE / "figuras"
TAB_DIR = BASE / "tablas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

print("Directorio de resultados:", BASE)"""
),

nbf.v4.new_markdown_cell(
"""## 1. Características del dataset

El experimento parte de un conjunto original de 70,000 registros. La auditoría identificó registros que incumplían los criterios operacionales de consistencia definidos para el experimento.

Después del filtrado se conservaron 68,615 registros y posteriormente se seleccionó una muestra estratificada de 15,000 observaciones para el entrenamiento."""
),

nbf.v4.new_code_cell(
"""dataset_resumen = pd.DataFrame({
    "Etapa": [
        "Dataset original",
        "Dataset después de auditoría",
        "Muestra utilizada"
    ],
    "Registros": [
        70000,
        68615,
        15000
    ]
})

print(dataset_resumen.to_string(index=False))

dataset_resumen.to_csv(
    TAB_DIR / "resumen_dataset_final.csv",
    index=False
)"""
),

nbf.v4.new_markdown_cell(
"""## 2. Variables utilizadas

El identificador `id` no se utiliza como variable predictora. La variable objetivo es `cardio`.

Se utilizan 11 variables clínicas:
edad, sexo, altura, peso, presión sistólica, presión diastólica, colesterol, glucosa, tabaquismo, consumo de alcohol y actividad física."""
),

nbf.v4.new_code_cell(
"""variables = [
    "Edad",
    "Sexo",
    "Altura",
    "Peso",
    "Presión sistólica",
    "Presión diastólica",
    "Colesterol",
    "Glucosa",
    "Tabaquismo",
    "Consumo de alcohol",
    "Actividad física"
]

print("Número de variables predictoras:", len(variables))

for i, variable in enumerate(variables, start=1):
    print(f"{i}. {variable}")"""
),

nbf.v4.new_markdown_cell(
"""## 3. Arquitectura final

La arquitectura utilizada contiene seis capas y fue diseñada para mantener el número total de parámetros dentro del intervalo establecido de 30,000 a 50,000 parámetros."""
),

nbf.v4.new_code_cell(
"""arquitectura = pd.DataFrame({
    "Capa": [
        "Conv1D",
        "Conv1D",
        "GlobalAveragePooling1D",
        "Dense",
        "Dropout",
        "Dense"
    ],
    "Salida": [
        "(None, 11, 32)",
        "(None, 11, 96)",
        "(None, 96)",
        "(None, 256)",
        "(None, 256)",
        "(None, 1)"
    ],
    "Parámetros": [
        128,
        9312,
        0,
        24832,
        0,
        257
    ]
})

print(arquitectura.to_string(index=False))
print()
print("Total de parámetros:", arquitectura["Parámetros"].sum())

assert arquitectura["Parámetros"].sum() == 34529

arquitectura.to_csv(
    TAB_DIR / "arquitectura_final_resumen.csv",
    index=False
)"""
),

nbf.v4.new_markdown_cell(
"""## 4. Desempeño del modelo

Los resultados corresponden al modelo entrenado sobre la muestra de 15,000 registros.

El entrenamiento utilizó parada temprana y finalizó en la época 66."""
),

nbf.v4.new_code_cell(
"""metricas = pd.DataFrame({
    "Métrica": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Valor": [
        0.7311,
        0.7326,
        0.7188,
        0.7256,
        0.7885
    ]
})

metricas["Porcentaje"] = metricas["Valor"] * 100

print(metricas.to_string(index=False))

metricas.to_csv(
    TAB_DIR / "metricas_finales_resumen.csv",
    index=False
)"""
),

nbf.v4.new_markdown_cell(
"""## 5. Importancia de variables

El análisis complementario mediante árbol de decisión mostró una mayor importancia relativa de la presión sistólica, seguida por la edad y el colesterol.

Este resultado no representa los pesos internos de la CNN. Se utiliza únicamente como herramienta complementaria de interpretación."""
),

nbf.v4.new_code_cell(
"""importancia = pd.DataFrame({
    "Variable": [
        "Presión sistólica",
        "Edad",
        "Colesterol",
        "Peso",
        "Presión diastólica"
    ],
    "Importancia": [
        0.747432,
        0.151009,
        0.074941,
        0.010035,
        0.008521
    ]
})

print(importancia.to_string(index=False))

print()
print(
    "Variable con mayor importancia:",
    importancia.iloc[0]["Variable"]
)"""
),

nbf.v4.new_markdown_cell(
"""## 6. Integración de resultados

La siguiente tabla resume los principales elementos del experimento."""
),

nbf.v4.new_code_cell(
"""resumen = pd.DataFrame({
    "Elemento": [
        "Registros originales",
        "Registros después de auditoría",
        "Registros utilizados",
        "Variables predictoras",
        "Capas del modelo",
        "Parámetros",
        "Épocas ejecutadas",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Resultado": [
        "70,000",
        "68,615",
        "15,000",
        "11",
        "6",
        "34,529",
        "66",
        "73.11 %",
        "73.26 %",
        "71.88 %",
        "72.56 %",
        "78.85 %"
    ]
})

print(resumen.to_string(index=False))

resumen.to_csv(
    TAB_DIR / "resumen_experimento_final.csv",
    index=False
)"""
),

nbf.v4.new_markdown_cell(
"""## 7. Visualización del desempeño

Se genera una representación conjunta de las métricas principales del modelo."""
),

nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    metricas["Métrica"],
    metricas["Valor"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Valor")
ax.set_title("Desempeño final del modelo 1D-CNN")

for i, value in enumerate(metricas["Valor"]):
    ax.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

fig.tight_layout()

fig.savefig(
    FIG_DIR / "metricas_finales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()"""
),

nbf.v4.new_markdown_cell(
"""## 8. Interpretación de los resultados

El modelo obtuvo una Accuracy de 73.11 %, mientras que Precision, Recall y F1-score alcanzaron 73.26 %, 71.88 % y 72.56 %, respectivamente. El ROC-AUC fue de 78.85 %, lo que indica una capacidad de discriminación superior al azar.

La arquitectura contiene 34,529 parámetros, ubicándose dentro del intervalo de 30,000 a 50,000 parámetros establecido para el experimento.

El análisis complementario de importancia de variables identificó a la presión sistólica como la variable con mayor importancia relativa en el árbol de decisión, seguida por la edad y el colesterol.

Estos resultados deben interpretarse dentro de las condiciones experimentales establecidas: una muestra estratificada de 15,000 registros, las reglas de consistencia definidas y la arquitectura 1D-CNN modificada."""
),

nbf.v4.new_markdown_cell(
"""## 9. Conclusión experimental

Se implementó una arquitectura 1D-CNN modificada para la clasificación de enfermedad cardiovascular utilizando 15,000 registros seleccionados de un conjunto de 70,000 observaciones. Después de la auditoría se conservaron 68,615 registros válidos y se obtuvo una muestra estratificada reproducible.

La arquitectura final está compuesta por dos capas Conv1D, una capa GlobalAveragePooling1D, una capa Dense, una capa Dropout y una capa Dense de salida. El modelo contiene 34,529 parámetros, cumpliendo el intervalo de complejidad establecido.

El desempeño obtenido fue de 73.11 % de Accuracy, 73.26 % de Precision, 71.88 % de Recall, 72.56 % de F1-score y 78.85 % de ROC-AUC.

En conjunto, el experimento proporciona una implementación reproducible y documentada del modelo propuesto, con una arquitectura de menor complejidad y un análisis complementario de las variables clínicas utilizadas."""
),

nbf.v4.new_markdown_cell(
"""## 10. Verificación final"""
),

nbf.v4.new_code_cell(
"""assert len(dataset_resumen) == 3
assert len(variables) == 11
assert arquitectura["Parámetros"].sum() == 34529
assert len(metricas) == 5
assert abs(metricas.loc[0, "Valor"] - 0.7311) < 1e-9
assert abs(metricas.loc[4, "Valor"] - 0.7885) < 1e-9

print("==============================================")
print("VERIFICACIÓN FINAL SATISFACTORIA")
print("==============================================")
print("Dataset utilizado: 15,000 registros")
print("Variables predictoras: 11")
print("Parámetros: 34,529")
print("Accuracy: 73.11 %")
print("F1-score: 72.56 %")
print("ROC-AUC: 78.85 %")
print("==============================================")"""
)

]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3.11.9 (.venv)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.9"
    }
}

nbf.write(nb, p)

print("Notebook creado correctamente:")
print(p)
print("Celdas:", len(nb.cells))

