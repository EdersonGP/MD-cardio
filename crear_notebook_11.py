import nbformat as nbf
from pathlib import Path

p = Path("notebooks/nuevo_experimento/11_arquitectura_modelo.ipynb")

nb = nbf.v4.new_notebook()

nb.cells = [

nbf.v4.new_markdown_cell(
"""# 11. Arquitectura del modelo 1D-CNN propuesto

**Objetivo:** documentar y representar la arquitectura utilizada en el experimento, mostrando sus capas, dimensiones, parámetros y flujo de procesamiento.

La arquitectura se analiza a partir del modelo entrenado previamente en el Notebook 09. No se realiza un nuevo entrenamiento."""
),

nbf.v4.new_code_cell(
"""import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

MODEL_PATH = Path("../../resultados/nuevo_experimento/modelo_nueva_cnn.keras")
OUT_DIR = Path("../../resultados/nuevo_experimento")
FIG_DIR = OUT_DIR / "figuras"
TAB_DIR = OUT_DIR / "tablas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

print("Modelo cargado correctamente.")
print("Número de capas:", len(model.layers))
print("Parámetros totales:", model.count_params())"""
),

nbf.v4.new_markdown_cell(
"""## 1. Resumen de la arquitectura

La siguiente información se obtiene directamente del modelo entrenado. De esta manera, la descripción de la arquitectura coincide con la implementación ejecutada."""
),

nbf.v4.new_code_cell(
"""model.summary()"""
),

nbf.v4.new_markdown_cell(
"""## 2. Parámetros por capa

Se extrae automáticamente el número de parámetros entrenables y no entrenables de cada capa."""
),

nbf.v4.new_code_cell(
"""rows = []

for layer in model.layers:
    rows.append({
        "capa": layer.name,
        "tipo": layer.__class__.__name__,
        "forma_salida": str(layer.output.shape),
        "parametros": layer.count_params(),
        "entrenables": sum(
            int(tf.size(w).numpy()) for w in layer.trainable_weights
        ),
        "no_entrenables": sum(
            int(tf.size(w).numpy()) for w in layer.non_trainable_weights
        )
    })

arquitectura = pd.DataFrame(rows)

print(arquitectura.to_string(index=False))

arquitectura.to_csv(
    TAB_DIR / "arquitectura_capas.csv",
    index=False
)"""
),

nbf.v4.new_markdown_cell(
"""## 3. Verificación del número de parámetros

El número total de parámetros debe coincidir con el valor obtenido durante el entrenamiento del Notebook 09."""
),

nbf.v4.new_code_cell(
"""TOTAL_ESPERADO = 34529
TOTAL_OBTENIDO = model.count_params()

print("Parámetros esperados:", TOTAL_ESPERADO)
print("Parámetros obtenidos:", TOTAL_OBTENIDO)

assert TOTAL_OBTENIDO == TOTAL_ESPERADO

print("Verificación satisfactoria: la arquitectura contiene 34,529 parámetros.")"""
),

nbf.v4.new_markdown_cell(
"""## 4. Flujo de procesamiento

El modelo recibe las variables clínicas como una secuencia de características y las procesa mediante las capas convolucionales antes de realizar la clasificación final.

El flujo general es:

**Variables clínicas → capas convolucionales → reducción de características → capa densa → salida Sigmoid → clasificación cardiovascular.**"""
),

nbf.v4.new_code_cell(
"""print("FLUJO DEL MODELO")
print("=" * 60)

for i, layer in enumerate(model.layers, start=1):
    print(
        f"{i}. {layer.name} | "
        f"{layer.__class__.__name__} | "
        f"salida={layer.output.shape} | "
        f"parámetros={layer.count_params()}"
    )"""
),

nbf.v4.new_markdown_cell(
"""## 5. Representación visual de la arquitectura

La siguiente figura resume visualmente el flujo del modelo. Los valores de parámetros se obtienen directamente de la arquitectura cargada."""
),

nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(15, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

layers = []

for layer in model.layers:
    layers.append({
        "nombre": layer.name,
        "tipo": layer.__class__.__name__,
        "params": layer.count_params(),
        "salida": str(layer.output.shape)
    })

x_positions = list(range(1, len(layers) + 1))

if len(x_positions) > 8:
    step = 8 / (len(x_positions) - 1)
    x_positions = [1 + i * step for i in range(len(layers))]

for i, (x, info) in enumerate(zip(x_positions, layers)):

    text = (
        f"{info['tipo']}\\n"
        f"{info['nombre']}\\n"
        f"Salida: {info['salida']}\\n"
        f"Parámetros: {info['params']:,}"
    )

    ax.text(
        x,
        5,
        text,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.6",
            fill=False,
            linewidth=1.5
        )
    )

    if i < len(layers) - 1:
        ax.annotate(
            "",
            xy=(x_positions[i+1] - 0.45, 5),
            xytext=(x + 0.45, 5),
            arrowprops=dict(arrowstyle="->", linewidth=1.5)
        )

ax.text(
    5,
    8.5,
    "Arquitectura propuesta de la 1D-CNN",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

ax.text(
    5,
    1.3,
    f"Total de parámetros: {model.count_params():,}",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "arquitectura_nueva_cnn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()"""
),

nbf.v4.new_markdown_cell(
"""## 6. Verificaciones finales

Se comprueba que el modelo pueda cargarse correctamente, que el número de parámetros coincida con el entrenamiento y que la tabla de arquitectura contenga todas las capas."""
),

nbf.v4.new_code_cell(
"""print("Número de capas:", len(model.layers))
print("Parámetros totales:", model.count_params())
print("Filas de la tabla de arquitectura:", len(arquitectura))

assert model.count_params() == 34529
assert len(arquitectura) == len(model.layers)
assert arquitectura["parametros"].sum() == 34529

print()
print("Todas las verificaciones fueron satisfactorias.")"""
),

nbf.v4.new_markdown_cell(
"""## 7. Conclusión

La arquitectura analizada corresponde exactamente al modelo utilizado en el experimento. La extracción automática permite documentar las capas, dimensiones y parámetros sin introducir modificaciones respecto al modelo entrenado.

El modelo contiene **34,529 parámetros**, valor que se encuentra dentro del rango de 30,000 a 50,000 parámetros establecido para el experimento.

Esta arquitectura constituye la implementación modificada respecto al modelo de referencia y será utilizada para el análisis final de desempeño."""
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

