"""Reorganización documental única. No importa ni entrena modelos.

Ejecutar desde cualquier directorio con Python >= 3.11. Conserva respaldos
byte a byte y crea una secuencia nueva sin ejecutar sus celdas.
"""
from pathlib import Path
import hashlib
import json
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
DEST = NB / "nuevo_experimento"
BACKUP = NB / "respaldo_nuevo_experimento_original"
HIST = NB / "historico_dataset_1025"
MANIFEST = ROOT / "docs" / "respaldo_reorganizacion.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": textwrap.dedent(source).strip()}


SETUP = '''
from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "data/nuevo_dataset/cardio_train.csv").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from src.experimento import (
    ORIGINAL, RESULTADOS, VARIABLES, NOMBRES, cargar_datos, reglas_calidad,
    cargar_metricas, comparar_ejecuciones, tabla_configuracion, verificar_configuracion,
    construir_cnn, tabla_modelo, dibujar_arquitectura, guardar_figura_nueva,
    importancia_guardada, resumen_dataset, verificar_predicciones, tensorflow_disponible,
)
'''


def notebook(name, title, intro, cells):
    if name.startswith("11_"):
        for cell in cells:
            if cell["cell_type"] == "code" and cell["source"].startswith("import tensorflow as tf"):
                cell["source"] = ('import importlib.util\n'
                    'if not tensorflow_disponible():\n'
                    '    print("TensorFlow no disponible: verificación de configuración serializada realizada; carga de pesos con Keras pendiente en este kernel.")\n'
                    '    fig = dibujar_arquitectura(configuracion)\n'
                    '    print(guardar_figura_nueva(fig, "arquitectura_modelo_guardado_verificada.png"))\n'
                    '    plt.show()\n'
                    'else:\n' + textwrap.indent(cell["source"], '    '))
    contents = [md(f"# {title}\n\n{intro}\n\n"
                   "**Dataset principal:** `data/nuevo_dataset/cardio_train.csv`. "
                   "Este cuaderno no ejecuta entrenamientos. Los resultados históricos "
                   "se leen de archivos conservados; no se sustituyen por nuevas ejecuciones."), code(SETUP), *cells]
    for i, cell in enumerate(contents):
        cell["id"] = f"cardio-{name[:2]}-{i:03d}"
    out = {"cells": contents, "metadata": {
        "kernelspec": {"display_name": "Python 3 (Cardio)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "cardio": {"dataset": "cardio_train.csv", "entrenamiento": False,
                   "resultados": "artefactos existentes", "reorganizacion_documental": True}},
        "nbformat": 4, "nbformat_minor": 5}
    (DEST / f"{name}.ipynb").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    if MANIFEST.exists():
        raise SystemExit("La reorganización ya tiene manifiesto. No se sobrescriben respaldos ni notebooks.")
    old = sorted(NB.glob("*.ipynb"))
    current = sorted(DEST.glob("*.ipynb"))
    assert len(old) == 6 and len(current) == 6, "Inventario inesperado: revisar antes de continuar."
    preserved = sorted(p for base in [ROOT / "data", ROOT / "resultados"] for p in base.rglob("*") if p.is_file())
    preserved += sorted(ROOT.glob("crear_notebook_*.py"))
    snapshot = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in preserved}
    backups = {}
    # Todas las copias se completan y verifican ANTES del primer movimiento.
    for sources, target in [(old, HIST / "respaldo_original"), (current, BACKUP)]:
        target.mkdir(parents=True, exist_ok=True)
        for p in sources:
            q = target / p.name
            if q.exists():
                raise RuntimeError(f"Respaldo ya existente: {q}")
            shutil.copy2(p, q)
            assert sha(p) == sha(q)
            backups[str(p.relative_to(ROOT)).replace("\\", "/")] = {
                "respaldo": str(q.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(q)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"descripcion": "Respaldo previo a cualquier movimiento; sin entrenamiento",
                "archivos_conservados": snapshot, "notebooks_originales": backups}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for p in old:
        shutil.move(str(p), str(HIST / p.name))
    # Los nombres que no pertenecen a la secuencia aprobada se reubican, no se eliminan.
    relocated = BACKUP / "reubicados"
    relocated.mkdir(exist_ok=True)
    for p in current:
        if p.name in {"07_auditoria_dataset.ipynb", "09_evaluacion_y_comparacion.ipynb", "11_arquitectura_modelo.ipynb"}:
            shutil.move(str(p), str(relocated / p.name))

    notebook("01_exploracion_dataset", "01. Exploración del dataset cardiovascular", "Caracterizar el conjunto original y explicar sus variables.", [
        code('df = cargar_datos(ORIGINAL)\nprint("Dimensiones:", df.shape)\ndisplay(df.head())\ndisplay(pd.DataFrame({"tipo": df.dtypes.astype(str), "nulos": df.isna().sum()}))\nprint("Columnas:", df.columns.tolist())\nprint("Nulos:", int(df.isna().sum().sum()))\nprint("Duplicados exactos (incluye id):", int(df.duplicated().sum()))\nprint("Repeticiones sin id:", int(df.drop(columns="id").duplicated().sum()))'),
        md("## Diccionario de variables\nLos códigos categóricos son categorías, no magnitudes continuas. La edad original está en días. `id` identifica registros y `cardio` es el objetivo; quedan 11 predictoras. Coincidencias sin identificador no demuestran por sí solas que sean el mismo paciente."),
        code('descripciones = {"id": "Identificador; excluido del modelo", "age": "Edad en días; se convierte a años dividiendo entre 365.25", "gender": "Sexo codificado (1/2)", "height": "Altura en centímetros", "weight": "Peso en kilogramos", "ap_hi": "Presión arterial sistólica", "ap_lo": "Presión arterial diastólica", "cholesterol": "Colesterol: categorías 1 normal, 2 superior, 3 muy superior", "gluc": "Glucosa: categorías 1 normal, 2 superior, 3 muy superior", "smoke": "Tabaquismo: 0/1", "alco": "Consumo de alcohol: 0/1", "active": "Actividad física: 0/1", "cardio": "Objetivo: 0 ausencia, 1 presencia de enfermedad cardiovascular"}\ndisplay(pd.DataFrame({"variable": list(descripciones), "descripcion": list(descripciones.values())}))\ndisplay(df.describe().T)\ndisplay(df["cardio"].value_counts().sort_index().to_frame("registros"))\ndisplay(df["cardio"].value_counts(normalize=True).sort_index().mul(100).to_frame("porcentaje"))'),
        code('fig, axes = plt.subplots(2, 3, figsize=(13, 7))\nexplorar = df.assign(age=df.age / 365.25)\nfor ax, col in zip(axes.flat, ["age", "height", "weight", "ap_hi", "ap_lo", "cardio"]):\n    ax.hist(explorar[col], bins=30)\n    ax.set_title(NOMBRES.get(col, col))\nfig.suptitle("Dataset original: incluye valores extremos, sin filtrado")\nfig.tight_layout()\nplt.show()'),
        md("Los valores extremos se mantienen visibles en esta exploración. Su identificación y tratamiento se documentan en 02 y 03; el original permanece intacto.")])

    notebook("02_auditoria_calidad", "02. Auditoría de calidad", "Identificar problemas sin modificar el dataset original.", [
        code('df = cargar_datos(ORIGINAL)\nreglas = reglas_calidad(df)\ndisplay(df.isna().sum().to_frame("nulos"))\nprint("Duplicados exactos:", int(df.duplicated().sum()))\ndisplay(df[["age", "height", "weight", "ap_hi", "ap_lo"]].describe(percentiles=[.01,.5,.99]).T)\ndisplay(reglas.sum().rename("incumplen").to_frame())\ninvalidos = reglas.any(axis=1)\nprint("Incumplen dos o más criterios:", int(reglas.sum(axis=1).ge(2).sum()))\nprint("Total excluido por unión de criterios:", int(invalidos.sum()))\nprint("Total conservado:", int((~invalidos).sum()))\nassert int(invalidos.sum()) == 1385\nassert int((~invalidos).sum()) == 68615'),
        md("## Criterios operacionales\nSe conservan edades entre 18 y 100 años, alturas entre 120 y 220 cm, pesos entre 30 y 250 kg, presión sistólica entre 70 y 250 y diastólica entre 40 y 200; además, sistólica > diastólica. Los límites son inclusivos salvo la relación estricta. **Son criterios operacionales del experimento, no criterios médicos universales.** Un registro que incumple varias reglas se cuenta una sola vez en el total excluido. La ausencia de nulos y duplicados exactos no garantiza consistencia."),
        code('display(df.loc[invalidos].head(10))')])

    notebook("03_limpieza_preprocesamiento", "03. Limpieza y preprocesamiento", "Verificar la limpieza ya utilizada y conservar su archivo derivado.", [
        code('original = cargar_datos(ORIGINAL)\nlimpio_esperado = original.loc[~reglas_calidad(original).any(axis=1)].reset_index(drop=True)\nruta = RESULTADOS / "cardio_limpio.csv"\nif ruta.exists():\n    limpio = cargar_datos(ruta)\n    pd.testing.assert_frame_equal(limpio, limpio_esperado, check_dtype=False)\n    print("Dataset limpio existente verificado; no se sobrescribe.")\nelse:\n    limpio_esperado.to_csv(ruta, sep=";", index=False)\n    limpio = limpio_esperado\nassert len(original) == 70000 and len(limpio) == 68615\nprint(f"{len(original):,} → {len(limpio):,}; excluidos: {len(original)-len(limpio):,}")'),
        md("## Representación para modelado\nEl CSV limpio conserva las 13 columnas originales y la edad en días para mantener trazabilidad. La conversión se aplica a una copia en memoria. `id` se conserva para rastrear registros, pero se excluye de X. `cardio` se separa como y. El escalador no se ajusta aquí: debe ajustarse después de separar entrenamiento y evaluación."),
        code('modelo = limpio.copy()\nmodelo["age"] = modelo["age"] / 365.25\nX = modelo[VARIABLES].copy()\ny = modelo["cardio"].copy()\nassert X.shape[1] == 11\nassert "id" not in X and "cardio" not in X\nprint("Predictoras:", X.columns.tolist())\nprint("Objetivo:", y.name)\ndisplay(X.describe().T)')])

    notebook("04_preparacion_datos", "04. Preparación de datos", "Usar la muestra existente y distinguir protocolos ejecutados de un procedimiento recomendado sin fuga.", [
        code('muestra = cargar_datos(RESULTADOS / "cardio_15000.csv")\nlimpio = cargar_datos(RESULTADOS / "cardio_limpio.csv")\nassert len(muestra) == 15000 and muestra.id.is_unique\nassert set(muestra.id).issubset(set(limpio.id))\npd.testing.assert_frame_equal(muestra.set_index("id").sort_index(), limpio.set_index("id").loc[muestra.id].sort_index(), check_dtype=False)\ndisplay(muestra.cardio.value_counts().sort_index().to_frame("registros"))\ndata = muestra.copy()\ndata["age"] /= 365.25\nX, y = data[VARIABLES], data.cardio\nprint("X:", X.shape, "y:", y.shape)\nassert len(VARIABLES) == 11'),
        md("## Protocolos que produjeron los resultados existentes\nLa muestra fue obtenida por selección estratificada con semilla 42. **No se vuelve a muestrear.** La ejecución inicial utilizó StandardScaler y particiones explícitas 70/15/15. La definitiva separó 15 % de prueba, ajustó MinMaxScaler sobre el 85 % restante y reservó luego 20 % de ese bloque con `validation_split`. Esto implica 68/17/15 efectivos. La validación influyó en el escalamiento de la ejecución definitiva; la prueba no. No se atribuyen las métricas guardadas al procedimiento recomendado siguiente."),
        code('display(comparar_ejecuciones())'),
        md("## Procedimiento recomendado sin fuga: solo preparación, sin entrenamiento\nSe separan primero entrenamiento, validación y prueba, con estratificación y semilla fija. El escalador se ajusta exclusivamente al entrenamiento. Este ejemplo usa 70/15/15 y MinMaxScaler; **no es una nueva ejecución evaluada ni reproduce las métricas definitivas**. La auditoría entre particiones compara características sin `id`, ya que un identificador único oculta coincidencias de contenido."),
        code('from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import MinMaxScaler\nX_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, stratify=y, random_state=42)\nX_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=42)\nassert set(X_train.index).isdisjoint(X_val.index)\nassert set(X_train.index).isdisjoint(X_test.index)\nassert set(X_val.index).isdisjoint(X_test.index)\nscaler = MinMaxScaler().fit(X_train)\nX_train_cnn = scaler.transform(X_train).astype("float32")[..., np.newaxis]\nX_val_cnn = scaler.transform(X_val).astype("float32")[..., np.newaxis]\nX_test_cnn = scaler.transform(X_test).astype("float32")[..., np.newaxis]\nassert scaler.n_samples_seen_ == len(X_train)\nfor bloque in [X_train_cnn, X_val_cnn, X_test_cnn]:\n    assert bloque.shape[1:] == (11, 1)\nprint("Formas:", X_train_cnn.shape, X_val_cnn.shape, X_test_cnn.shape)\nfor nombre, a, b in [("train/val", X_train, X_val), ("train/test", X_train, X_test), ("val/test", X_val, X_test)]:\n    print(nombre, "patrones predictores compartidos:", len(set(map(tuple, a.to_numpy())) & set(map(tuple, b.to_numpy()))))'),
        md("Si hay patrones coincidentes entre particiones, deben informarse; no implican automáticamente identidad de pacientes. Cambiar a una partición por grupos sería otro protocolo y exigiría resultados nuevos. Las categorías numéricas y el orden fijo de las variables se mantienen para respetar la arquitectura validada.")])

    architecture = '''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout
model = Sequential([
    Input(shape=(11, 1)),
    Conv1D(32, kernel_size=3, padding="same", activation="relu"),
    Conv1D(96, kernel_size=3, padding="same", activation="relu"),
    GlobalAveragePooling1D(),
    Dense(256, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid"),
])
model.summary()
assert model.count_params() == 34529
assert 30000 <= model.count_params() <= 50000
display(tabla_modelo(model))
'''
    architecture = ('import importlib.util\n'
                    'if not tensorflow_disponible():\n'
                    '    print("TensorFlow no está disponible en este kernel. Se verifica la configuración del modelo guardado; no se construye ni se ejecuta model.summary().")\n'
                    '    tabla = verificar_configuracion()\n'
                    '    display(tabla)\n'
                    'else:\n' + textwrap.indent(textwrap.dedent(architecture).strip(), '    ')
                    + '\n    tabla = tabla_modelo(model)')
    notebook("05_diseno_entrenamiento_cnn", "05. Diseño y documentación del entrenamiento CNN", "TensorFlow/Keras construye explícitamente la CNN. Crear una arquitectura no ajusta sus pesos.", [
        code(architecture),
        md("## Configuración de entrenamiento ya ejecutada\nAdam, entropía cruzada binaria, batch size 32, máximo 100 épocas y EarlyStopping con restauración de mejores pesos. La ejecución inicial usó paciencia 15 y terminó tras 44 épocas; la definitiva usó paciencia 10 y terminó tras 66. Los pesos restaurados no tienen por qué ser los de la última época. **No se llama a `model.fit` en esta secuencia.**"),
        code('display(comparar_ejecuciones())\nfor nombre in ["historial_nueva_cnn.csv", "historial_entrenamiento.csv"]:\n    historial = pd.read_csv(RESULTADOS / "tablas" / nombre)\n    print(nombre, "épocas:", len(historial), "mejor val_loss: época", int(np.argmin(historial.val_loss)) + 1)\n    display(historial.tail(3))'),
        md("El modelo construido arriba tiene pesos recién inicializados y se usa solo para demostrar la arquitectura. El modelo entrenado que respalda las métricas es `resultados/nuevo_experimento/modelo_nueva_cnn.keras`.")])

    notebook("06_evaluacion_inicial", "06. Evaluación inicial y evidencia conservada", "Leer la evaluación inicial y mostrar separadamente la evidencia definitiva, sin mezclar sus archivos.", [
        code('display(comparar_ejecuciones())\nprint("Métricas iniciales:")\ndisplay(cargar_metricas("metricas_nueva_cnn.csv"))\nprint("Métricas definitivas:")\ndisplay(cargar_metricas("metricas_evaluacion.csv"))'),
        md("## Ejecución inicial: StandardScaler\nLas figuras con sufijo `_nueva_cnn` pertenecen a esta ejecución. Su reporte de clasificación permanece en el notebook original respaldado; no se guardó un archivo de predicciones independiente para reconstruirlo."),
        code('for nombre in ["matriz_confusion_nueva_cnn.png", "curva_roc_nueva_cnn.png", "curva_accuracy_nueva_cnn.png", "curva_loss_nueva_cnn.png"]:\n    display(Image(filename=str(RESULTADOS / "figuras" / nombre)))'),
        md("## Ejecución definitiva: MinMaxScaler\nLas figuras sin ese sufijo, el reporte de clasificación y los tres archivos de predicciones corresponden a la ejecución definitiva. Se recalculan únicamente las métricas desde las predicciones guardadas para comprobar su coherencia, sin inferencia ni entrenamiento."),
        code('metricas, matriz = verificar_predicciones()\ndisplay(metricas)\ndisplay(pd.DataFrame(matriz, index=["Real 0", "Real 1"], columns=["Predicho 0", "Predicho 1"]))\nprint((RESULTADOS / "classification_report.txt").read_text(encoding="utf-8"))\nfor nombre in ["matriz_confusion.png", "curva_roc.png", "curva_accuracy.png", "curva_loss.png"]:\n    display(Image(filename=str(RESULTADOS / "figuras" / nombre)))'),
        md("Las métricas de precision, recall y F1 corresponden a `cardio=1`, con umbral 0.5. ROC-AUC utiliza probabilidades, no clases predichas.")])

    notebook("07_auditoria_experimental", "07. Auditoría experimental", "La numeración organiza la exposición metodológica, no inventa una cronología nueva de experimentos.", [
        code('display(comparar_ejecuciones())'),
        md("## Misma arquitectura, protocolos diferentes\nLos antiguos notebooks 08 y 09 usaron exactamente la misma arquitectura de 34,529 parámetros. StandardScaler frente a MinMaxScaler, la selección de particiones, la reserva de validación, las métricas de compilación y la paciencia de parada cambiaron. Los 44 y 66 son épocas ejecutadas, no necesariamente las de los pesos restaurados. La comparación no aísla una modificación arquitectónica ni el efecto causal del escalador."),
        md("## Fuga de información y reproducibilidad\nEn la ejecución inicial StandardScaler se ajustó solo sobre entrenamiento. En la definitiva MinMaxScaler se ajustó sobre el bloque de 12,750 registros antes de reservar la validación interna. Esa validación no fue estratificada explícitamente. La prueba de 2,250 quedó fuera del ajuste del escalador. El notebook 04 describe un procedimiento futuro separado y sin esa influencia de validación, sin adjudicarle las métricas existentes.\n\nNo se guardaron el escalador definitivo ni identificadores explícitos de las particiones. La muestra y el código original están conservados; no se afirma una reproducibilidad bit a bit de un nuevo entrenamiento. Las coincidencias sin `id` requieren revisión específica y no se resuelven eliminando registros retrospectivamente."),
        md("## Alcance de las comparaciones\nLas pruebas de ambas ejecuciones se generaron con procedimientos distintos; no basta con que tengan igual tamaño. Solo hay una ejecución por protocolo y no se estimó variabilidad entre semillas ni intervalos de confianza. Los resultados del dataset histórico no son una referencia numérica comparable con el dataset definitivo. La importancia del árbol usa otra partición 80/20 y no es una atribución de la CNN."),
        code('display(verificar_configuracion())')])

    notebook("08_nueva_arquitectura_cnn", "08. Arquitectura definitiva de la 1D-CNN", "Modificación respecto del modelo histórico del artículo; se mantiene idéntica entre las dos ejecuciones del nuevo dataset.", [
        code(architecture),
        md("## Flujo paso a paso\nEntrada `(11,1)`\n↓\nConv1D: 32 filtros, kernel 3, ReLU, same → `(11,32)`\n↓\nConv1D: 96 filtros, kernel 3, ReLU, same → `(11,96)`\n↓\nGlobalAveragePooling1D → `(96,)`\n↓\nDense 256, ReLU → `(256,)`\n↓\nDropout 0.30 → `(256,)`\n↓\nDense 1, Sigmoid → probabilidad de `cardio=1`.\n\nLas convoluciones combinan patrones locales en el orden fijo de las 11 variables. No se trata de una serie temporal: ese orden es una decisión del modelado y una limitación que debe explicarse. GlobalAveragePooling promedia sobre las posiciones y evita una capa Flatten de gran dimensión. Dropout no añade parámetros."),
        code('display(tabla)\nfig = dibujar_arquitectura(tabla)\nruta = guardar_figura_nueva(fig, "arquitectura_definitiva_desde_codigo.png")\nprint("Diagrama obtenido de las capas del modelo:", ruta)\nplt.show()'),
        md("## Cálculo de parámetros y requisito del docente\nConv1D(32): `(3 × 1 + 1) × 32 = 128`.\n\nConv1D(96): `(3 × 32 + 1) × 96 = 9,312`.\n\nDense(256): `(96 + 1) × 256 = 24,832`.\n\nDense(1): `(256 + 1) × 1 = 257`.\n\nGlobalAveragePooling1D y Dropout: 0. Total: **34,529**, dentro de **30,000–50,000**. La reducción espacial mediante promedio global permite controlar la entrada de la capa densa y cumplir ese presupuesto. La referencia histórica usaba cuatro convoluciones, normalización por lotes y Flatten; no se reconstruye ni se entrena aquí. Cumplir el rango de parámetros no garantiza por sí mismo mejor rendimiento."),
        code('guardado = verificar_configuracion()\nassert tabla.parametros.sum() == guardado.parametros.sum() == 34529\nprint("Arquitectura construida y configuración guardada: 34,529 parámetros.")')])

    notebook("09_evaluacion_comparacion", "09. Evaluación y comparación de protocolos", "Comparar exclusivamente los resultados existentes de StandardScaler y MinMaxScaler.", [
        code('comparacion = comparar_ejecuciones()\ndisplay(comparacion)\nfig, ax = plt.subplots(figsize=(10, 5))\ncomparacion.set_index("Protocolo")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].T.plot.bar(ax=ax)\nax.set_ylim(0, 1)\nax.set_ylabel("Valor")\nax.set_title("Resultados existentes: pruebas generadas con procedimientos diferentes")\nfig.tight_layout()\nplt.show()'),
        md("Ambas ejecuciones tienen 34,529 parámetros. Cambian escalador, particiones y validación, paciencia y número de épocas. La diferencia de resultados es descriptiva: no demuestra que MinMaxScaler sea superior por sí solo ni que cambió la arquitectura. No se efectuó una comparación controlada con un mismo test, varias semillas o intervalos de confianza. La limitación del escalamiento previo a la reserva de validación se conserva explícitamente."),
        code('verificadas, matriz = verificar_predicciones()\ndisplay(verificadas)\nprint("Matriz definitiva:", matriz.tolist())')])

    notebook("10_importancia_variables", "10. Importancia de variables: análisis complementario", "Conservar el análisis ya calculado mediante DecisionTreeClassifier y explicar su procedencia sin volver a ajustar modelos.", [
        md("## Procedimiento que originó la tabla\nSe usó la misma muestra de 15,000, edad en años, exclusión de `id` y `cardio`, división estratificada 80/20 con semilla 42 y `DecisionTreeClassifier(max_depth=5, random_state=42)`. La siguiente función conserva el código de cálculo como documentación reproducible. **No se invoca**: ejecutar este notebook solo lee la importancia existente. El árbol no es un sustituto entrenado para explicar las predicciones de la CNN."),
        code('from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import train_test_split\n\ndef procedimiento_original_importancia_no_ejecutado():\n    """Solo referencia del cálculo original; no llamar durante esta reorganización."""\n    data = cargar_datos(RESULTADOS / "cardio_15000.csv").copy()\n    data["age"] /= 365.25\n    X, y = data[VARIABLES], data.cardio\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)\n    tree = DecisionTreeClassifier(max_depth=5, random_state=42)\n    tree.fit(X_train, y_train)\n    return pd.DataFrame({"variable": X.columns, "importancia": tree.feature_importances_}).sort_values("importancia", ascending=False)\n'),
        code('importancia = importancia_guardada()\ndisplay(importancia)\nassert len(importancia) == 11\nassert np.isclose(importancia.importancia.sum(), 1)\nfig, ax = plt.subplots(figsize=(10, 6))\norden = importancia.sort_values("importancia")\nax.barh(orden.variable_es, orden.importancia)\nax.set_xlabel("Importancia por reducción de impureza del árbol")\nax.set_title("Importancia complementaria: DecisionTreeClassifier, no atribución de la CNN")\nfig.tight_layout()\nprint(guardar_figura_nueva(fig, "importancia_variables_desde_csv.png"))\nplt.show()'),
        md("Las importancias son del árbol y dependen de su partición y profundidad. No son pesos internos, explicaciones causales ni atribuciones de la CNN. Un valor cero no demuestra irrelevancia clínica. La tabla conserva las variables originales y añade nombres en español para su interpretación.")])

    notebook("11_verificacion_arquitectura", "11. Verificación automática del modelo guardado", "Comprobar la configuración serializada y contrastarla con el modelo cargado mediante TensorFlow/Keras.", [
        code('configuracion = verificar_configuracion()\ndisplay(configuracion)\nassert len(configuracion) == 6\nassert configuracion.parametros.sum() == 34529'),
        code('import tensorflow as tf\nmodel = tf.keras.models.load_model(RESULTADOS / "modelo_nueva_cnn.keras", compile=False)\nmodel.summary()\nassert len(model.layers) == 6\nassert tuple(model.input_shape[1:]) == (11, 1)\nassert model.count_params() == 34529\nobtenida = tabla_modelo(model)\ndisplay(obtenida)\nassert obtenida.tipo.tolist() == configuracion.tipo.tolist()\nassert obtenida.parametros.tolist() == configuracion.parametros.tolist()\nfor capa, esperada in zip(model.layers, configuracion.to_dict("records")):\n    assert tuple(capa.output.shape[1:]) == tuple(esperada["salida"] )\nassert model.layers[0].filters == 32 and model.layers[1].filters == 96\nfor capa in model.layers[:2]:\n    assert capa.kernel_size == (3,) and capa.padding == "same"\n    assert capa.activation.__name__ == "relu"\nassert model.layers[3].units == 256\nassert model.layers[3].activation.__name__ == "relu"\nassert np.isclose(model.layers[4].rate, .30)\nassert model.layers[5].units == 1\nassert model.layers[5].activation.__name__ == "sigmoid"\nassert sum(int(np.prod(w.shape)) for w in model.trainable_weights) == 34529\nassert not model.non_trainable_weights\nfig = dibujar_arquitectura(obtenida)\nprint(guardar_figura_nueva(fig, "arquitectura_modelo_guardado_verificada.png"))\nplt.show()'),
        md("La tabla se deriva del modelo cargado; el diagrama se genera desde esa tabla y no se pega como decoración. El control de configuración también funciona leyendo el contenedor `.keras` sin TensorFlow. Cargar con `compile=False` no cambia los pesos ni vuelve a entrenar.")])

    notebook("12_analisis_final", "12. Análisis final basado en artefactos", "Integrar datos, metodología y resultados leyendo archivos existentes, sin transcribir métricas manualmente.", [
        code('dataset = resumen_dataset()\ncomparacion = comparar_ejecuciones()\narquitectura = verificar_configuracion()\nmetricas = cargar_metricas("metricas_evaluacion.csv")\nimportancia = importancia_guardada()\nverificadas, matriz = verificar_predicciones()\ndisplay(dataset)\ndisplay(comparacion)\ndisplay(arquitectura)\ndisplay(metricas)\ndisplay(importancia)\nassert arquitectura.parametros.sum() == 34529'),
        md("## Metodología y límites\nEl original se auditó mediante reglas operacionales; se filtró y se seleccionó una muestra estratificada ya existente. Se excluyó `id`, se convirtió edad a años y se mantuvieron 11 predictoras. La CNN se implementó con TensorFlow/Keras. Los dos protocolos usan la misma arquitectura.\n\nLa ejecución definitiva ajustó el escalador antes de reservar validación interna; la prueba quedó fuera del ajuste. Las pruebas de ambas ejecuciones no se obtuvieron con el mismo procedimiento. No hay estimación de variabilidad entre semillas ni validación externa. Las importancias corresponden a un árbol independiente. No se guardó el escalador definitivo ni una lista explícita de identificadores de las particiones. La secuencia documenta resultados existentes y no acredita una nueva evaluación del procedimiento recomendado sin fuga."),
        code('from IPython.display import Markdown\nvalores = metricas.set_index("Metrica").Valor\nfilas = dataset.set_index("Etapa").Registros\nepocas = len(pd.read_csv(RESULTADOS / "tablas/historial_entrenamiento.csv"))\ntotal = int(arquitectura.parametros.sum())\ntexto = (f"La muestra experimental contiene **{int(filas.iloc[-1]):,} registros** seleccionados de "\n         f"{int(filas.iloc[0]):,}; el conjunto limpio contiene {int(filas.iloc[1]):,}. "\n         f"El modelo utiliza {len(VARIABLES)} predictoras y **{total:,} parámetros**. "\n         f"El entrenamiento definitivo ejecutó {epocas} épocas con restauración de mejores pesos. "\n         + "Resultados conservados: " + "; ".join(f"{m}: **{v:.4f}**" for m, v in valores.items()) + ". "\n         + f"La variable de mayor importancia en el árbol complementario es {importancia.iloc[0].variable_es}. "\n         + "Estos resultados describen el protocolo ejecutado y deben interpretarse con las limitaciones documentadas.")\ndisplay(Markdown(texto))\nprint("Verificación de métricas desde predicciones y parámetros desde modelo: satisfactoria.")')])

    # Ningún artefacto de resultados, dataset o generador original debe cambiar.
    for rel, digest in snapshot.items():
        assert sha(ROOT / rel) == digest, rel
    for info in backups.values():
        assert sha(ROOT / info["respaldo"]) == info["sha256"]
    assert len(list(DEST.glob("*.ipynb"))) == 12
    print("12 notebooks creados; respaldos verificados; datos y resultados originales intactos.")


if __name__ == "__main__":
    main()
