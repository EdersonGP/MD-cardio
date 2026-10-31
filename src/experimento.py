"""Lectura y verificación del experimento Cardio. Sin entrenamiento automático.

Los archivos originales de resultados son de solo lectura para estas funciones.
Las figuras nuevas se escriben exclusivamente en figuras/reorganizacion/.
"""
from pathlib import Path
from functools import lru_cache
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/nuevo_dataset/cardio_train.csv"
RESULTADOS = ROOT / "resultados/nuevo_experimento"
VARIABLES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
             "cholesterol", "gluc", "smoke", "alco", "active"]
NOMBRES = dict(zip(VARIABLES, ["Edad", "Sexo", "Altura", "Peso", "Presión sistólica",
               "Presión diastólica", "Colesterol", "Glucosa", "Tabaquismo",
               "Consumo de alcohol", "Actividad física"]))


@lru_cache(maxsize=1)
def tensorflow_disponible():
    """No basta encontrar el paquete: su runtime nativo debe poder importarse."""
    try:
        import tensorflow  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def cargar_datos(path):
    df = pd.read_csv(path, sep=";")
    assert df.columns.tolist() == ["id", *VARIABLES, "cardio"]
    assert df.isna().sum().sum() == 0
    assert set(df.cardio.unique()) == {0, 1}
    return df


def reglas_calidad(df):
    return pd.DataFrame({
        "edad": ~(df.age / 365.25).between(18, 100),
        "altura": ~df.height.between(120, 220),
        "peso": ~df.weight.between(30, 250),
        "ap_hi": ~df.ap_hi.between(70, 250),
        "ap_lo": ~df.ap_lo.between(40, 200),
        "presion_relacion": ~(df.ap_hi > df.ap_lo),
    })


def cargar_metricas(nombre):
    tabla = pd.read_csv(RESULTADOS / "tablas" / nombre)
    tabla.columns = ["Metrica", "Valor"]
    assert tabla.Metrica.tolist() == ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    assert tabla.Valor.between(0, 1).all()
    return tabla


def comparar_ejecuciones():
    resumen = pd.read_csv(RESULTADOS / "tablas/resumen_nueva_cnn.csv").set_index("elemento").valor
    muestra = cargar_datos(RESULTADOS / "cardio_15000.csv")
    n = len(muestra)
    inicial = cargar_metricas("metricas_nueva_cnn.csv").set_index("Metrica").Valor
    final = cargar_metricas("metricas_evaluacion.csv").set_index("Metrica").Valor
    # Tamaños del protocolo original, sin ajustar ni entrenar un modelo.
    from sklearn.model_selection import train_test_split
    indices_train, indices_test = train_test_split(np.arange(n), test_size=.15,
                                                  stratify=muestra.cardio, random_state=42)
    ajuste = int(len(indices_train) * (1 - .20))  # Keras validation_split, último bloque.
    rows = []
    for protocolo, tr, va, te, history, metrics in [
        ("StandardScaler", int(resumen["Entrenamiento"]), int(resumen["Validación"]),
         int(resumen["Prueba"]), "historial_nueva_cnn.csv", inicial),
        ("MinMaxScaler", ajuste, len(indices_train) - ajuste, len(indices_test),
         "historial_entrenamiento.csv", final),
    ]:
        rows.append({"Protocolo": protocolo, "Entrenamiento": tr, "Validación": va,
                     "Prueba": te, "Épocas": len(pd.read_csv(RESULTADOS / "tablas" / history)),
                     "Parámetros": int(resumen["Parámetros del modelo"]),
                     "Accuracy": metrics["Accuracy"], "Precision": metrics["Precision"],
                     "Recall": metrics["Recall"], "F1": metrics["F1-score"], "ROC-AUC": metrics["ROC-AUC"]})
    return pd.DataFrame(rows)


def tabla_configuracion():
    """Derivar dimensiones y parámetros de config.json, sin importar TensorFlow."""
    with zipfile.ZipFile(RESULTADOS / "modelo_nueva_cnn.keras") as z:
        config = json.loads(z.read("config.json"))
    layers = config["config"]["layers"]
    shape = None
    rows = []
    for layer in layers:
        tipo, c = layer["class_name"], layer["config"]
        if tipo == "InputLayer":
            shape = tuple(c["batch_shape"][1:])
            assert shape == (11, 1)
            continue
        assert shape is not None
        filtros = kernel = unidades = dropout = None
        activacion = c.get("activation", "—")
        padding = c.get("padding", "—")
        if tipo == "Conv1D":
            filtros, kernel = c["filters"], c["kernel_size"][0]
            assert c["strides"] == [1] and c["dilation_rate"] == [1] and c["groups"] == 1
            assert padding == "same" and c["use_bias"]
            params = (kernel * shape[-1] + 1) * filtros
            shape = (shape[0], filtros)
        elif tipo == "GlobalAveragePooling1D":
            assert not c["keepdims"] and c["data_format"] == "channels_last"
            params, shape = 0, (shape[-1],)
        elif tipo == "Dense":
            unidades = c["units"]
            assert c["use_bias"]
            params, shape = (shape[-1] + 1) * unidades, (unidades,)
        elif tipo == "Dropout":
            params, dropout = 0, c["rate"]
        else:
            raise AssertionError(f"Capa inesperada: {tipo}")
        rows.append({"capa": c["name"], "tipo": tipo, "salida": shape,
                     "parametros": params, "filtros": filtros, "kernel": kernel,
                     "activacion": activacion, "padding": padding,
                     "unidades": unidades, "dropout": dropout})
    return pd.DataFrame(rows)


def verificar_configuracion():
    tabla = tabla_configuracion()
    assert tabla.tipo.tolist() == ["Conv1D", "Conv1D", "GlobalAveragePooling1D", "Dense", "Dropout", "Dense"]
    assert tabla.parametros.tolist() == [128, 9312, 0, 24832, 0, 257]
    assert tabla.salida.tolist() == [(11, 32), (11, 96), (96,), (256,), (256,), (1,)]
    assert tabla.filtros.iloc[:2].tolist() == [32, 96]
    assert tabla.kernel.iloc[:2].tolist() == [3, 3]
    assert tabla.activacion.tolist() == ["relu", "relu", "—", "relu", "—", "sigmoid"]
    assert tabla.unidades.iloc[[3, 5]].tolist() == [256, 1]
    assert np.isclose(tabla.dropout.iloc[4], .30)
    assert int(tabla.parametros.sum()) == 34529
    guardada = pd.read_csv(RESULTADOS / "tablas/arquitectura_capas.csv")
    assert guardada.tipo.tolist() == tabla.tipo.tolist()
    assert guardada.parametros.tolist() == tabla.parametros.tolist()
    assert guardada.entrenables.sum() == 34529 and guardada.no_entrenables.sum() == 0
    assert guardada.forma_salida.tolist() == [str((None, *s)) for s in tabla.salida]
    return tabla


def construir_cnn():
    """Construir, nunca ajustar. Los notebooks 05/08 muestran también este código explícito."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout
    model = Sequential([
        Input(shape=(11, 1)),
        Conv1D(32, 3, padding="same", activation="relu"),
        Conv1D(96, 3, padding="same", activation="relu"),
        GlobalAveragePooling1D(), Dense(256, activation="relu"),
        Dropout(.30), Dense(1, activation="sigmoid"),
    ])
    assert model.count_params() == 34529
    return model


def tabla_modelo(model):
    rows = []
    for layer in model.layers:
        c = layer.get_config()
        rows.append({"capa": layer.name, "tipo": type(layer).__name__,
                     "salida": tuple(layer.output.shape[1:]), "parametros": layer.count_params(),
                     "filtros": c.get("filters"), "kernel": (c.get("kernel_size") or [None])[0],
                     "activacion": c.get("activation", "—"), "padding": c.get("padding", "—"),
                     "unidades": c.get("units"), "dropout": c.get("rate")})
    return pd.DataFrame(rows)


def dibujar_arquitectura(tabla):
    """Diagrama vertical construido desde las capas, dimensiones y parámetros recibidos."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(-.9, 8)
    ax.axis("off")
    boxes = [("Entrada", "(11, 1) · 11 variables, 1 canal")]
    for r in tabla.to_dict("records"):
        titulo = r["tipo"]
        detalles = []
        if pd.notna(r.get("filtros")):
            titulo += f" · {int(r['filtros'])} filtros"
            detalles.append(f"kernel {int(r['kernel'])} · padding {r['padding']}")
        if pd.notna(r.get("unidades")):
            titulo += f" · {int(r['unidades'])} neuronas"
        if pd.notna(r.get("dropout")):
            titulo += f" · tasa {r['dropout']:.2f}"
        if r.get("activacion", "—") != "—":
            detalles.append(r["activacion"])
        detalles.append(f"Salida {tuple(r['salida'])} · Parámetros {int(r['parametros']):,}")
        boxes.append((titulo, "\n".join(detalles)))
    for i, (titulo, detalle) in enumerate(boxes):
        y = 7 - i
        ax.text(5, y, titulo + "\n" + detalle, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": "#eaf2fa", "edgecolor": "#245784"})
        if i:
            ax.annotate("", xy=(5, y + .35), xytext=(5, y + .66),
                        arrowprops={"arrowstyle": "->", "color": "#245784", "lw": 1.8})
    ax.set_title("1D-CNN definitiva · TensorFlow/Keras", fontsize=16, pad=16)
    ax.text(5, -.3, f"Total: {int(tabla.parametros.sum()):,} parámetros · Requisito: 30,000–50,000",
            ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


def guardar_figura_nueva(fig, nombre):
    destino = RESULTADOS / "figuras/reorganizacion" / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    return destino


def importancia_guardada():
    tabla = pd.read_csv(RESULTADOS / "tablas/importancia_variables.csv")
    assert set(tabla.variable) == set(VARIABLES)
    assert len(tabla) == 11 and np.isclose(tabla.importancia.sum(), 1)
    assert tabla.importancia.ge(0).all()
    tabla["variable_es"] = tabla.variable.map(NOMBRES)
    return tabla.sort_values("importancia", ascending=False).reset_index(drop=True)


def resumen_dataset():
    rows = []
    for etapa, path in [("Original", ORIGINAL), ("Limpio", RESULTADOS / "cardio_limpio.csv"),
                        ("Muestra", RESULTADOS / "cardio_15000.csv")]:
        df = cargar_datos(path)
        rows.append({"Etapa": etapa, "Registros": len(df), "Columnas": len(df.columns),
                     "Predictoras": len(VARIABLES), "Clase 0": int(df.cardio.eq(0).sum()),
                     "Clase 1": int(df.cardio.eq(1).sum())})
    return pd.DataFrame(rows)


def verificar_predicciones():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    y, prob, pred = [np.load(RESULTADOS / f"{n}.npy", allow_pickle=False).ravel()
                     for n in ["y_test", "y_prob", "y_pred"]]
    assert y.shape == prob.shape == pred.shape == (2250,)
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    assert set(np.unique(y)) == {0, 1}
    assert np.array_equal(pred, (prob >= .5).astype(int))
    valores = [accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred),
               f1_score(y, pred), roc_auc_score(y, prob)]
    tabla = cargar_metricas("metricas_evaluacion.csv")
    np.testing.assert_allclose(tabla.Valor, valores, rtol=0, atol=1e-12)
    return pd.DataFrame({"Metrica": tabla.Metrica, "Valor": valores}), confusion_matrix(y, pred)
