import nbformat as nbf
from pathlib import Path

p = Path("notebooks/nuevo_experimento/08_nueva_arquitectura_cnn.ipynb")
p.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb.cells = [

nbf.v4.new_markdown_cell("""# 08. Nueva arquitectura 1D-CNN

**Objetivo:** desarrollar una arquitectura de red neuronal convolucional 1D modificada para la clasificación de enfermedad cardiovascular, utilizando una muestra estratificada de 15 000 registros y limitando el número de parámetros del modelo al intervalo de 30 000 a 50 000.
"""),

nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
"""),

nbf.v4.new_markdown_cell("""## 1. Configuración experimental

Se utiliza la muestra estratificada de 15 000 registros generada durante la auditoría del dataset. El identificador `id` no se utiliza como variable predictora y la variable `cardio` constituye la variable objetivo.
"""),

nbf.v4.new_code_cell("""DATA_PATH = Path("../../resultados/nuevo_experimento/cardio_15000.csv")
OUT_DIR = Path("../../resultados/nuevo_experimento")
FIG_DIR = OUT_DIR / "figuras"
TAB_DIR = OUT_DIR / "tablas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("Dataset:", DATA_PATH)
print("Semilla:", RANDOM_STATE)
"""),

nbf.v4.new_code_cell("""df = pd.read_csv(DATA_PATH, sep=";")

print("Dimensiones:", df.shape)
print("Nulos:", int(df.isnull().sum().sum()))
print("Duplicados:", int(df.duplicated().sum()))
print("\\nDistribución de cardio:")
print(df["cardio"].value_counts().sort_index())
"""),

nbf.v4.new_markdown_cell("""## 2. Preparación de las variables

La columna `id` se excluye porque únicamente identifica el registro. La edad, almacenada originalmente en días, se transforma a años. La variable `cardio` se utiliza como objetivo binario.
"""),

nbf.v4.new_code_cell("""df_model = df.copy()

df_model["age"] = df_model["age"] / 365.25

X = df_model.drop(columns=["id", "cardio"])
y = df_model["cardio"]

print("Dimensión de X:", X.shape)
print("Dimensión de y:", y.shape)

print("\\nVariables predictoras:")
print(X.columns.tolist())

print("\\nNúmero de variables predictoras:", X.shape[1])
"""),

nbf.v4.new_markdown_cell("""## 3. División entrenamiento, validación y prueba

Se utiliza una división 70/15/15 con estratificación respecto de `cardio`. La estratificación mantiene una distribución similar de las dos clases en los tres subconjuntos.
"""),

nbf.v4.new_code_cell("""X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Entrenamiento:", X_train.shape, y_train.shape)
print("Validación:", X_val.shape, y_val.shape)
print("Prueba:", X_test.shape, y_test.shape)

print("\\nDistribución entrenamiento:")
print(y_train.value_counts(normalize=True).sort_index())

print("\\nDistribución validación:")
print(y_val.value_counts(normalize=True).sort_index())

print("\\nDistribución prueba:")
print(y_test.value_counts(normalize=True).sort_index())
"""),

nbf.v4.new_markdown_cell("""## 4. Escalamiento

El escalador se ajusta exclusivamente con los datos de entrenamiento y posteriormente se aplica a validación y prueba. De esta manera se evita utilizar información de los subconjuntos de evaluación durante el ajuste del preprocesamiento.
"""),

nbf.v4.new_code_cell("""scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Train escalado:", X_train_scaled.shape)
print("Validation escalado:", X_val_scaled.shape)
print("Test escalado:", X_test_scaled.shape)
"""),

nbf.v4.new_markdown_cell("""## 5. Preparación de la entrada para 1D-CNN

La entrada se representa como:

`(número de registros, número de variables, canales)`

Por tanto, cada observación contiene 11 variables clínicas y un canal.
"""),

nbf.v4.new_code_cell("""X_train_cnn = X_train_scaled[..., np.newaxis]
X_val_cnn = X_val_scaled[..., np.newaxis]
X_test_cnn = X_test_scaled[..., np.newaxis]

print("Entrada CNN:")
print("Train:", X_train_cnn.shape)
print("Validation:", X_val_cnn.shape)
print("Test:", X_test_cnn.shape)

assert X_train_cnn.shape[1:] == (11, 1)
print("\\nLa entrada cumple la forma esperada (11, 1).")
"""),

nbf.v4.new_markdown_cell("""## 6. Diseño de la nueva arquitectura

La arquitectura se modifica respecto al modelo de referencia para reducir su complejidad y mantener el número de parámetros entre 30 000 y 50 000.

La estructura propuesta es:

**Entrada (11, 1) → Conv1D(32) → Conv1D(96) → GlobalAveragePooling1D → Dense(256) → Dropout(0.30) → salida sigmoid.**
"""),

nbf.v4.new_code_cell("""model = Sequential([
    Input(shape=(11, 1)),

    Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    Conv1D(
        filters=96,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    GlobalAveragePooling1D(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.summary()
"""),

nbf.v4.new_markdown_cell("""## 7. Verificación del número de parámetros

El modelo debe cumplir la restricción metodológica establecida para el experimento: entre 30 000 y 50 000 parámetros totales.
"""),

nbf.v4.new_code_cell("""total_params = model.count_params()

print("========================================")
print("PARÁMETROS TOTALES:", total_params)
print("========================================")

assert 30000 <= total_params <= 50000, (
    f"La arquitectura tiene {total_params} parámetros y "
    "no cumple el rango 30 000–50 000."
)

print("✓ La arquitectura cumple el rango solicitado.")
"""),

nbf.v4.new_markdown_cell("""## 8. Compilación

Se utiliza Adam como optimizador y entropía cruzada binaria como función de pérdida, debido a que `cardio` representa una clasificación binaria.
"""),

nbf.v4.new_code_cell("""model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("Modelo compilado correctamente.")
"""),

nbf.v4.new_markdown_cell("""## 9. Entrenamiento

Se utiliza EarlyStopping para detener el entrenamiento cuando la pérdida de validación deje de mejorar y recuperar los mejores pesos encontrados.
"""),

nbf.v4.new_code_cell("""early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("\\nÉpocas ejecutadas:", len(history.history["loss"]))
"""),

nbf.v4.new_markdown_cell("""## 10. Curvas de entrenamiento"""),

nbf.v4.new_code_cell("""history_df = pd.DataFrame(history.history)
history_df.index = history_df.index + 1
history_df.index.name = "epoch"

history_path = TAB_DIR / "historial_nueva_cnn.csv"
history_df.to_csv(history_path)

fig, ax = plt.subplots()
ax.plot(history_df.index, history_df["accuracy"], label="Entrenamiento")
ax.plot(history_df.index, history_df["val_accuracy"], label="Validación")
ax.set_title("Accuracy durante el entrenamiento")
ax.set_xlabel("Época")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "curva_accuracy_nueva_cnn.png", dpi=300)
plt.show()

fig, ax = plt.subplots()
ax.plot(history_df.index, history_df["loss"], label="Entrenamiento")
ax.plot(history_df.index, history_df["val_loss"], label="Validación")
ax.set_title("Loss durante el entrenamiento")
ax.set_xlabel("Época")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "curva_loss_nueva_cnn.png", dpi=300)
plt.show()

print("Historial guardado en:", history_path)
"""),

nbf.v4.new_markdown_cell("""## 11. Evaluación sobre el conjunto de prueba"""),

nbf.v4.new_code_cell("""y_prob = model.predict(X_test_cnn, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

metrics_df = pd.DataFrame({
    "metrica": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "valor": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_path = TAB_DIR / "metricas_nueva_cnn.csv"
metrics_df.to_csv(metrics_path, index=False)

print(metrics_df.to_string(index=False))
print("\\nMétricas guardadas en:", metrics_path)
"""),

nbf.v4.new_markdown_cell("""## 12. Reporte de clasificación"""),

nbf.v4.new_code_cell("""print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Sin enfermedad", "Con enfermedad"],
        digits=4,
        zero_division=0
    )
)
"""),

nbf.v4.new_markdown_cell("""## 13. Matriz de confusión"""),

nbf.v4.new_code_cell("""cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_title("Matriz de confusión — nueva 1D-CNN")
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor real")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()

cm_path = FIG_DIR / "matriz_confusion_nueva_cnn.png"
fig.savefig(cm_path, dpi=300)
plt.show()

print("Matriz guardada en:", cm_path)
"""),

nbf.v4.new_markdown_cell("""## 14. Curva ROC"""),

nbf.v4.new_code_cell("""fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("Curva ROC — nueva 1D-CNN")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.legend()
fig.tight_layout()

roc_path = FIG_DIR / "curva_roc_nueva_cnn.png"
fig.savefig(roc_path, dpi=300)
plt.show()

print("ROC guardada en:", roc_path)
"""),

nbf.v4.new_markdown_cell("""## 15. Resumen experimental"""),

nbf.v4.new_code_cell("""summary = pd.DataFrame({
    "elemento": [
        "Registros utilizados",
        "Variables predictoras",
        "Entrenamiento",
        "Validación",
        "Prueba",
        "Parámetros del modelo",
        "Épocas máximas",
        "Épocas ejecutadas",
        "Batch size"
    ],
    "valor": [
        len(df),
        X.shape[1],
        len(X_train),
        len(X_val),
        len(X_test),
        total_params,
        100,
        len(history.history["loss"]),
        32
    ]
})

display(summary)

summary.to_csv(
    TAB_DIR / "resumen_nueva_cnn.csv",
    index=False
)
"""),

nbf.v4.new_markdown_cell("""## 16. Conclusión

La nueva arquitectura 1D-CNN fue diseñada para reducir la cantidad de parámetros respecto del modelo de referencia y cumplir el intervalo de 30 000 a 50 000 parámetros establecido para el experimento. La evaluación sobre un conjunto de prueba independiente permitirá analizar su desempeño mediante Accuracy, Precision, Recall, F1-score y ROC-AUC.

Los resultados obtenidos se utilizarán posteriormente para comparar el comportamiento de la arquitectura modificada y discutir su relación con el objetivo de reducir la complejidad del modelo.
""")
]

nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3.11.9 (.venv)",
    "language": "python",
    "name": "python3"
}

nb["metadata"]["language_info"] = {
    "name": "python",
    "version": "3.11.9"
}

nbf.write(nb, p)

print("Notebook creado correctamente:")
print(p)
print("Celdas:", len(nb.cells))
