# Auditoría de reproducción del artículo semilla

Documento de control metodológico del proyecto Cardio. Compara lo que el artículo semilla declara con lo que este proyecto ha hecho hasta ahora (Notebooks 01 a 05). En todo el documento se usan cuatro etiquetas: **Artículo** (lo que el artículo declara), **Nuestro trabajo** (lo que se hizo en este proyecto), **Diferencia** (lo que no coincide) y **Pendiente** (lo que todavía no se ha hecho o no se puede determinar).

Fuentes: `articulo/articulo_semilla.pdf` (texto, Figura 7, Tablas 1 a 4 y Figuras 9 a 12 leídas directamente del PDF), los notebooks `notebooks/01` a `05` y los archivos de `resultados/`.

## 1. Objetivo de la auditoría

Comprobar qué partes del método del artículo han sido reproducidas, cuáles fueron decisiones propias del proyecto y cuáles siguen pendientes. La finalidad es poder explicar con precisión, ante terceros, qué se está reproduciendo del artículo y qué corresponde a decisiones experimentales propias, y evitar que una decisión del proyecto se presente como si fuera del artículo.

## 2. Identificación del artículo

| Campo | Dato |
|---|---|
| Título | A one-dimensional convolutional neural network-based deep learning approach for predicting cardiovascular diseases |
| Autores | Dhafer G. Honi y Laszlo Szathmary (University of Debrecen, Department of IT, Hungría) |
| Año | 2024 (disponible en línea el 22 de junio de 2024) |
| Revista | Informatics in Medicine Unlocked, volumen 49, artículo 101535 |
| DOI | 10.1016/j.imu.2024.101535 |
| Dataset utilizado | UCI Heart Disease, descrito en el artículo como 14 atributos de 1025 pacientes (enlace del artículo: https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Objetivo del artículo | Predecir enfermedades cardiovasculares con una red neuronal convolucional unidimensional (1D-CNN), combinada con preprocesamiento, análisis del dataset y selección de variables, evaluada con una división entrenamiento/prueba y con validación cruzada K-Fold |

## 3. Dataset

| Aspecto | Artículo | Nuestro trabajo | Diferencia | Pendiente |
|---|---|---|---|---|
| Número de registros | 1025 pacientes (sección 3.1) | 1025 registros en `data/heart.csv` | Ninguna | Ninguno |
| Número de variables | 14 atributos (Tabla 1) | 14 columnas | Ninguna | Ninguno |
| Variables predictoras | 13: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal | Las mismas 13, con los mismos nombres | Ninguna | Ninguno |
| Variable objetivo | `target`: 0 = sin enfermedad, 1 = con enfermedad (Tabla 1) | `target`: 499 registros con 0 y 526 con 1 | Ninguna en la definición | Ninguno |
| Fuente | UCI Heart Disease, con enlace al repositorio de UCI | Archivo local `data/heart.csv` | No se registró en el repositorio la procedencia exacta del archivo | Documentar la procedencia del archivo de 1025 filas |
| Valores faltantes | El artículo menciona "missing values" en la descripción general de la metodología, sin detallar su tratamiento | 0 valores faltantes (Notebook 01) | No comparable: el artículo no describe el tratamiento | Ninguno |
| Duplicados | El artículo no los menciona | 723 duplicados exactos (Notebook 01) | Es un aspecto que el artículo no documenta | Ver secciones 9 y 10 |

Dos observaciones sobre el dataset del artículo:

- La sección 3.3.1 describe la eliminación de variables como ID, número de seguro social (ccf), localización del dolor (painloc), nombres de pacientes y "58 atributos adicionales". Esas variables no existen en un archivo de 14 columnas, por lo que esa descripción corresponde a una versión más amplia de la base de datos original. El artículo no explica cómo se llega desde esa versión hasta 1025 pacientes con 14 atributos.
- En la revisión de literatura, el artículo menciona un trabajo previo cuyo dataset tenía 303 instancias y 14 atributos. Esta auditoría no pudo comprobar que el archivo de 1025 filas proceda directamente del enlace de UCI citado. Lo único verificable es que nuestro archivo coincide con lo que el artículo describe (1025 pacientes, 14 atributos).

## 4. Preprocesamiento

| Aspecto | Artículo | Nuestro trabajo | Diferencia | Pendiente |
|---|---|---|---|---|
| Tratamiento de datos | Tres tareas: limpieza, codificación categórica y reescalado (sección 3.3). Menciona valores faltantes y outliers sin describir su tratamiento | Revisión de faltantes (0) y de duplicados; sin tratamiento de outliers; flujo principal sin eliminar nada | El artículo no detalla el procedimiento de limpieza, así que no se puede comparar paso a paso | Aclarar si el artículo trató outliers; no puede determinarse con el texto |
| Escalamiento | Reescalado a [0, 1] de todos los atributos con la fórmula (X - Xmin) / (Xmax - Xmin) (sección 3.3.3) | `MinMaxScaler` a [0, 1] sobre las 13 variables, ajustado solo con `X_train` y aplicado a `X_train` y `X_test` (Notebook 02) | El artículo no indica si el escalador se ajustó antes o después de dividir; el proyecto lo ajustó solo con entrenamiento por criterio anti-fuga de información | Ninguno |
| Codificación | One-Hot Encoding para sex, cp, fbs, restecg, exang, slope, thal y target (sección 3.3.2); `ca` no aparece en esa lista | No se aplicó One-Hot; se mantienen las 13 columnas numéricas originales | Diferencia deliberada. One-Hot elevaría las 13 columnas a 30 y no coincidiría con la entrada (13, 1) de la Tabla 2 y la Figura 7. El propio artículo es ambiguo: describe One-Hot, pero usa una entrada de 13 variables | Evaluar una variante con One-Hot en un experimento posterior |
| División entrenamiento/prueba | 70 % / 30 % (sección 3.8). No indica semilla ni estratificación | 70 % / 30 %, `random_state=42`, `stratify=y` | La proporción coincide; semilla y estratificación son decisiones del proyecto | Ninguno |
| Chi-cuadrado | Usa el ranking Chi-cuadrado para valorar la importancia de las variables (sección 3.4, Figura 6) | No se calculó | El proyecto no lo ha reproducido | Calcular el ranking Chi-cuadrado y compararlo con el del artículo |

## 5. Selección de variables

**Artículo.** Según el ranking Chi-cuadrado (Figura 6), las variables más influyentes son **cp, ca, oldpeak y thalach**. El resumen del artículo habla de "selección automática de variables esenciales" y la sección 3 indica que se realizó selección de características y escalamiento.

**Aclaración importante.** La arquitectura del artículo (Tabla 2 y Figura 7) recibe una entrada de **13 variables**, con forma (13, 1). Por eso el texto del artículo no permite afirmar que la CNN se haya entrenado únicamente con esas cuatro variables. El artículo usa el ranking para analizar la importancia de las variables, pero no describe un modelo entrenado con un subconjunto reducido.

**Nuestro trabajo.** La CNN se entrenó con las 13 variables, sin selección de características (Notebooks 03, 04 y 05). Tampoco debe afirmarse que el proyecto entrenó con cp, ca, oldpeak y thalach.

**Pendiente.** Calcular el Chi-cuadrado sobre el dataset y comprobar si el ranking coincide con el del artículo. Una variante con solo las cuatro variables sería un experimento adicional, no una reproducción del artículo.

## 6. Arquitectura 1D-CNN

### 6.1 Comparación

La fuente principal usada por el proyecto es la **Figura 7** del artículo. La Tabla 2 y el texto presentan diferencias, que se detallan en la sección 6.2.

| Elemento | Artículo | Nuestro trabajo | Diferencia |
|---|---|---|---|
| Entrada | (13, 1) | (13, 1) | Ninguna |
| Conv1D, bloque 1 | 32 filtros, kernel 2, activación sigmoid | 32 filtros, kernel 2, sigmoid | Ninguna |
| Conv1D, bloque 2 | 64 filtros, kernel 2, sigmoid (Figura 7) | 64 filtros, kernel 2, sigmoid | Ninguna respecto de la Figura 7; difiere de la Tabla 2 (ver 6.2) |
| Conv1D, bloque 3 | 128 filtros, kernel 2, sigmoid (Figura 7) | 128 filtros, kernel 2, sigmoid | Ninguna respecto de la Figura 7; difiere de la Tabla 2 (ver 6.2) |
| Conv1D, bloque 4 | 128 filtros, kernel 2, sigmoid | 128 filtros, kernel 2, sigmoid | Ninguna |
| Padding | No especificado | `valid` (valor por defecto de Keras) | Decisión del proyecto, elegida porque produce el Flatten de 1152 de la Figura 7 |
| Pooling | No se describe ninguna capa de pooling | No hay pooling | Ninguna |
| Batch Normalization | Después de cada bloque convolucional (4 capas) | 4 capas, tras cada Conv1D | Ninguna. Parámetros internos no especificados; se usaron los valores por defecto de Keras |
| Dropout | 0.2 tras el bloque 1, 0.5 tras el bloque 2 y 0.1 en la parte densa (Figura 7 y Tabla 2) | 0.2 tras el bloque 1, 0.5 tras el bloque 2 y 0.1 tras Dense(200) | Ninguna respecto de la Figura 7 y la Tabla 2; el texto la ubica distinto (ver 6.2) |
| Flatten | 1152 valores (Figura 7) o 1280 (Tabla 2) | 1152 valores | Se siguió la Figura 7 |
| Dense oculta | 200 neuronas, ReLU | 200 neuronas, ReLU | Ninguna |
| Salida | 1 neurona, sigmoid | 1 neurona, sigmoid | Ninguna |
| Orden dentro del bloque | Conv, BatchNorm, Dropout (Tabla 2) | Conv, BatchNorm, Dropout | Ninguna |
| Parámetros totales | La Tabla 2 no da total; la suma de sus filas es 301,137 | 285,873 (véase la nota siguiente) | Ver 6.2 |

Nota sobre los parámetros: el total de 285,873 corresponde al modelo construido (285,169 entrenables y 704 no entrenables). La cifra de 301,137 se obtiene sumando las filas de la Tabla 2; el artículo no la reporta como total.

### 6.2 Inconsistencias encontradas en el artículo

Estas inconsistencias no se ocultan ni se corrigen: se documentan tal como aparecen.

| N.º | Inconsistencia | Detalle |
|---|---|---|
| 1 | Tamaño del Flatten | La Figura 7 indica 1152 valores y la Tabla 2 indica 1280. Con cuatro convoluciones de kernel 2 sin relleno, la longitud pasa de 13 a 9 y 9 × 128 = 1152. El valor 1280 equivale a 10 × 128 |
| 2 | Parámetros de la capa Dense(200) | La Tabla 2 da 256,200, que es consistente con un Flatten de 1280 (1280 × 200 + 200). Con 1152 serían 230,600 |
| 3 | Parámetros de las convoluciones 2 y 3 | La Tabla 2 da 2080 y 8256. Con los filtros de la Figura 7 (64 y 128, kernel 2) serían 4160 y 16,512. Coinciden las convoluciones 1 y 4 (96 y 32,896) |
| 4 | Tabla 2 internamente | Las filas de Batch Normalization de la Tabla 2 tienen 128, 256, 512 y 512 parámetros, que corresponden a 32, 64, 128 y 128 canales, es decir, los filtros de la Figura 7. Pero los parámetros de las convoluciones 2 y 3 de la misma tabla no corresponden a esos canales |
| 5 | Primera fila de la Tabla 2 | La fila "Input layer" tiene 96 parámetros y activación sigmoid, es decir, mezcla la entrada con la primera convolución |
| 6 | Longitud tras la primera convolución | Los valores de la Tabla 2 (13, 12, 11, 10) y el Flatten de 1280 son compatibles con que la primera convolución conserve la longitud 13; los de la Figura 7 (Flatten 1152) no. El artículo no indica el padding. Esto es una inferencia de esta auditoría, no una declaración del artículo |
| 7 | Ubicación del Dropout de 0.1 | El texto dice que los dropout se aplican "después de cada capa convolucional" con 20 %, 50 % y 10 %. La Figura 7 y la Tabla 2 colocan el de 0.1 después de la capa densa |
| 8 | Número de tamaños de filtro | El texto menciona filtros de tamaños "(32, 64, 128)"; la Figura 7 usa 32, 64, 128 y 128 |
| 9 | Función de activación | La sección 3 dice que ReLU se usó como función de transferencia de la capa oculta, mientras que las convoluciones usan sigmoid según la Figura 7 y la Tabla 2. Solo la capa Dense(200) usa ReLU |

Decisión del proyecto ante estas inconsistencias: se construyó el modelo de la Figura 7 (filtros 32, 64, 128 y 128, `padding="valid"`, Flatten de 1152). La reproducción exacta de los números de la Tabla 2 no es posible con una sola arquitectura. **Pendiente:** si se quisiera aproximar la Tabla 2, habría que probar una variante, por ejemplo con relleno `same` en la primera convolución, y declararla como variante propia.

## 7. Hiperparámetros

### 7.1 Parámetros especificados por el artículo

| Parámetro | Valor | Dónde aparece |
|---|---|---|
| Épocas | 150 | Sección 4.1 |
| Proporción de división | 70 % entrenamiento, 30 % prueba | Sección 3.8 |
| Filtros, kernel, activaciones, dropout, neuronas | Ver sección 6 | Figura 7, Tabla 2 y sección 3.5 |
| Validación cruzada | K-Fold con K = 5 y K = 10 | Sección 4.2 |

### 7.2 Parámetros que nuestro proyecto tuvo que definir

El artículo no los especifica; no deben presentarse como del artículo.

| Parámetro | Valor | Observación |
|---|---|---|
| Optimizador | Adam | Decisión del proyecto |
| Learning rate | 0.001 | Decisión del proyecto |
| Función de pérdida | `binary_crossentropy` | Decisión del proyecto |
| Métrica de compilación | `accuracy` | Decisión del proyecto |
| Batch size | 32 | Decisión del proyecto |
| Semilla aleatoria | 42, con operaciones deterministas de TensorFlow | Decisión del proyecto, para reproducibilidad |
| Validación interna | `validation_split=0.20`: 573 registros ajustan los pesos y 144 validan en el Experimento A; 168 y 43 en el Experimento B | Decisión del proyecto |
| Padding | `valid` | Decisión del proyecto |
| Umbral de clasificación | 0.5 | Decisión del proyecto |
| Estratificación de la división | `stratify=y` | Decisión del proyecto |

### 7.3 Parámetros que el artículo no especifica

Optimizador, learning rate, pérdida, métricas de compilación, batch size, padding y stride, inicialización de pesos, parámetros internos de Batch Normalization, semilla de la división, uso de estratificación, existencia de early stopping o regularización adicional y detalles de la validación interna. Las curvas de las Figuras 10 y 11 del artículo tienen leyendas "Test Accuracy" y "Test Accuracy Loss", lo que sugiere que el artículo pudo haber usado el conjunto de prueba como validación durante el entrenamiento; esto es una interpretación de las leyendas y no una declaración explícita. En nuestro trabajo la validación proviene solo del conjunto de entrenamiento.

## 8. Experimento A: 1025 registros

**Nuestro trabajo.** Notebook 04. Se conservaron los 1025 registros, con sus duplicados, por ser el escenario descrito por el artículo.

| Elemento | Valor |
|---|---|
| Registros | 1025 |
| División | 70 / 30, `random_state=42`, `stratify=y` |
| Entrenamiento | 717 registros (349 de la clase 0 y 368 de la clase 1) |
| Prueba | 308 registros (150 de la clase 0 y 158 de la clase 1) |
| Arquitectura | 1D-CNN de la Figura 7, 285,873 parámetros, `padding="valid"` |
| Entrenamiento | 150 épocas, Adam 0.001, `binary_crossentropy`, batch size 32, validación interna del 20 % |
| Escalado | `MinMaxScaler` ajustado solo con entrenamiento |

Resultados sobre el conjunto de prueba (umbral 0.5):

| Métrica | Valor |
|---|---:|
| Accuracy | 97.40 % |
| Precision | 98.08 % |
| Sensibilidad (recall) | 96.84 % |
| Especificidad | 98.00 % |
| F1-score | 97.45 % |
| AUC-ROC | 0.9982 |
| Loss de prueba | 0.0782 |

Matriz de confusión (clase 1 como positiva):

| | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | 147 (TN) | 3 (FP) |
| Real 1 | 5 (FN) | 153 (TP) |

En entrenamiento, la última época alcanzó accuracy de 0.9930 (entrenamiento) y 0.9931 (validación interna). Las curvas no mostraron señales claras de sobreajuste.

## 9. Problema de duplicados

| Dato | Valor |
|---|---:|
| Registros originales | 1025 |
| Duplicados exactos | 723 |
| Filas distintas | 302 |
| Veces que aparece cada fila distinta | Entre 3 y 8 (promedio 3.39) |
| Filas de prueba con copia exacta en entrenamiento (Experimento A) | 286 de 308 |
| Porcentaje | 92.9 % |

El artículo no menciona los duplicados. Con los duplicados dentro del dataset y una división aleatoria, la mayoría de los registros de prueba tiene una copia idéntica en entrenamiento. Esto puede producir una **estimación optimista** del rendimiento, porque el modelo se evalúa con datos que ya vio. Las métricas del Experimento A deben leerse con esa cautela.

Los duplicados **no son necesariamente la única causa** de las diferencias entre nuestros resultados y los del artículo, ni de las diferencias entre los Experimentos A y B. Intervienen también el tamaño de los conjuntos, la partición elegida, los hiperparámetros no especificados y las discrepancias de arquitectura. Los datos disponibles no permiten aislar el efecto de cada factor.

## 10. Experimento B: sin duplicados

**Nuestro trabajo.** Notebook 05. Se eliminaron los duplicados únicamente en memoria; `data/heart.csv` no se modificó. La arquitectura, los hiperparámetros, el escalado, la semilla y el umbral son los mismos del Experimento A (se verificó que el modelo tiene los mismos 285,873 parámetros). La única diferencia metodológica es el tratamiento de los duplicados.

| Elemento | Valor |
|---|---|
| Registros únicos | 302 (138 de la clase 0 y 164 de la clase 1) |
| División | 70 / 30, `random_state=42`, `stratify=y` |
| Entrenamiento | 211 registros (96 y 115 por clase) |
| Prueba | 91 registros (42 y 49 por clase) |
| Filas de prueba presentes en entrenamiento | 0 de 91 |

Resultados sobre el conjunto de prueba (umbral 0.5):

| Métrica | Valor |
|---|---:|
| Accuracy | 81.32 % |
| Precision | 82.00 % |
| Sensibilidad (recall) | 83.67 % |
| Especificidad | 78.57 % |
| F1-score | 82.83 % |
| AUC-ROC | 0.8537 |
| Loss de prueba | 1.6145 |

Matriz de confusión: TN 33, FP 9, FN 8, TP 41 (17 errores en 91 registros).

En entrenamiento, la accuracy llegó a 0.9881 frente a 0.7674 en validación, y la loss de validación aumentó después de la época 54, lo que es compatible con sobreajuste.

### Comparación entre A y B

Los valores del Experimento A y B provienen de `resultados/tablas/comparacion_experimentos.csv`. La diferencia se calcula como B menos A, en puntos porcentuales (pp).

| Métrica | Experimento A: 1025 | Experimento B: 302 | Diferencia |
|---|---:|---:|---:|
| Accuracy | 97.40 % | 81.32 % | -16.08 pp |
| Precision | 98.08 % | 82.00 % | -16.08 pp |
| Sensibilidad | 96.84 % | 83.67 % | -13.17 pp |
| Especificidad | 98.00 % | 78.57 % | -19.43 pp |
| F1-score | 97.45 % | 82.83 % | -14.62 pp |
| AUC-ROC | 99.82 % | 85.37 % | -14.45 pp |

Todas las métricas disminuyeron en el Experimento B. Los resultados sugieren que la presencia de copias exactas entre entrenamiento y prueba contribuía al rendimiento observado en el Experimento A. Sin embargo, B también tiene menos datos de entrenamiento (211 frente a 717) y un conjunto de prueba mucho menor (91 frente a 308), donde un solo error equivale a 1.10 pp de accuracy frente a 0.32 pp en A. Este diseño no permite separar el efecto de los duplicados del efecto del tamaño de los datos. Una sola partición y una sola semilla tampoco permiten medir la variabilidad.

## 11. Comparación con el artículo

| Escenario | Artículo | Nuestro trabajo |
|---|---:|---:|
| Train/Test (accuracy) | 99.95 % | Experimento A: 97.40 %; Experimento B: 81.32 % |
| K-Fold, K = 5 (accuracy) | 98.53 % (Tabla 4 y sección 4.2) o 99.12 % (Conclusiones) | Pendiente |
| K-Fold, K = 10 (accuracy) | 99.12 % (Tabla 4 y sección 4.2) o 98.53 % (Conclusiones) | Pendiente |

Los escenarios no son completamente equivalentes, por estas razones:

- **Train/Test.** El Experimento A es el más cercano al escenario del artículo (1025 registros, 70/30, misma arquitectura de la Figura 7), pero no es una réplica exacta: hiperparámetros no especificados, semilla y estratificación elegidas por el proyecto y posibles diferencias de implementación. El Experimento B cambia el dataset (302 registros), por lo que no es comparable directamente con el artículo.
- **Reporte del artículo.** La matriz de confusión de la Figura 12 del artículo suma 308 registros (150, 0, 3 y 155), lo que implica una accuracy de 99.03 %, no de 99.95 %. Además, la fila de prueba de la Tabla 3 muestra "soporte" 158 y la de entrenamiento 150; esos valores coinciden con los conteos por clase del conjunto de prueba de 308 registros, por lo que el significado de esas filas no es claro. Además, si el conjunto de prueba del artículo tuviera los mismos conteos por clase que el nuestro (150 y 158), la etiqueta "Heart Disease" de la Figura 12 correspondería a la clase 0, contraria a la codificación de la Tabla 1; esta auditoría no puede determinarlo con certeza.
- **Valores de K.** La Tabla 4 y la sección 4.2 asignan 98.53 % a K = 5 y 99.12 % a K = 10, pero las Conclusiones los intercambian. La Figura 9 ilustra 10 particiones con 80 % entrenamiento, 10 % prueba y 10 % validación, un esquema distinto al K-Fold estándar.
- **Otras inconsistencias numéricas del artículo.** Para K = 5, la Tabla 4 da precision 99 % y recall 99.92 %, valores a partir de los cuales el F1-score sería aproximadamente 99.46 %, mientras que la tabla reporta 99.91 %. La ecuación (5) del artículo, tal como se imprime, omite el factor 2 de la fórmula del F1-score.
- **Métricas que el artículo no reporta.** Especificidad y AUC-ROC no aparecen en el artículo, por lo que no pueden compararse.

Con estos elementos no puede afirmarse que el modelo del proyecto sea mejor o peor que el del artículo; las diferencias de cifras no son interpretables sin resolver las condiciones experimentales.

## 12. Qué hemos reproducido

- Dataset de 1025 registros y 14 columnas, con 13 predictoras y `target` (Notebook 01).
- Reescalado de todas las variables a [0, 1] con la fórmula Min-Max (Notebook 02).
- División 70 % / 30 % (Notebook 02).
- Entrada de la red con forma (13, 1) (Notebook 03).
- Arquitectura 1D-CNN de la Figura 7: cuatro bloques Conv1D (32, 64, 128 y 128 filtros, kernel 2, sigmoid) con Batch Normalization, Dropout 0.2 y 0.5, Flatten, Dense(200) con ReLU, Dropout 0.1 y salida sigmoid (Notebook 03).
- Entrenamiento de 150 épocas con los 1025 registros (Notebook 04).
- Evaluación con accuracy, precision, recall y F1-score, matriz de confusión y curvas de accuracy y loss (Notebook 04).
- Un experimento adicional, que el artículo no contempla, para estudiar el efecto de los duplicados (Notebook 05).

Se trata de reproducciones **parciales o aproximadas**, porque varios hiperparámetros no están especificados y la arquitectura del artículo tiene inconsistencias internas.

## 13. Qué no hemos reproducido todavía

- **K-Fold con K = 5**: no ejecutado.
- **K-Fold con K = 10**: no ejecutado.
- Matrices de confusión por K-Fold y curvas de accuracy y loss por pliegue (Figuras 13, 14 y 15 del artículo).
- **Ranking Chi-cuadrado** y la Figura 6 (cp, ca, oldpeak, thalach): no calculado.
- **One-Hot Encoding**: no aplicado por ser incompatible con la entrada (13, 1); el artículo lo describe, y su efecto no ha sido evaluado.
- **Variante de arquitectura de la Tabla 2** (Flatten de 1280 y parámetros de las convoluciones 2 y 3): no reproducida; solo se siguió la Figura 7.
- **Esquema de validación del artículo**: no está claro si el artículo usó el conjunto de prueba como validación (leyendas "Test" en las Figuras 10 y 11) ni cómo definió las particiones de la Figura 9; no se ha replicado.
- **Tratamiento de outliers** y detalles de limpieza que el artículo menciona sin describir.
- **Procedencia del archivo** de 1025 filas: pendiente de documentar.
- **Diferencias metodológicas relevantes ya identificadas:** hiperparámetros definidos por el proyecto (sección 7.2), escalador ajustado solo con entrenamiento, semilla y estratificación propias y una sola partición evaluada.

## 14. Conclusión de la auditoría

El proyecto tiene una **reproducción parcial** del método del artículo: mismo dataset en tamaño y estructura, mismo escalado, misma proporción de división, misma entrada y una arquitectura basada en la Figura 7, entrenada 150 épocas. Los resultados obtenidos (Experimento A: 97.40 % de accuracy; Experimento B: 81.32 %) deben interpretarse junto con las limitaciones documentadas: duplicados entre entrenamiento y prueba en el Experimento A, hiperparámetros definidos por el proyecto, inconsistencias entre la Figura 7, la Tabla 2 y el texto del artículo, e inconsistencias en las métricas reportadas por el artículo.

Todavía **no debe presentarse como una replicación completa**. Para poder hacerlo falta, como mínimo, ejecutar y documentar K-Fold con K = 5 y K = 10, calcular el ranking Chi-cuadrado y resolver, o declarar explícitamente, las diferencias de arquitectura y de esquema de validación señaladas en este documento.
