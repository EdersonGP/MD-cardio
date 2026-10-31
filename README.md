# Cardio: clasificación de enfermedad cardiovascular

La única secuencia metodológica principal está en `notebooks/nuevo_experimento/`,
numerada del 01 al 12. Usa `data/nuevo_dataset/cardio_train.csv`: 70,000 registros,
13 columnas, 11 predictoras y objetivo `cardio`. El dataset anterior de 1025
registros se conserva como antecedente histórico y no forma parte del experimento
principal.

## Secuencia definitiva

| Notebook | Responsabilidad |
|---|---|
| 01_exploracion_dataset | Estructura, diccionario de variables y exploración |
| 02_auditoria_calidad | Nulos, duplicados, extremos y reglas operacionales |
| 03_limpieza_preprocesamiento | Verificación de 70,000 → 68,615 y representación de variables |
| 04_preparacion_datos | Muestra existente, particiones y ejemplo de escalamiento sin fuga |
| 05_diseno_entrenamiento_cnn | Construcción explícita TensorFlow/Keras e historial existente |
| 06_evaluacion_inicial | Resultados iniciales y evidencia definitiva diferenciados |
| 07_auditoria_experimental | Diferencias de protocolos y limitaciones |
| 08_nueva_arquitectura_cnn | Arquitectura definitiva, cálculo de parámetros y diagrama desde código |
| 09_evaluacion_comparacion | Comparación descriptiva de ejecuciones guardadas |
| 10_importancia_variables | Importancia existente del árbol y gráfico generado desde su tabla |
| 11_verificacion_arquitectura | Verificación del modelo guardado, configuración y parámetros |
| 12_analisis_final | Integración automática de archivos, sin transcripción de métricas |

## Modelo y resultados conservados

TensorFlow/Keras: entrada `(11,1)` → Conv1D(32, kernel 3, ReLU, same) →
Conv1D(96, kernel 3, ReLU, same) → GlobalAveragePooling1D → Dense(256, ReLU) →
Dropout(0.30) → Dense(1, sigmoid). **34,529 parámetros**, dentro de 30,000–50,000.

El artefacto entrenado es `resultados/nuevo_experimento/modelo_nueva_cnn.keras`.
La muestra experimental tiene 15,000 registros; no todos se usaron para ajustar
pesos: el protocolo definitivo utilizó 10,200 para entrenamiento, 2,550 para
validación y 2,250 para prueba.

La fuente de métricas definitivas es `tablas/metricas_evaluacion.csv`:
Accuracy 0.7311, Precision 0.7326, Recall 0.7188, F1 0.7256 y ROC-AUC 0.7885
(valores mostrados redondeados). Las predicciones guardadas permiten verificarlas
sin entrenamiento ni inferencia.

## Dos ejecuciones, una arquitectura

La inicial usó StandardScaler, división explícita 70/15/15 y 44 épocas.
La definitiva usó MinMaxScaler, 15 % de prueba y validación interna del 20 %
del bloque restante, con 66 épocas. Ambas tienen la misma arquitectura.
Las pruebas se generaron mediante procedimientos distintos; la comparación no
aísla el efecto del escalador. La validación definitiva participó en el ajuste
del escalador antes de su reserva; la prueba no. El procedimiento sin fuga del
notebook 04 es una recomendación documentada y no produjo las métricas existentes.

La importancia de variables proviene de un `DecisionTreeClassifier(max_depth=5)`
independiente, con división 80/20. No es una atribución interna de la CNN.

## Preservación

- `notebooks/historico_dataset_1025/`: seis cuadernos históricos y copias originales.
- `notebooks/respaldo_nuevo_experimento_original/`: seis cuadernos nuevos originales,
  con todas sus salidas conservadas byte a byte.
- `docs/respaldo_reorganizacion.json`: hashes previos y rutas de respaldo.
- Todos los datos, tablas, figuras, modelos y predicciones originales se conservan.
- Los generadores `crear_notebook_*.py` de la raíz se conservan como originales.
  **No ejecutarlos sobre la secuencia reorganizada:** sus destinos y contenidos
  corresponden a la numeración anterior y pueden sobrescribir cuadernos actuales.
- `scripts/reorganizar_experimento.py` realizó una migración documental única y
  se niega a repetirla si ya existe el manifiesto de respaldo.

## Uso y comprobaciones

Seleccionar un kernel con las dependencias de `requirements.txt`, en particular
TensorFlow/Keras, NumPy, pandas, scikit-learn y matplotlib. Los notebooks resuelven
la raíz desde la ubicación del proyecto; no dependen del estado de otro cuaderno.
Los notebooks 05, 08 y 11 requieren TensorFlow para construir o cargar el modelo,
pero **no lo entrenan**.

El notebook 10 define, sin invocarla, la función que documenta el ajuste original
del árbol. Sus celdas ejecutadas leen exclusivamente la importancia ya guardada.
Las nuevas figuras se escriben en `resultados/nuevo_experimento/figuras/reorganizacion/`;
no reemplazan las figuras originales. El notebook 03 solo genera el CSV limpio
si falta; si existe, comprueba su contenido sin sobrescribirlo.

Para comprobar respaldo, integridad, arquitectura serializada, métricas y seguridad
de los cuadernos, sin entrenar:

```powershell
python scripts/verificar_reorganizacion.py
```

La validación no reinstala dependencias, no modifica el entorno virtual y no
requiere ejecutar los notebooks históricos. Véase `docs/metodologia_definitiva.md`.
