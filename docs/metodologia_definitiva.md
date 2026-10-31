# Organización y trazabilidad del experimento definitivo

## Fuentes y traslado de contenido

Los nuevos 01–03 distribuyen exploración, auditoría y limpieza del antiguo 07.
El nuevo 04 combina la muestra existente del antiguo 07 con preparación de los
antiguos 08/09, diferenciando los procedimientos realmente ejecutados de la
recomendación sin fuga. Los nuevos 05/06 documentan arquitectura, historiales y
evaluaciones existentes. El nuevo 07 audita las diferencias. El nuevo 08 explica
la arquitectura definitiva con código y diagrama. El nuevo 09 compara tablas
existentes. Los nuevos 10/11 mantienen importancia y verificación, y el 12 integra
artefactos automáticamente.

Esta es una reorganización documental de experimentos ya realizados. No implica
que la auditoría precediera cronológicamente a todos los entrenamientos.

## Contrato de artefactos originales

Rutas relativas a `resultados/nuevo_experimento/`:

| Artefacto | Procedencia y función |
|---|---|
| cardio_limpio.csv | Filtrado operacional; 68,615 registros; edad en días e id conservados |
| cardio_15000.csv | Muestra estratificada existente; no se vuelve a seleccionar |
| tablas/metricas_nueva_cnn.csv | Evaluación inicial, StandardScaler |
| tablas/historial_nueva_cnn.csv | 44 épocas de la ejecución inicial |
| tablas/resumen_nueva_cnn.csv | Tamaños y configuración de la ejecución inicial |
| figuras/*_nueva_cnn.png | Evaluación inicial, salvo arquitectura_nueva_cnn.png, que documenta el modelo final |
| tablas/metricas_evaluacion.csv | Fuente de precisión completa para las métricas definitivas |
| tablas/historial_entrenamiento.csv | 66 épocas de la ejecución definitiva |
| classification_report.txt | Reporte definitivo; 2,250 observaciones de prueba |
| y_test.npy, y_prob.npy, y_pred.npy | Verificación de métricas definitivas sin volver a entrenar ni predecir |
| modelo_nueva_cnn.keras | Modelo entrenado definitivo; config.json y pesos preservados |
| tablas/arquitectura_capas.csv | Arquitectura extraída originalmente del modelo final |
| tablas/importancia_variables.csv | Importancia por impureza del árbol independiente |
| tablas/*final*.csv y metricas_finales_resumen.csv | Resúmenes históricos, algunos redondeados o transcritos; no sustituyen las fuentes anteriores |

Los nuevos cuadernos leen las fuentes de precisión completa. Los resúmenes
originales se preservan aunque no sean la fuente principal de sus conclusiones.
El reporte inicial está preservado en las salidas del notebook 08 original.

## Limpieza y representación

Reglas operacionales inclusivas: edad 18–100 años; altura 120–220 cm; peso
30–250 kg; presión sistólica 70–250; diastólica 40–200; relación sistólica >
diastólica. Se excluye la unión de incumplimientos: 1,385 registros. Estos rangos
no se presentan como criterios médicos universales. La edad se convierte mediante
`age / 365.25` en copias de modelado, sin cambiar los CSV conservados.

Orden de predictoras: age, gender, height, weight, ap_hi, ap_lo, cholesterol,
gluc, smoke, alco, active. `id` no es predictor y `cardio` es el objetivo.
La CNN trata las variables tabulares como secuencia de características; no se
afirma que constituyan una serie temporal. Las categorías codificadas se mantienen
para respetar el modelo ya evaluado.

## Limitaciones conservadas

- StandardScaler y MinMaxScaler no se compararon bajo un único protocolo controlado.
- La validación interna definitiva influyó en el escalamiento; no se corrige
  retrospectivamente ni se sustituyen las métricas originales.
- El ejemplo sin fuga separa todos los conjuntos antes de ajustar el escalador;
  no tiene resultados nuevos asociados.
- Identificadores únicos no descartan patrones repetidos sin id. El notebook 04
  informa coincidencias entre subconjuntos sin eliminarlas automáticamente.
- No se serializó el escalador definitivo ni un inventario explícito de las
  particiones originales.
- No hay validación externa ni repetición entre semillas para estimar variabilidad.
- Las importancias del árbol no explican directamente los pesos o decisiones de la CNN.
- La auditoría del artículo conservada en resultados/auditoria pertenece al
  experimento anterior; sus pendientes no son el estado actual de esta secuencia.

## Garantías de la reorganización

Antes del primer movimiento se copiaron y verificaron todos los notebooks.
El manifiesto almacena hashes de datasets, resultados y generadores originales.
Los cuadernos históricos se movieron conservando sus bytes. Los nuevos cuadernos
no ejecutan ajuste de modelos. La construcción de la CNN y la lectura del modelo
son operaciones diferentes de entrenar. El único código de ajuste de árbol está
encapsulado como referencia no invocada en el notebook 10.
