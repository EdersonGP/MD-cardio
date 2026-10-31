"""Verificar integridad y resultados sin entrenar modelos.

--ejecutar-celdas ejecuta la secuencia documental y guarda sus salidas nuevas.
Los ajustes de cualquier DecisionTreeClassifier o modelo TensorFlow se bloquean.
No se modifica ningún artefacto original listado en el manifiesto.
"""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".jupyter_runtime/matplotlib"))
NOMBRES = [
    "01_exploracion_dataset", "02_auditoria_calidad", "03_limpieza_preprocesamiento",
    "04_preparacion_datos", "05_diseno_entrenamiento_cnn", "06_evaluacion_inicial",
    "07_auditoria_experimental", "08_nueva_arquitectura_cnn", "09_evaluacion_comparacion",
    "10_importancia_variables", "11_verificacion_arquitectura", "12_analisis_final",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integridad():
    manifest = json.loads((ROOT / "docs/respaldo_reorganizacion.json").read_text(encoding="utf-8"))
    for rel, digest in manifest["archivos_conservados"].items():
        assert sha(ROOT / rel) == digest, f"Artefacto original alterado: {rel}"
    for rel, info in manifest["notebooks_originales"].items():
        assert sha(ROOT / info["respaldo"]) == info["sha256"], f"Respaldo alterado: {rel}"
        if "/nuevo_experimento/" not in rel:
            historico = ROOT / "notebooks/historico_dataset_1025" / Path(rel).name
            assert sha(historico) == info["sha256"]
    return {"artefactos_originales_intactos": len(manifest["archivos_conservados"]),
            "notebooks_respaldados_intactos": len(manifest["notebooks_originales"])}


def verificar_notebooks():
    import nbformat
    folder = ROOT / "notebooks/nuevo_experimento"
    assert sorted(p.stem for p in folder.glob("*.ipynb")) == NOMBRES
    for nombre in NOMBRES:
        nb = nbformat.read(folder / f"{nombre}.ipynb", as_version=4)
        nbformat.validate(nb)
        assert nb.metadata.cardio.dataset == "cardio_train.csv"
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            assert "heart.csv" not in cell.source
            assert "resultados/csv" not in cell.source
            tree = ast.parse(cell.source)
            compile(tree, f"{nombre}.ipynb", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"fit_generator", "train_on_batch", "fit_predict", "fit_transform_model"}
                    if node.func.attr == "fit":
                        # Escalador permitido. Ajuste del árbol permitido solo como función NO invocada.
                        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                        if owner != "scaler" and not (isinstance(node.func.value, ast.Call)
                                and isinstance(node.func.value.func, ast.Name)
                                and node.func.value.func.id == "MinMaxScaler"):
                            assert nombre.startswith("10_") and owner == "tree"
                            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
                            assert fn.name == "procedimiento_original_importancia_no_ejecutado"
                            assert node in list(ast.walk(fn))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "procedimiento_original_importancia_no_ejecutado"
    return {"notebooks_validos": len(NOMBRES), "dependencias_dataset_anterior": 0,
            "llamadas_activas_entrenamiento_modelos": 0}


def ejecutar_documentacion():
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from sklearn.tree import DecisionTreeClassifier
    from src.experimento import tensorflow_disponible

    def prohibido(*args, **kwargs):
        raise RuntimeError("Entrenamiento bloqueado durante verificación documental")

    original_fit = DecisionTreeClassifier.fit
    DecisionTreeClassifier.fit = prohibido
    tf_model = tf_fit = None
    if tensorflow_disponible():
        import tensorflow as tf
        tf_model, tf_fit = tf.keras.Model, tf.keras.Model.fit
        tf_model.fit = prohibido
    InteractiveShell.instance()
    ejecutadas = {}
    try:
        for nombre in NOMBRES:
            path = ROOT / "notebooks/nuevo_experimento" / f"{nombre}.ipynb"
            nb = nbformat.read(path, as_version=4)
            namespace = {"__name__": "__main__"}
            count = 0
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                plt.close("all")
                with capture_output() as captured:
                    exec(compile(cell.source, f"{nombre}:celda-{count}", "exec"), namespace)
                outputs = []
                for stream, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                    if value:
                        outputs.append(nbformat.v4.new_output("stream", name=stream, text=value))
                for display in captured.outputs:
                    outputs.append(nbformat.v4.new_output("display_data", data=display.data,
                                                         metadata=display.metadata))
                for num in plt.get_fignums():
                    buffer = io.BytesIO()
                    plt.figure(num).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                    outputs.append(nbformat.v4.new_output("display_data", data={
                        "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        "text/plain": "Figura generada por la celda de código"}, metadata={}))
                plt.close("all")
                cell.outputs = outputs
                cell.execution_count = count
            nb.metadata["verificacion_documental"] = {
                "entrenamientos": 0, "tensorflow_disponible": tensorflow_disponible(),
                "metodo": "Ejecución secuencial de celdas Python; ajuste de modelos bloqueado",
                "nota": "Si falta TensorFlow se verifica configuración serializada; no se afirma carga de pesos con Keras."}
            nbformat.validate(nb)
            nbformat.write(nb, path)
            ejecutadas[nombre] = count
            print(f"OK {nombre}: {count} celdas de código", flush=True)
    finally:
        DecisionTreeClassifier.fit = original_fit
        if tf_model is not None:
            tf_model.fit = tf_fit
    return ejecutadas


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ejecutar-celdas", action="store_true")
    args = parser.parse_args()
    report = {**integridad(), **verificar_notebooks()}
    from src.experimento import (ORIGINAL, RESULTADOS, cargar_datos, reglas_calidad,
        verificar_configuracion, verificar_predicciones, resumen_dataset,
        comparar_ejecuciones, importancia_guardada, dibujar_arquitectura, guardar_figura_nueva,
        tensorflow_disponible)
    import numpy as np
    import pandas as pd
    original = cargar_datos(ORIGINAL)
    limpio = cargar_datos(RESULTADOS / "cardio_limpio.csv")
    muestra = cargar_datos(RESULTADOS / "cardio_15000.csv")
    assert original.shape == (70000, 13) and original.duplicated().sum() == 0
    esperado = original.loc[~reglas_calidad(original).any(axis=1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(limpio, esperado, check_dtype=False)
    assert len(limpio) == 68615 and len(muestra) == 15000
    pd.testing.assert_frame_equal(muestra.set_index("id").sort_index(),
        limpio.set_index("id").loc[muestra.id].sort_index(), check_dtype=False)
    arquitectura = verificar_configuracion()
    metricas, matriz = verificar_predicciones()
    assert [round(v, 4) for v in metricas.Valor] == [.7311, .7326, .7188, .7256, .7885]
    importancia = importancia_guardada()
    assert len(importancia) == 11 and np.isclose(importancia.importancia.sum(), 1)
    report.update({"dataset_original": len(original), "dataset_limpio": len(limpio),
                   "muestra": len(muestra), "parametros_configuracion_guardada": int(arquitectura.parametros.sum()),
                   "metricas_desde_predicciones": dict(zip(metricas.Metrica, metricas.Valor)),
                   "matriz_confusion": matriz.tolist(), "entrenamientos_ejecutados": 0,
                   "tensorflow_disponible": tensorflow_disponible(),
                   "alcance_arquitectura": "Configuración serializada y tabla original; carga con Keras solo si TensorFlow está disponible"})
    if args.ejecutar_celdas:
        report["celdas_ejecutadas"] = ejecutar_documentacion()
    # Productos nuevos, separados de todas las fuentes originales.
    destino = RESULTADOS / "tablas/reorganizacion"
    destino.mkdir(parents=True, exist_ok=True)
    comparar_ejecuciones().to_csv(destino / "comparacion_protocolos.csv", index=False)
    resumen_dataset().to_csv(destino / "resumen_dataset_verificado.csv", index=False)
    reglas_calidad(original).sum().rename("incumplen").to_csv(destino / "auditoria_criterios.csv")
    arquitectura.to_csv(destino / "arquitectura_verificada.csv", index=False)
    fig = dibujar_arquitectura(arquitectura)
    guardar_figura_nueva(fig, "arquitectura_definitiva_desde_codigo.png")
    report.update(integridad())
    verificar_notebooks()
    (ROOT / "docs/verificacion_reorganizacion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
