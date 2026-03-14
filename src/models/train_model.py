"""
train_model.py
==============
Módulo de entrenamiento para el proyecto Student Success Classification.

Responsabilidad única:
    Tomar el dataset procesado, dividirlo en sets de entrenamiento
    y test, entrenar el modelo de clasificación y guardarlo en disco.

¿Qué hace este módulo?
    1. Separa el DataFrame en features (X) y variable objetivo (y)
    2. Divide los datos en train/test con proporciones configurables
    3. Entrena un RandomForestClassifier con hiperparámetros del config
    4. Guarda el modelo entrenado en disco (formato .joblib)

¿Qué NO hace este módulo?
    - No calcula métricas de evaluación (eso es evaluate_model.py)
    - No genera visualizaciones
    - No hace predicciones sobre datos nuevos (eso es predict.py)
    - No carga ni procesa datos

¿Por qué Random Forest como modelo base?
    - Robusto: resiste bien el overfitting con múltiples árboles
    - Interpretable: ofrece feature_importances_ (qué variables importan)
    - Sin escalado: no necesita normalizar los datos previamente
    - Versátil: funciona bien con datasets mixtos (numéricos + categóricos)
    Alternativas evaluadas:
    - Regresión Logística: más simple e interpretable, pero asume
      relaciones lineales — puede no capturar patrones complejos
    - XGBoost: más preciso en competiciones, pero más complejo
      de configurar y menos interpretable para un portfolio

Flujo de datos:
    data/processed/ → [train_model.py] → models/random_forest_model.joblib
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def split_features_target(df: pd.DataFrame, target_column: str):
    """
    Separa el DataFrame en features (X) y variable objetivo (y).

    ¿Qué son features y target?
        - Features (X): las columnas que el modelo usa para predecir.
          Ejemplo: notas del primer semestre, edad, beca...
        - Target (y): la columna que queremos predecir.
          Ejemplo: 0 = Graduado, 1 = Abandono, 2 = Matriculado

    ¿Por qué esta separación es necesaria?
        scikit-learn espera siempre dos argumentos separados en .fit(X, y).
        Nunca entrena con X e y juntos en un mismo DataFrame.

    Args:
        df (pd.DataFrame): Dataset completo con features y target.
        target_column (str): Nombre de la columna objetivo.
                             Ejemplo: "target_encoded"
                             (leído desde config.yaml)

    Returns:
        tuple: (X, y) donde:
               - X (pd.DataFrame): Features del modelo
               - y (pd.Series): Variable objetivo
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"   ✅ Features (X): {X.shape[1]} columnas, {X.shape[0]} filas")
    print(f"   ✅ Target (y): '{target_column}' — valores únicos: {sorted(y.unique())}")

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """
    Divide los datos en conjuntos de entrenamiento y evaluación.

    ¿Por qué dividir los datos?
        Si entrenamos y evaluamos el modelo con los mismos datos,
        es como darle al estudiante las respuestas del examen antes
        del examen. El modelo "memoriza" en lugar de "aprender".
        La división garantiza que la evaluación sea honesta.

    ¿Por qué stratify=y?
        Asegura que la proporción de cada clase (Graduate, Dropout,
        Enrolled) sea la misma en train y en test.
        Sin stratify, podría ocurrir que el test tuviera muchos
        Dropouts y pocos Graduates, dando métricas engañosas.
        Con datasets desbalanceados, stratify es esencial.

    Args:
        X (pd.DataFrame): Features del dataset completo.
        y (pd.Series): Variable objetivo del dataset completo.
        test_size (float): Proporción de datos para el test.
                           Ejemplo: 0.2 → 80% train, 20% test
                           (leído desde config.yaml)
        random_state (int): Semilla para reproducibilidad.
                            Con el mismo valor, la división siempre
                            es idéntica, permitiendo comparar experimentos.
                            (leído desde config.yaml)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # misma proporción de clases en train y test
    )

    print(f"   ✅ Train: {X_train.shape[0]} filas ({(1 - test_size):.0%})")
    print(f"   ✅ Test:  {X_test.shape[0]} filas ({test_size:.0%})")

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth,
    random_state: int
) -> RandomForestClassifier:
    """
    Entrena un modelo RandomForestClassifier con los datos de entrenamiento.

    ¿Qué es un Random Forest?
        Un ensemble (conjunto) de árboles de decisión.
        Cada árbol aprende de una muestra aleatoria de los datos.
        La predicción final es la votación mayoritaria de todos los árboles.
        Más árboles = más estable, pero más lento de entrenar.

    ¿Por qué n_estimators=100 como punto de partida?
        Es el valor por defecto recomendado en la literatura. Con 100
        árboles el modelo ya es estable. A partir de ~200, la mejora
        marginal es mínima y el coste computacional sube.

    ¿Qué significa max_depth=None?
        Los árboles crecen sin límite de profundidad hasta que cada
        hoja tiene muestras de una sola clase (hoja "pura").
        Esto puede causar overfitting. Si el modelo memoriza demasiado
        los datos de entrenamiento, se puede limitar max_depth (ej: 10).

    Args:
        X_train (pd.DataFrame): Features del conjunto de entrenamiento.
        y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
        n_estimators (int): Número de árboles en el bosque.
                            (leído desde config.yaml)
        max_depth (int | None): Profundidad máxima de cada árbol.
                                None = sin límite.
                                (leído desde config.yaml)
        random_state (int): Semilla de aleatoriedad para reproducibilidad.
                            (leído desde config.yaml)

    Returns:
        RandomForestClassifier: Modelo entrenado, listo para predecir.
    """
    print(f"   🌲 Entrenando Random Forest ({n_estimators} árboles)...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    print("   ✅ Entrenamiento completado.")
    return model


def save_model(model: RandomForestClassifier, config: dict) -> str:
    """
    Guarda el modelo entrenado en disco usando el formato joblib.

    ¿Por qué guardar el modelo?
        El entrenamiento puede tardar horas con datasets grandes.
        Guardar el modelo permite reutilizarlo sin re-entrenar:
        - Para hacer predicciones en producción
        - Para comparar distintas versiones del modelo
        - Para compartirlo con el equipo sin compartir el código

    ¿Por qué joblib y no pickle?
        Ambos serializan objetos Python, pero joblib es más eficiente
        con arrays numéricos grandes (como los de scikit-learn).
        Es la opción recomendada por la propia documentación de sklearn.

    ¿Por qué .joblib y no .pkl?
        Es solo una convención de extensión. El contenido es igual,
        pero .joblib deja claro que el archivo fue generado con joblib.

    Args:
        model (RandomForestClassifier): Modelo entrenado a guardar.
        config (dict): Diccionario de configuración del proyecto.
                       Se esperan las claves:
                       - config["paths"]["models"] (str)
                       - config["paths"]["model_filename"] (str)

    Returns:
        str: Ruta completa donde se guardó el modelo.
    """
    models_dir     = config["paths"]["models"]
    model_filename = config["paths"]["model_filename"]
    model_path     = os.path.join(models_dir, model_filename)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"   💾 Modelo guardado en: {model_path}")
    return model_path


def train_model(df: pd.DataFrame, config: dict):
    """
    Función principal de entrenamiento. Orquesta todo el proceso.

    Ejecuta en orden:
        1. Separar features (X) y target (y)
        2. Dividir en train/test
        3. Entrenar el modelo Random Forest
        4. Guardar el modelo en disco

    Returns también los conjuntos de test (X_test, y_test) para que
    el pipeline pueda pasarlos inmediatamente a evaluate_model.py
    sin necesidad de volver a leer datos del disco.

    Args:
        df (pd.DataFrame): Dataset procesado (desde data/processed/).
        config (dict): Diccionario de configuración del proyecto.

    Returns:
        tuple: (model, X_test, y_test) donde:
               - model: el modelo RandomForest entrenado
               - X_test (pd.DataFrame): features del conjunto de evaluación
               - y_test (pd.Series): etiquetas reales del conjunto de evaluación
    """
    print("🤖 Iniciando entrenamiento del modelo...")

    target_column = config["training"]["target_column"]
    test_size     = config["training"]["test_size"]
    n_estimators  = config["model"]["n_estimators"]
    max_depth     = config["model"]["max_depth"]
    random_state  = config["model"]["random_state"]
    train_rs      = config["training"]["random_state"]

    X, y = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, train_rs)
    model = train_random_forest(X_train, y_train, n_estimators, max_depth, random_state)
    save_model(model, config)

    print("✅ Modelo entrenado y guardado correctamente.")
    return model, X_test, y_test
