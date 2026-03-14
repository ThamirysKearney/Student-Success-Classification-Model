"""
feature_engineering.py
======================
Módulo de ingeniería de variables para el proyecto Student Success Classification.

Responsabilidad única:
    Transformar el dataset limpio (interim) en un dataset enriquecido
    con nuevas variables, listo para ser usado por el modelo de ML.

¿Qué hace este módulo?
    1. Crea grupos de edad (binning de variable continua → categórica)
    2. Codifica la variable objetivo en valores numéricos (Label Encoding)
    3. Aplica One-Hot Encoding a variables categóricas
    4. Elimina columnas originales que ya no se necesitan
    5. Guarda el resultado en data/processed/

¿Qué NO hace este módulo?
    - No genera visualizaciones (eso va en los notebooks)
    - No entrena modelos
    - No carga datos (eso hace load_data.py)
    - No limpia datos (eso hace preprocess.py)

¿Qué es Feature Engineering?
    Es el arte de crear nuevas columnas a partir de las existentes
    para que el modelo de ML pueda aprender mejor.
    Ejemplo: la edad exacta (21 años) puede ser menos informativa
    que saber si el estudiante es "adulto" o "tradicional".

Flujo de datos:
    data/interim/ → [feature_engineering.py] → data/processed/
"""

import os
import pandas as pd


def create_age_groups(df: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    """
    Crea una nueva variable categórica de grupos de edad a partir de la edad exacta.

    Técnica utilizada: Binning (también llamado Discretización).
    Consiste en convertir una variable numérica continua en intervalos.

    ¿Por qué hacer esto?
        Los modelos de ML a veces capturan mejor los patrones cuando
        trabajamos con rangos en lugar de valores exactos.
        Por ejemplo, el riesgo de abandono puede ser parecido para
        estudiantes de 18, 19 y 20 años, pero muy diferente al de
        un estudiante de 35. El binning captura esa idea.

    ¿Por qué pd.cut() y no pd.qcut()?
        - pd.cut()  → intervalos de igual ANCHO  (tú defines los límites)
        - pd.qcut() → intervalos de igual TAMAÑO (mismo nº de datos por grupo)
        Usamos pd.cut() porque los límites de edad tienen un significado
        real (18-21 = edad universitaria tradicional), no queremos
        que pandas los calcule automáticamente.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'age_at_enrollment'.
        bins (list): Lista de límites de los intervalos.
                     Ejemplo: [17, 21, 25, 30, 60]
                     (leída desde config.yaml)
        labels (list): Etiquetas para cada intervalo.
                       Ejemplo: ["18-21 (Traditional)", "22-25", ...]
                       (leída desde config.yaml)

    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'age_group' añadida.
                      La columna original 'age_at_enrollment' se mantiene
                      hasta que sea eliminada en una etapa posterior.
    """
    df_out = df.copy()

    df_out['age_group'] = pd.cut(
        df_out['age_at_enrollment'],
        bins=bins,
        labels=labels
    )

    print(f"   ✅ Variable 'age_group' creada. Distribución:")
    print(df_out['age_group'].value_counts().to_string())

    return df_out


def encode_target(df: pd.DataFrame, target_mapping: dict) -> pd.DataFrame:
    """
    Codifica la variable objetivo (target) de texto a valores numéricos.

    Técnica utilizada: Label Encoding.
    Convierte categorías de texto en números enteros.

    ¿Por qué necesitamos hacer esto?
        Los algoritmos de Machine Learning trabajan con matrices numéricas.
        No pueden procesar texto como "Graduate" o "Dropout" directamente.
        Debemos convertirlos a números antes de entrenar.

    ¿Por qué Label Encoding para el target y no One-Hot Encoding?
        One-Hot Encoding crea una columna por cada categoría y es ideal
        para variables PREDICTORAS (features), porque evita que el modelo
        asuma un orden entre categorías.
        Para la variable OBJETIVO (target) en clasificación, los algoritmos
        de scikit-learn esperan un único vector numérico (0, 1, 2...),
        no múltiples columnas.

    Args:
        df (pd.DataFrame): DataFrame con columna 'target' en formato texto.
        target_mapping (dict): Diccionario de mapeo texto → número.
                               Ejemplo: {"Graduate": 0, "Dropout": 1, "Enrolled": 2}
                               (leído desde config.yaml)

    Returns:
        pd.DataFrame: DataFrame con nueva columna 'target_encoded' (numérica).
                      La columna 'target' original se mantiene hasta la
                      etapa de eliminación de columnas.
    """
    df_out = df.copy()

    df_out['target_encoded'] = df_out['target'].map(target_mapping)

    # Verificamos que no haya valores sin mapear (NaN resultantes)
    unmapped_count = df_out['target_encoded'].isnull().sum()
    if unmapped_count > 0:
        print(f"   ⚠️  {unmapped_count} valores en 'target' no pudieron ser mapeados.")
    else:
        print(f"   ✅ Variable 'target_encoded' creada. Distribución:")
        print(df_out['target_encoded'].value_counts().to_string())

    return df_out


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a la columna 'age_group'.

    Técnica utilizada: One-Hot Encoding (también llamado Dummy Encoding).
    Convierte una columna categórica en múltiples columnas binarias (0 o 1).

    ¿Por qué One-Hot Encoding para 'age_group' y no Label Encoding?
        'age_group' es una variable categórica nominal.
        Si usáramos Label Encoding (0, 1, 2, 3), el modelo asumiría
        que "30+ Adult" (3) es "más grande" que "18-21 Traditional" (0),
        lo cual no tiene sentido matemáticamente.
        One-Hot Encoding evita ese problema al no imponer un orden.

    ¿Qué es drop_first=True?
        Si tenemos 4 grupos de edad, get_dummies crea 4 columnas.
        Pero con 3 columnas podemos representar los mismos 4 grupos
        (el grupo no representado equivale a que todas las otras sean 0).
        drop_first=True elimina una columna redundante, evitando
        el problema de "multicolinealidad perfecta" en el modelo.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'age_group' categórica.

    Returns:
        pd.DataFrame: DataFrame con 'age_group' reemplazada por columnas
                      binarias (ej: age_group_22-25, age_group_26-30, ...).
    """
    df_out = pd.get_dummies(df, columns=['age_group'], drop_first=True)

    print(f"   ✅ One-Hot Encoding aplicado a 'age_group'.")
    return df_out


def drop_redundant_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Elimina columnas que ya no son necesarias tras el feature engineering.

    ¿Qué columnas se eliminan y por qué?
        - 'target': ya tenemos 'target_encoded' (su versión numérica)
        - 'age_at_enrollment': ya tenemos 'age_group' (su versión categorizada)

        Mantener estas columnas originales causaría problemas:
        - 'target' (texto) no puede entrar al modelo
        - 'age_at_enrollment' duplicaría información con 'age_group'

    ¿Por qué errors='ignore'?
        Si alguna columna de la lista no existe en el DataFrame
        (por ejemplo, porque ya fue eliminada antes), el proceso
        no debe romperse. errors='ignore' lo maneja silenciosamente.

    Args:
        df (pd.DataFrame): DataFrame con columnas redundantes.
        columns_to_drop (list): Lista de nombres de columnas a eliminar.
                                (leída desde config.yaml)

    Returns:
        pd.DataFrame: DataFrame sin las columnas redundantes.
    """
    df_out = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"   ✅ Columnas eliminadas: {columns_to_drop}")
    return df_out


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda el DataFrame con features engineered en `data/processed/`.

    ¿Por qué `processed/` y no `interim/`?
        En nuestra arquitectura:
        - `interim/`   → datos limpios, SIN feature engineering
        - `processed/` → datos limpios, CON feature engineering ← aquí

        Este es el dataset final, listo para ser consumido por el modelo.
        Es la "verdad única" que el módulo de entrenamiento usará.

    Args:
        df (pd.DataFrame): DataFrame con features completos.
        output_path (str): Ruta donde guardar el CSV.
                           Ejemplo: "data/processed/engineered_student_data.csv"

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   💾 Datos guardados en: {output_path}")


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Función principal de feature engineering. Orquesta todas las transformaciones.

    Ejecuta en orden:
        1. Crear grupos de edad (binning)
        2. Codificar la variable objetivo (label encoding)
        3. Aplicar One-Hot Encoding a variables categóricas
        4. Eliminar columnas originales ya no necesarias
        5. Guardar el resultado en data/processed/

    Esta es la única función que el pipeline debe llamar desde fuera.

    Args:
        df (pd.DataFrame): Dataset limpio recibido desde preprocess.py.
        config (dict): Diccionario de configuración del proyecto.
                       Se esperan las claves:
                       - config["feature_engineering"]["age_bins"] (list)
                       - config["feature_engineering"]["age_labels"] (list)
                       - config["feature_engineering"]["target_mapping"] (dict)
                       - config["preprocessing"]["columns_to_drop"] (list)
                       - config["paths"]["engineered_data"] (str)

    Returns:
        pd.DataFrame: Dataset con todas las features listo para entrenamiento.
    """
    print("⚙️  Iniciando Feature Engineering...")

    age_bins        = config["feature_engineering"]["age_bins"]
    age_labels      = config["feature_engineering"]["age_labels"]
    target_mapping  = config["feature_engineering"]["target_mapping"]
    columns_to_drop = config["preprocessing"]["columns_to_drop"]
    output_path     = config["paths"]["engineered_data"]

    df = create_age_groups(df, age_bins, age_labels)
    df = encode_target(df, target_mapping)
    df = apply_one_hot_encoding(df)
    df = drop_redundant_columns(df, columns_to_drop)

    save_processed_data(df, output_path)

    print(f"✅ Feature Engineering completado. Shape final: {df.shape}")
    return df
