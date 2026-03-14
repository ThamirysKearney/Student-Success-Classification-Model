"""
preprocess.py
=============
Módulo de preprocesado de datos para el proyecto Student Success Classification.

Responsabilidad única:
    Recibir el dataset crudo y devolverlo limpio y normalizado,
    listo para la fase de Feature Engineering.

¿Qué hace este módulo?
    1. Normaliza los nombres de columnas (mayúsculas → minúsculas)
    2. Elimina filas duplicadas
    3. Imputa valores nulos (si los hubiera)
    4. Optimiza los tipos de datos para reducir uso de memoria

¿Qué NO hace este módulo?
    - No crea nuevas variables (eso es Feature Engineering)
    - No entrena ningún modelo
    - No hace visualizaciones

Flujo de datos:
    data/raw/ → [load_data.py] → DataFrame crudo
              → [preprocess.py] → DataFrame limpio → data/interim/
"""

import os
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de todas las columnas del DataFrame.

    Convierte los nombres a minúsculas y reemplaza espacios y barras
    por guiones bajos. Esto garantiza que el código sea consistente
    y evita errores como df['Age at Enrollment'] vs df['age_at_enrollment'].

    ¿Por qué es importante?
        Los datasets reales suelen tener nombres de columnas
        inconsistentes: algunas en mayúsculas, otras con espacios,
        otras con caracteres especiales. Normalizarlos al principio
        evita errores a lo largo de todo el pipeline.

    Ejemplo de transformación:
        'Age at Enrollment' → 'age_at_enrollment'
        'Curricular units/2nd sem grade' → 'curricular_units_2nd_sem_grade'

    Args:
        df (pd.DataFrame): DataFrame con nombres de columnas originales.

    Returns:
        pd.DataFrame: DataFrame con nombres de columnas normalizados.
                      El DataFrame original no se modifica (enfoque inmutable).
    """
    # Creamos una copia para no modificar el DataFrame original.
    # Este patrón se llama "enfoque inmutable" y es más seguro
    # que modificar los datos en el sitio (inplace).
    df_clean = df.copy()

    df_clean.columns = (
        df_clean.columns
        .str.lower()               # 'Age' → 'age'
        .str.replace(' ', '_')     # 'age at enrollment' → 'age_at_enrollment'
        .str.replace('/', '_')     # 'units/sem' → 'units_sem'
        .str.strip()               # elimina espacios al inicio y al final
    )

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas duplicadas del DataFrame e informa cuántas se eliminaron.

    Una fila duplicada es aquella que tiene exactamente los mismos valores
    en todas las columnas que otra fila. En datasets reales pueden aparecer
    por errores en la ingesta de datos o por registros duplicados en la fuente.

    Args:
        df (pd.DataFrame): DataFrame que puede contener duplicados.

    Returns:
        pd.DataFrame: DataFrame sin filas duplicadas.
    """
    rows_before = len(df)
    df_clean = df.drop_duplicates()
    rows_after = len(df_clean)

    removed = rows_before - rows_after
    if removed > 0:
        print(f"   ⚠️  Se eliminaron {removed} filas duplicadas.")
    else:
        print("   ✅ No se encontraron filas duplicadas.")

    return df_clean


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa (rellena) los valores nulos con la mediana de cada columna.

    ¿Por qué la mediana y no la media?
        La mediana es más robusta ante outliers (valores extremos).
        Por ejemplo, si la mayoría de los estudiantes tienen 20 años
        pero hay uno con 80, la media se desplaza hacia arriba.
        La mediana no se ve afectada por esos casos extremos.

    ¿Por qué solo columnas numéricas?
        La mediana solo tiene sentido para valores numéricos.
        Las columnas de tipo texto (object) se manejarían de forma
        diferente (por ejemplo, con la moda), pero en este dataset
        todas las columnas son numéricas o ya están codificadas.

    Args:
        df (pd.DataFrame): DataFrame con posibles valores nulos.

    Returns:
        pd.DataFrame: DataFrame con valores nulos imputados.
    """
    total_nulls = df.isnull().sum().sum()

    if total_nulls == 0:
        print("   ✅ No se encontraron valores nulos.")
        return df

    print(f"   ⚠️  Se encontraron {total_nulls} valores nulos. Imputando con mediana...")

    # fillna con la mediana de cada columna numérica.
    # Usamos .copy() para no modificar el DataFrame original.
    median_values = df.median(numeric_only=True)
    df_clean = df.fillna(median_values)

    return df_clean


def optimize_data_types(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Optimiza los tipos de datos numéricos para reducir el uso de memoria.

    Por defecto, pandas carga todos los enteros como int64 (8 bytes por valor).
    Si una columna tiene pocos valores únicos (como 0 y 1), podemos
    guardarla como int8 (1 byte), ahorrando hasta 8 veces memoria.

    ¿Cuándo importa esto?
        En este dataset pequeño el impacto es mínimo. Pero si el proyecto
        escala a millones de estudiantes, puede marcar la diferencia entre
        que el proceso quepa o no en la memoria del servidor.

    Args:
        df (pd.DataFrame): DataFrame con columnas enteras posiblemente ineficientes.
        threshold (int): Si una columna tiene menos valores únicos que este
                         umbral, se convierte a un tipo más ligero.
                         Valor recomendado: 100 (configurado en config.yaml).

    Returns:
        pd.DataFrame: DataFrame con tipos de datos optimizados.
    """
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=['int64']).columns:
        if df_clean[col].nunique() < threshold:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    return df_clean


def save_interim_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda el DataFrame limpio en la carpeta `data/interim/`.

    ¿Por qué `interim/` y no `processed/`?
        En nuestra arquitectura de datos:
        - `interim/` → datos limpios (sin duplicados, nulos resueltos)
        - `processed/` → datos con feature engineering completo

        El archivo que genera este módulo está limpio pero aún no
        tiene las variables creadas por Feature Engineering.
        Por eso va a `interim/`, no a `processed/`.

    Args:
        df (pd.DataFrame): DataFrame limpio listo para guardar.
        output_path (str): Ruta completa donde guardar el archivo CSV.
                           Ejemplo: "data/interim/cleaned_student_data.csv"

    Returns:
        None
    """
    # Creamos la carpeta si no existe (os.makedirs es seguro con exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"   💾 Datos guardados en: {output_path}")


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Función principal de preprocesado. Orquesta todos los pasos de limpieza.

    Ejecuta en orden:
        1. Normalización de nombres de columnas
        2. Eliminación de duplicados
        3. Imputación de valores nulos
        4. Optimización de tipos de datos
        5. Guardado del resultado en data/interim/

    Esta es la única función que el pipeline debe llamar desde fuera.
    Las funciones auxiliares (normalize, remove_duplicates, etc.) están
    diseñadas para ser pequeñas, testeables y reutilizables de forma
    independiente si fuera necesario.

    Args:
        df (pd.DataFrame): Dataset crudo recibido desde load_data.py.
        config (dict): Diccionario de configuración del proyecto.
                       Se esperan las claves:
                       - config["preprocessing"]["dtype_optimization_threshold"] (int)
                       - config["paths"]["cleaned_data"] (str)

    Returns:
        pd.DataFrame: Dataset limpio, listo para Feature Engineering.
    """
    print("🧹 Iniciando preprocesado de datos...")

    threshold   = config["preprocessing"]["dtype_optimization_threshold"]
    output_path = config["paths"]["cleaned_data"]

    df = normalize_column_names(df)
    print("   ✅ Nombres de columnas normalizados.")

    df = remove_duplicates(df)

    df = impute_missing_values(df)

    df = optimize_data_types(df, threshold)
    print("   ✅ Tipos de datos optimizados.")

    save_interim_data(df, output_path)

    print(f"✅ Preprocesado completado. Shape final: {df.shape}")
    return df
