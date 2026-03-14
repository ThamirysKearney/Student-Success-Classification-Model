"""
load_data.py
============
Módulo de carga de datos para el proyecto Student Success Classification.

Responsabilidad única:
    Obtener el dataset crudo desde la fuente oficial (UCI) o desde
    un archivo local de respaldo, y devolverlo como un DataFrame.

Este módulo NO limpia, NO transforma y NO analiza los datos.
Su único trabajo es traer los datos crudos al pipeline.

¿Por qué separar la carga de la limpieza?
    Si mañana cambia la fuente de datos (de UCI a una base de datos SQL,
    o a un archivo en la nube), solo necesitas modificar este archivo.
    El resto del pipeline permanece intacto.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_from_uci(dataset_id: int) -> pd.DataFrame:
    """
    Descarga el dataset desde el repositorio UCI Machine Learning.

    UCI (University of California, Irvine) mantiene un repositorio
    público de datasets para investigación en Machine Learning.
    Usamos la librería oficial `ucimlrepo` para acceder al dataset
    mediante su ID numérico.

    ¿Por qué separar features (X) y target (y)?
        La API de UCI devuelve las variables predictoras (X) y la
        variable objetivo (y) por separado. Las combinamos en un solo
        DataFrame para que el pipeline trabaje con una única tabla.

    Args:
        dataset_id (int): Identificador numérico del dataset en UCI.
                          Para este proyecto: 697.
                          Ver: https://archive.ics.uci.edu/dataset/697

    Returns:
        pd.DataFrame: Dataset completo con features y target
                      combinados en un único DataFrame.

    Raises:
        Exception: Si la API de UCI no está disponible.
    """
    print(f"🌐 Conectando con UCI Machine Learning Repository (ID: {dataset_id})...")

    # Descargamos el dataset usando la API oficial de UCI.
    # La variable se llama 'dataset' — nombre corto y descriptivo.
    dataset = fetch_ucirepo(id=dataset_id)

    # La API devuelve features (X) y target (y) por separado.
    # Los combinamos en un único DataFrame con pd.concat().
    # axis=1 significa "unir columna a columna" (no fila a fila).
    features = dataset.data.features
    target   = dataset.data.targets
    df = pd.concat([features, target], axis=1)

    print(f"✅ Dataset cargado desde UCI. Dimensiones: {df.shape}")
    return df


def load_from_local(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV local.

    Esta función se usa como respaldo cuando no hay conexión a internet
    o cuando queremos trabajar con una versión fija del dataset sin
    depender de la disponibilidad de la API de UCI.

    ¿Por qué el separador es ';' y no ','?
        El dataset original de UCI usa punto y coma (;) como separador
        en el CSV. Es importante especificarlo, de lo contrario pandas
        leería todo el archivo como una sola columna gigante.

    Args:
        file_path (str): Ruta al archivo CSV local.
                         Ejemplo: "data/raw/data.csv"

    Returns:
        pd.DataFrame: Dataset cargado desde el archivo local.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ No se encontró el archivo local en: '{file_path}'\n"
            f"   Asegúrate de que el archivo existe o que hay conexión a internet."
        )

    print(f"📂 Cargando datos desde archivo local: {file_path}")
    df = pd.read_csv(file_path, sep=";")
    print(f"✅ Dataset cargado desde local. Dimensiones: {df.shape}")
    return df


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Función principal de carga. Orquesta el proceso completo.

    Intenta primero descargar desde UCI. Si falla (sin conexión,
    API caída, etc.), usa el archivo local como respaldo.

    Esta es la única función que el pipeline debe llamar desde fuera.
    Las funciones `load_from_uci` y `load_from_local` son auxiliares
    internas que no deberían usarse directamente desde otros módulos.

    ¿Por qué recibir `config` como parámetro y no leerlo aquí?
        Porque si este módulo leyera el config.yaml directamente,
        sería más difícil de testear — tendríamos que crear un archivo
        YAML real para poder ejecutar los tests.
        Al recibir el config como diccionario, podemos pasar
        cualquier diccionario en los tests, sin archivos externos.
        Este patrón se llama "Inyección de Dependencias".

    Args:
        config (dict): Diccionario con la configuración del proyecto.
                       Se espera que contenga las claves:
                       - config["data_source"]["uci_dataset_id"] (int)
                       - config["data_source"]["local_fallback_path"] (str)

    Returns:
        pd.DataFrame: Dataset crudo listo para pasar a la fase
                      de preprocesado (preprocess.py).
    """
    dataset_id    = config["data_source"]["uci_dataset_id"]
    fallback_path = config["data_source"]["local_fallback_path"]

    try:
        return load_from_uci(dataset_id)

    except Exception as uci_error:
        print(f"⚠️  No se pudo conectar con UCI: {uci_error}")
        print(f"🔄 Intentando carga desde archivo local...")
        return load_from_local(fallback_path)
