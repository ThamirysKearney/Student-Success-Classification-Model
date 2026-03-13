"""
Script de Limpieza de Datos - Proyecto: Student Success Classification
Este script realiza la carga desde UCI, limpieza inicial y normalización del dataset.
Diseñado para ser escalable y documentado para portafolios de alto nivel.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os

def load_data():
    """Carga el dataset 'Predict Students Dropout and Academic Success' desde UCI Repository."""
    print("🚀 Iniciando carga de datos desde UCI Laboratory...")
    try:
        # fetch dataset 
        student_predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
        
        # data (as pandas dataframes) 
        X = student_predict_students_dropout_and_academic_success.data.features 
        y = student_predict_students_dropout_and_academic_success.data.targets 
        
        # Combinar en un solo DataFrame
        df = pd.concat([X, y], axis=1)
        print(f"✅ Datos cargados correctamente. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error al cargar desde UCI: {e}")
        # Intento de carga local si falla la API
        if os.path.exists('data/raw/data.csv'):
            return pd.read_csv('data/raw/data.csv', sep=';')
        raise

def clean_data(df):
    """Realiza la limpieza y normalización de los datos."""
    print("🧹 Iniciando proceso de limpieza...")
    
    # 1. Normalización de nombres de columnas
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # 2. Verificación de duplicados
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"   - Se eliminaron {initial_rows - len(df)} filas duplicadas.")
    
    # 3. Tratamiento de valores nulos (El dataset UCI suele venir limpio, pero validamos)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("   - Valores nulos encontrados. Aplicando imputación por mediana...")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("   - No se encontraron valores nulos.")

    # 4. Ajuste de tipos de datos (Optimización de memoria)
    # Convertimos columnas binarias o de pocos valores a tipos más ligeros
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].nunique() < 100:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    print("✅ Proceso de limpieza finalizado.")
    return df

def save_processed_data(df, path='data/processed/cleaned_student_data.csv'):
    """Guarda el dataset procesado asegurando que la carpeta existe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Dataset guardado en: {path}")

if __name__ == "__main__":
    try:
        raw_df = load_data()
        clean_df = clean_data(raw_df)
        save_processed_data(clean_df)
        print("\n🏆 ¡Proceso completado con éxito! Camino libre para el EDA.")
    except Exception as e:
        print(f"\n⚠️ Error crítico en el pipeline: {e}")
