# Este archivo convierte la carpeta `src/data/` en un módulo de Python.
# Gracias a él, podemos importar funciones así:
#
#   from src.data.load_data import load_dataset
#   from src.data.preprocess import clean_data
#
# Sin este archivo, Python no reconocería `src/data/` como un paquete
# y los imports anteriores fallarían con un ModuleNotFoundError.
