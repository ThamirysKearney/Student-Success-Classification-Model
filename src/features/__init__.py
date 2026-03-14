# Este archivo convierte la carpeta `src/features/` en un módulo de Python.
# Gracias a él, podemos importar funciones así:
#
#   from src.features.feature_engineering import engineer_features
#
# Sin este archivo, Python no reconocería `src/features/` como un paquete
# y los imports anteriores fallarían con un ModuleNotFoundError.
