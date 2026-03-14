# Este archivo convierte la carpeta `src/models/` en un módulo de Python.
# Gracias a él, podemos importar funciones así:
#
#   from src.models.train_model import train_model
#   from src.models.predict import predict_batch
#
# Sin este archivo, Python no reconocería `src/models/` como un paquete
# y los imports anteriores fallarían con un ModuleNotFoundError.
