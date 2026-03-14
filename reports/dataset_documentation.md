# 📊 Dataset Elegido: Predict Students' Dropout and Academic Success
## Fuente: UCI Machine Learning Repository

---

## 🎯 ¿Por qué este dataset?

Este dataset de la UCI (University of California, Irvine) es el **estándar de referencia** para proyectos de clasificación de éxito/fracaso estudiantil por varias razones:

| Criterio | Por qué importa para nuestro proyecto |
|---|---|
| **Datos reales** | Provienen de una institución de educación superior real (Portugal) |
| **Problema relevante** | El abandono estudiantil es un problema social y económico real — los recruiters valoran el impacto social |
| **Complejidad adecuada** | 4.424 instancias y 36 variables: suficiente para mostrar habilidades sin ser abrumador |
| **Clase desbalanceada** | El desbalance en las clases es un reto real que demuestra conocimiento avanzado |
| **Benchmark conocido** | Los reclutadores de Data Science conocen este dataset — es un punto de conversación excelente |
| **Descarga gratuita** | Sin restricciones de uso para proyectos académicos y portfolios |

---

## 📋 Ficha Técnica del Dataset

```
Nombre:      Predict Students' Dropout and Academic Success
Fuente:      UCI Machine Learning Repository (ID: 697)
URL:         https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
Año:         2021
País:        Portugal
Tamaño:      520.8 KB
Instancias:  4.424 (cada fila = 1 estudiante)
Variables:   36 features + 1 objetivo (target)
Valores nulos: NO (ya preprocesado por los autores)
Tarea:       Clasificación Multiclase
```

---

## 🏷️ Variable Objetivo (Target)

La columna que queremos predecir se llama **`Target`** y tiene **3 clases**:

| Clase | Significado | Reto |
|---|---|---|
| `Dropout` | El estudiante abandonó la carrera | Clase mayoritaria (desbalanceo) |
| `Enrolled` | El estudiante sigue matriculado (en proceso) | Clase minoritaria |
| `Graduate` | El estudiante se graduó exitosamente | Clase mayoritaria |

> ⚠️ **El desbalanceo de clases es un reto importante** que abordaremos con técnicas como SMOTE o ajuste de pesos en el modelo. Esto es exactamente el tipo de problema que diferencia a un Data Scientist junior de uno con criterio.

---

## 📌 Las 36 Variables (Features principales)

Las variables se agrupan en 4 categorías:

### 1. Variables de Inscripción (al momento de entrar)
- Tipo de candidatura (vía de acceso a la universidad)
- Orden de preferencia de la carrera
- Turno (diurno/nocturno)
- Grado previo (bachillerato, licenciatura, etc.)
- Nacionalidad

### 2. Variables Socioeconómicas
- Ocupación profesional del padre y la madre
- Nivel educativo del padre y la madre
- Si tiene beca o no
- Si está al día con el pago de la matrícula
- Si es refugiado, desplazado o tiene necesidades educativas especiales

### 3. Variables Macroeconómicas (del país)
- Tasa de desempleo
- Tasa de inflación
- PIB (Producto Interior Bruto)

### 4. Variables de Rendimiento Académico
- Unidades curriculares aprobadas en 1er semestre
- Unidades curriculares inscritas en 1er semestre
- Nota media en 1er semestre
- Unidades curriculares aprobadas en 2º semestre
- Nota media en 2º semestre

---

## 🔍 Preguntas de Negocio que Responderemos

1. ¿Qué factor predice mejor el abandono: el rendimiento académico o la situación socioeconómica?
2. ¿Hay diferencia en la tasa de abandono entre estudiantes diurnos y nocturnos?
3. ¿Los estudiantes con beca tienen menor tasa de abandono?
4. ¿Qué combinación de variables da el mejor modelo predictivo?

---

## 📥 Cómo Descargar el Dataset

### Opción A: Descarga directa (recomendada para el proyecto)
```
URL de descarga: https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip
```
1. Descomprime el ZIP
2. El archivo se llama `dataset.csv`
3. Lo guardaremos en `data/raw/dataset.csv` (NO se sube a GitHub)

### Opción B: Descarga desde Python con la librería oficial
```python
# Instalar la librería de UCI
pip install ucimlrepo

# En un script Python o Jupyter Notebook:
from ucimlrepo import fetch_ucirepo

# Descargar el dataset directamente
students = fetch_ucirepo(id=697)

# X = variables de entrada (features)
X = students.data.features  # DataFrame con 36 columnas

# y = variable objetivo (target)
y = students.data.targets   # Serie con 'Dropout', 'Enrolled', 'Graduate'

# Ver información del dataset
print(students.metadata)
print(students.variables)
```

---

## 📄 Cita del Dataset (Para el README del proyecto)

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

---

> **💡 Nota para el recruiter:** Elegí este dataset porque representa un problema de negocio real con impacto social medible. La institución educativa necesita identificar en una etapa temprana qué estudiantes están en riesgo de abandono para poder intervenir. Esto es exactamente el tipo de análisis predictivo que un Data Scientist aplica en sectores como EdTech, Seguros o RRHH.
