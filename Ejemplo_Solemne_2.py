

# Importar las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings


def main():
    # Ignorar las advertencias de convergencia para mantener la salida limpia
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Cargar el conjunto de datos Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dividir los datos en conjunto de entrenamiento y prueba con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Crear un pipeline que estandariza los datos y entrena la red neuronal
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=42, early_stopping=True))
    ])

    # Definir la cuadrícula de hiperparámetros para buscar
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__solver': ['adam'],  # 'adam' tiende a converger mejor que 'sgd' en muchos casos
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate': ['adaptive'],
        'mlp__learning_rate_init': [0.001, 0.01],  # Tasa de aprendizaje inicial
    }

    # Configurar la búsqueda en cuadrícula con validación cruzada de 5 pliegues
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # Número de pliegues
        n_jobs=-1,
        scoring='accuracy',
        verbose=2,
        return_train_score=True
    )

    # Entrenar el modelo usando GridSearchCV
    grid_search.fit(X_train, y_train)

    # Mostrar los mejores parámetros encontrados
    print("\nMejores parámetros encontrados:")
    print(grid_search.best_params_)

    # Mostrar la mejor puntuación de validación cruzada
    print(f"\nMejor precisión en validación cruzada: {grid_search.best_score_:.4f}")

    # Obtener todos los resultados de la búsqueda y mostrarlos como DataFrame ordenado
    results = pd.DataFrame(grid_search.cv_results_)
    results_sorted = results.sort_values(by='mean_test_score', ascending=False)
    print("\nResultados de la validación cruzada para cada combinación de hiperparámetros:")
    print(results_sorted[['mean_test_score', 'std_test_score', 'params']].to_string(index=False))

    # Evaluar el mejor modelo en el conjunto de prueba
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nReporte de clasificación en el conjunto de prueba:")
    print(classification_report(y_test, y_pred))

    print("Matriz de confusión en el conjunto de prueba:")
    print(confusion_matrix(y_test, y_pred))

    # Evaluar el modelo con validación cruzada en el conjunto completo
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print("\nValidación cruzada adicional con 5 pliegues en todo el conjunto de datos:")
    print(f"Precisión por pliegue: {cv_scores}")
    print(f"Precisión Media: {cv_scores.mean():.4f} (Desviación Estándar: {cv_scores.std():.4f})")

if __name__ == "__main__":
    main()

"""
---
Rúbrica de Evaluación: Red Neuronal con Validación Cruzada y GridSearch en scikit-learn

## **Rúbrica de Evaluación: Red Neuronal con Validación Cruzada y GridSearch en scikit-learn**

| **Criterio**                                      | **Descripción**                                                                                                                                                                              | **Puntuación Máxima** |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| **1. Carga y Exploración de Datos**               | - Importación correcta del conjunto de datos (e.g., Iris).<br>- División adecuada en conjuntos de entrenamiento y prueba.<br>- Manejo de variables independientes y dependientes.            | 14 puntos             |
| **2. Preprocesamiento de Datos**                  | - Uso correcto de `StandardScaler` para estandarizar los datos.<br>- Implementación adecuada dentro de un `Pipeline` para asegurar el preprocesamiento durante la validación cruzada.          | 14 puntos             |
| **3. Construcción del Pipeline**                  | - Creación efectiva de un `Pipeline` que incluye escalado y modelo de red neuronal.<br>- Uso apropiado de `MLPClassifier` con parámetros iniciales definidos.                                 | 14 puntos             |
| **4. Definición de la Cuadrícula de Hiperparámetros** | - Selección relevante de hiperparámetros para la red neuronal (e.g., `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`).<br>- Definición clara y completa de las combinaciones posibles. | 14 puntos             |
| **5. Configuración de GridSearchCV**              | - Implementación correcta de `GridSearchCV` con el `Pipeline`.<br>- Parámetros adecuados para la validación cruzada (e.g., `cv=5`, `n_jobs=-1`).<br>- Uso correcto de `scoring` y `verbose`.     | 14 puntos             |
| **6. Entrenamiento y Optimización del Modelo**    | - Ejecución correcta de `grid_search.fit()` en el conjunto de entrenamiento.<br>- Manejo adecuado de posibles advertencias o errores durante el entrenamiento.<br>- Tiempo de entrenamiento razonable. | 10 puntos             |
| **7. Evaluación de Resultados**                   | - Presentación clara de los mejores parámetros encontrados.<br>- Reporte preciso de la mejor puntuación de validación cruzada.<br>- Evaluación detallada del modelo en el conjunto de prueba (e.g., `classification_report`, `confusion_matrix`). | 10 puntos             |
| **8. Documentación y Claridad del Código**        | - Código bien estructurado y comentado.<br>- Uso adecuado de nombres de variables y funciones.<br>- Claridad en la presentación de resultados y explicaciones.                              | 5 puntos              |
| **9. Consideraciones Adicionales**                | - Inclusión de consideraciones sobre preprocesamiento, selección de hiperparámetros, y evaluación.<br>- Adaptabilidad del código a otros conjuntos de datos.                                | 5 puntos              |
| **Total**                                          |                                                                                                                                                                                              | **100 puntos**        |

---
### **Descripción de los Criterios**

1. **Carga y Exploración de Datos (14 puntos)**
   - **Importación de Datos**: Verificar que el conjunto de datos se carga correctamente utilizando `load_iris()` u otro método adecuado.
   - **División de Datos**: Asegurarse de que los datos se dividen correctamente en conjuntos de entrenamiento y prueba utilizando `train_test_split`, con una proporción adecuada (e.g., 80-20) y estratificación para mantener la distribución de clases.
   - **Asignación de Variables**: Confirmar que las variables independientes (`X`) y la variable dependiente (`y`) están correctamente asignadas.

2. **Preprocesamiento de Datos (14 puntos)**
   - **Estandarización**: Uso correcto de `StandardScaler` para estandarizar las características.
   - **Integración en Pipeline**: Verificar que el escalado se integra adecuadamente en un `Pipeline` para garantizar que se aplique correctamente durante la validación cruzada.

3. **Construcción del Pipeline (14 puntos)**
   - **Componentes del Pipeline**: Asegurarse de que el `Pipeline` incluye tanto el escalador como el clasificador de red neuronal (`MLPClassifier`).
   - **Configuración Inicial**: Verificar que `MLPClassifier` está configurado con parámetros iniciales razonables, como `max_iter`, `random_state`, y `early_stopping`.

4. **Definición de la Cuadrícula de Hiperparámetros (14 puntos)**
   - **Selección de Hiperparámetros**: Evaluar la relevancia de los hiperparámetros seleccionados para la optimización.
   - **Combinaciones de Parámetros**: Comprobar que las combinaciones de hiperparámetros son exhaustivas y pertinentes para mejorar el rendimiento del modelo, evitando configuraciones que dificulten la convergencia.

5. **Configuración de GridSearchCV (14 puntos)**
   - **Implementación Correcta**: Confirmar que `GridSearchCV` está configurado correctamente con el `Pipeline`, `param_grid`, y otros parámetros como `cv=5`, `n_jobs=-1`, `scoring='accuracy'`, y `verbose=2`.
   - **Optimización Eficiente**: Asegurar que se utilizan recursos de manera eficiente (e.g., `n_jobs=-1` para utilizar todos los núcleos disponibles).

6. **Entrenamiento y Optimización del Modelo (10 puntos)**
   - **Ejecución del Entrenamiento**: Verificar que el entrenamiento del modelo se realiza sin errores y que el proceso es eficiente.
   - **Manejo de Errores**: Comprobar que el código maneja adecuadamente posibles advertencias o errores durante el entrenamiento, como la no convergencia.

7. **Evaluación de Resultados (10 puntos)**
   - **Mejores Parámetros**: Presentar claramente los mejores parámetros encontrados por `GridSearchCV`.
   - **Puntuación de Validación Cruzada**: Reportar la mejor puntuación obtenida durante la validación cruzada.
   - **Evaluación en Conjunto de Prueba**: Incluir un reporte de clasificación y una matriz de confusión para evaluar el rendimiento final del modelo.

8. **Documentación y Claridad del Código (5 puntos)**
   - **Comentarios y Explicaciones**: El código debe estar bien comentado, explicando cada sección y los pasos realizados.
   - **Legibilidad**: Uso adecuado de nombres de variables y estructura del código para facilitar su comprensión.

9. **Consideraciones Adicionales (5 puntos)**
   - **Reflexión sobre el Proceso**: Incluir consideraciones sobre por qué se eligieron ciertos hiperparámetros o métodos.
   - **Adaptabilidad**: Demostrar que el código puede adaptarse fácilmente a otros conjuntos de datos o tareas similares.

---
### **Guía para la Evaluación**

- **Excepcional (90-100 puntos):** El código cumple con todos los criterios de manera excelente, mostrando una comprensión profunda de cada componente y una implementación sin errores. La documentación es clara y detallada.
  
- **Bueno (75-89 puntos):** El código cumple con la mayoría de los criterios, con pequeñas omisiones o errores menores. La documentación es adecuada pero podría ser más detallada.
  
- **Satisfactorio (60-74 puntos):** El código cumple con algunos criterios básicos, pero le faltan elementos importantes o presenta errores que afectan el rendimiento del modelo. La documentación es limitada.
  
- **Insuficiente (<60 puntos):** El código no cumple con los criterios principales, presenta errores significativos o falta de implementación de componentes esenciales. La documentación es deficiente o inexistente.

---
### **Comentarios Adicionales**

- **Originalidad y Creatividad:** Se valorará positivamente cualquier esfuerzo por mejorar el ejemplo proporcionado, como la inclusión de visualizaciones adicionales, el uso de técnicas avanzadas de preprocesamiento, o la experimentación con diferentes modelos.
  
- **Optimización y Eficiencia:** Se considerará la eficiencia del código, especialmente en términos de tiempo de ejecución y uso de recursos. El uso adecuado de `n_jobs=-1` para paralelizar la búsqueda de hiperparámetros es un aspecto positivo.

- **Resultados y Análisis:** Además de presentar los resultados, se valorará la capacidad de interpretar y analizar los resultados obtenidos, destacando insights relevantes sobre el rendimiento del modelo y posibles mejoras.

---
Fin del archivo `neural_network_with_rubric.py`
"""
