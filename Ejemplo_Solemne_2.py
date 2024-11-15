"""
Archivo: neural_network_with_rubric.py
Descripción: Implementación de una red neuronal con validación cruzada y búsqueda de hiperparámetros utilizando GridSearchCV en scikit-learn.
Incluye una rúbrica de evaluación como comentarios al final del archivo.
"""

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
| **7. Evaluación de Resultados**                   | - Presentación clara de los mejores parámetros encontrados.<br>- Reporte preciso de la mejor puntuación de