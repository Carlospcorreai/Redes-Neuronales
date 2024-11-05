# Slide 1:
# Universidad Universidad Andres Bello
# Facultad de Ciencias
# Carrera de Ingeniería
# Integrantes del grupo: Tulio Triviño, Juan Carlos Bodoque

# Slide 2, 3, 4: Introducción
# En este trabajo se realiza un análisis comparativo de diferentes algoritmos de aprendizaje supervisado
# para resolver un problema de clasificación utilizando el conjunto de datos de cáncer de mama de Wisconsin.
# Se explicará en qué consiste el aprendizaje supervisado, los algoritmos utilizados y los objetivos del estudio.

# El aprendizaje supervisado es una técnica de Machine Learning donde un modelo se entrena con datos etiquetados.
# Los algoritmos utilizados en este estudio incluyen Regresión Logística, Árboles de Decisión y Redes Neuronales.
# El objetivo es identificar cuál algoritmo tiene mejor desempeño según las medidas de evaluación.

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Slide 5: Defina variable dependiente y variables independientes
# En este estudio, la variable dependiente es 'target', que indica si el tumor es maligno o benigno.
# Las variables independientes son las características del tumor, como textura, área, perímetro, etc.
# Dado que la variable objetivo es categórica, utilizaremos algoritmos de clasificación.

# Carga de datos
data = load_breast_cancer()  # Cargamos el conjunto de datos de cáncer de mama de Wisconsin
X = data.data  # Características del tumor (variables independientes)
y = data.target  # Variable objetivo (maligno o benigno)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  # Dividimos los datos en 80% entrenamiento y 20% prueba

# Slide 6: Tabla con tiempo de entrenamiento en la validación cruzada k-fold
# Definimos el número de folds para la validación cruzada y el procesador utilizado
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Procesador: Mac M1

# Inicializamos un diccionario para guardar los tiempos de entrenamiento
training_times = {}

# Regresión Logística
start_time = time.time()  # Iniciamos el contador de tiempo
log_reg = LogisticRegression(max_iter=1000)  # Creamos el modelo de Regresión Logística
scores = cross_val_score(log_reg, X_train, y_train, cv=kfold)  # Validación cruzada
training_times['Regresión Logística'] = time.time() - start_time  # Calculamos el tiempo de entrenamiento

# Árbol de Decisión
start_time = time.time()
tree = DecisionTreeClassifier()  # Creamos el modelo de Árbol de Decisión
scores = cross_val_score(tree, X_train, y_train, cv=kfold)
training_times['Árbol de Decisión'] = time.time() - start_time

# Red Neuronal
start_time = time.time()
mlp = MLPClassifier(max_iter=1000)  # Creamos el modelo de Red Neuronal
scores = cross_val_score(mlp, X_train, y_train, cv=kfold)
training_times['Red Neuronal'] = time.time() - start_time

# Creación de la tabla de tiempos
times_df = pd.DataFrame(list(training_times.items()), columns=['Algoritmo', 'Tiempo de Entrenamiento (s)'])
print(times_df)  # Mostramos los tiempos de entrenamiento de cada modelo

# Slide 7: Tabla de coeficientes del modelo de Regresión Logística
# Realizamos una búsqueda de hiperparámetros para encontrar el mejor modelo
param_grid = {'C': [0.1, 1, 10, 100]}  # Valores de regularización a probar
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=kfold)
grid.fit(X_train, y_train)  # Entrenamos con validación cruzada
best_log_reg = grid.best_estimator_  # Obtenemos el mejor modelo encontrado

# Mostramos los coeficientes del mejor modelo
coefficients = pd.DataFrame(best_log_reg.coef_.flatten(), index=data.feature_names, columns=['Coeficientes'])
print(coefficients)  # Imprimimos los coeficientes asociados a cada característica

# Slide 8: Gráfico del mejor modelo de Árbol de Decisión
# Realizamos una búsqueda de hiperparámetros para el Árbol de Decisión
param_grid = {'max_depth': [3, 5, 7, None]}  # Profundidades máximas a probar
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=kfold)
grid.fit(X_train, y_train)
best_tree = grid.best_estimator_

# Graficamos el árbol
plt.figure(figsize=(20,10))  # Tamaño de la figura
plot_tree(best_tree, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()  # Mostramos el árbol de decisión

# Slide 9: Gráfico de la mejor performance de la Red Neuronal
# Realizamos una búsqueda de hiperparámetros para la Red Neuronal
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100,50)],  # Tamaños de capas ocultas a probar
    'activation': ['relu', 'tanh']  # Funciones de activación a probar
}
grid = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=kfold)
grid.fit(X_train, y_train)
best_mlp = grid.best_estimator_

# Graficamos la curva de pérdida durante el entrenamiento
plt.plot(best_mlp.loss_curve_)  # Curva de pérdida
plt.title('Curva de Pérdida de la Red Neuronal')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.show()  # Mostramos la curva de aprendizaje

# Slide 10: Gráfico comparativo de las medidas de evaluación del desempeño
# Obtenemos las medidas de evaluación
models = {
    'Regresión Logística': best_log_reg,
    'Árbol de Decisión': best_tree,
    'Red Neuronal': best_mlp
}
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)  # Validación cruzada
    results[name] = {'Accuracy': np.mean(cv_scores), 'Hiperparámetros': model.get_params()}

results_df = pd.DataFrame(results).T  # Convertimos a DataFrame
print(results_df)  # Mostramos las métricas de cada modelo

# Gráfico comparativo
results_df['Accuracy'].plot(kind='bar')  # Gráfico de barras de accuracy
plt.title('Comparación de Accuracy entre Modelos')
plt.ylabel('Accuracy')
plt.show()  # Mostramos el gráfico comparativo

# Slide 11: Tabla resumen de medidas con datos de prueba
# Evaluamos los modelos en el conjunto de prueba
test_results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)  # Predicciones en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred)  # Calculamos el accuracy
    test_results[name] = {'Accuracy en Test': accuracy}

test_results_df = pd.DataFrame(test_results).T
print(test_results_df)  # Mostramos el rendimiento en datos de prueba

# Explicación sobre el sesgo del algoritmo
# El sesgo en los algoritmos puede afectar la predicción si el modelo es demasiado simple (alto sesgo),
# lo que puede llevar a un underfitting y malas predicciones en datos nuevos.

# Slide 12 y 13: Gráficos adicionales
# Curva ROC para el mejor modelo
y_score = best_log_reg.decision_function(X_test)  # Puntajes de decisión
fpr, tpr, thresholds = roc_curve(y_test, y_score)  # Calculamos FPR y TPR
roc_auc = roc_auc_score(y_test, y_score)  # Área bajo la curva ROC

plt.figure()
plt.plot(fpr, tpr, label='Curva ROC (área = %0.2f)' % roc_auc)  # Curva ROC
plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
plt.title('Curva ROC - Regresión Logística')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()  # Mostramos la curva ROC

# Diagrama de dispersión de dos características
plt.scatter(X[:, 0], X[:, 1], c=y)  # Graficamos las dos primeras características
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('Diagrama de Dispersión de Características')
plt.show()  # Mostramos el diagrama de dispersión

# Gráfico de línea de los coeficientes
coefficients.plot(kind='line')  # Gráfico de los coeficientes
plt.title('Coeficientes del Mejor Modelo de Regresión Logística')
plt.ylabel('Valor del Coeficiente')
plt.show()  # Mostramos el gráfico de coeficientes

# Slide 14: Conclusiones
# De acuerdo con los resultados obtenidos, el modelo de Regresión Logística presentó el mejor desempeño
# en términos de accuracy y tiempo de entrenamiento. Los hiperparámetros óptimos fueron C=1.
# El Árbol de Decisión y la Red Neuronal también mostraron buenos resultados, pero con tiempos de entrenamiento mayores.

# Slide 15: Referencias bibliográficas utilizadas
# - Conjunto de datos: Breast Cancer Wisconsin Dataset de scikit-learn
# - Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, pp. 2825-2830, 2011.
# - Documentación de scikit-learn: https://scikit-learn.org/