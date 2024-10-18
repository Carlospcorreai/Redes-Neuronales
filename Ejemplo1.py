# Importar las librerías necesarias
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np  # Para manipulación de datos numéricos
from tensorflow.keras.models import Sequential  # Para crear un modelo secuencial (capa por capa)
from tensorflow.keras.layers import Dense  # Para agregar capas densamente conectadas (fully connected)
from sklearn.model_selection import train_test_split  # Para dividir el conjunto de datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para normalizar los datos

# Paso 1: Generar datos simulados
np.random.seed(0)  # Para reproducibilidad
X = np.random.rand(1000, 10)  # 1000 ejemplos con 10 características cada uno
y = (X.sum(axis=1) > 5).astype(int)  # Etiqueta binaria basada en la suma de las características

# Paso 2: Normalización de los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Paso 3: Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para crear el modelo con parámetros variables
def create_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(16, input_dim=10, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 4: Definir rejilla de hiperparámetros
batch_sizes = [10, 20, 40]
epochs_list = [50, 100]
optimizers = ['adam', 'rmsprop']
initializers = ['uniform', 'normal']

# Paso 5: Implementar búsqueda de hiperparámetros manualmente
best_accuracy = 0
best_params = {}

for batch_size in batch_sizes:
    for epochs in epochs_list:
        for optimizer in optimizers:
            for init in initializers:
                print(f"Probando: batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}, init={init}")
                model = create_model(optimizer=optimizer, init=init)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Evaluar el modelo en los datos de prueba
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                # Guardar los mejores hiperparámetros
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'optimizer': optimizer,
                        'init': init
                    }

# Imprimir los mejores parámetros y precisión
print(f"Mejor precisión: {best_accuracy} con los parámetros: {best_params}")

# Entrenar el modelo final con los mejores parámetros
best_model = create_model(optimizer=best_params['optimizer'], init=best_params['init'])
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# Paso 6: Evaluar el modelo final
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Paso 7: Hacer predicciones
predictions = best_model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convertir probabilidades en 0 o 1

# Imprimir algunas predicciones junto con las etiquetas reales
for i in range(10):
    print(f"Predicción: {predictions[i][0]}, Real: {y_test[i]}")

# Paso 8: Medidas de desempeño adicionales
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))