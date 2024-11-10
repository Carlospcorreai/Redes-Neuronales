from sklearn.datasets import load_breast_cancer
import pandas as pd

# Cargamos el conjunto de datos de cáncer de mama
data = load_breast_cancer()

# Convertimos los datos en un DataFrame de pandas y asignamos nombres a las columnas
df = pd.DataFrame(data.data, columns=data.feature_names)

# Agregamos una nueva columna 'target' al DataFrame con las etiquetas de clasificación
df['target'] = data.target

# Mostramos las primeras 5 filas del DataFrame para verificar que los datos se han cargado correctamente
print(df.head(1000))

