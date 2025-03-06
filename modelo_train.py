import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import utils as utils

## PUNTO 2
df = utils.leer_csv('loan_modificado.csv')
df = df.drop('LoanID', axis=1)
df = utils.eliminar_nulos(df)
df = utils.one_hot_encoding(df, 'LoanPurpose')
for col in ['Education', 'EmploymentType', 'MaritalStatus']:
    df = utils.label_encoding(df, col)

binary_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_columns:
    df = utils.binary_encoding(df, col)

df = utils.colocar_ultima_columna(df, 'Default')
df = utils.convertir_a_numerico(df)


## PUNTO 3
X = df.drop(columns=['Default']).values 
y = df['Default'].values  

np.random.seed(42)
tf.random.set_seed(42)
batch_size = 32

modelo = Sequential([
    Dense(X.shape[1], activation='relu', input_shape=(X.shape[1],)),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
datosEntrenamiento = modelo.fit(X_train, y_train, epochs=10, batch_size=batch_size)

train_loss, train_acc = modelo.evaluate(X_train, y_train)
test_loss, test_acc = modelo.evaluate(X_test, y_test)

print('Precisión en entrenamiento:', train_acc)
print('Precisión en test:', test_acc)


# PUNTO 4
combinaciones = utils.combinaciones_modelos()
utils.entrenar_modelo_automatico(combinaciones, X, X_train, y_train, X_test, y_test)

