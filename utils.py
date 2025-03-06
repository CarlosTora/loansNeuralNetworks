import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
import os


def leer_csv(ruta):
    return pd.read_csv(ruta, sep=',')

def eliminar_nulos(df):
    return df.dropna()

def one_hot_encoding(df, columna):
    return pd.get_dummies(df, columns=[columna])

def label_encoding(df, columna):
    le = LabelEncoder()
    df[columna] = le.fit_transform(df[columna])
    return df

def binary_encoding(df, columna):
    df[columna] = df[columna].map({'Yes': 1, 'No': 0})
    return df

def colocar_ultima_columna(df, columna):
    return df[[col for col in df.columns if col != columna] + [columna]]

def convertir_a_numerico(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')  
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    return df


def combinaciones_modelos():
    combinaciones = [
    (10, 2, 4, 'SGD', 48, False, 0),
    (10, 2, 4, 'SGD', 48, True, 0.2),
    (10, 2, 4, 'Adam', 48, False, 0),
    (10, 2, 4, 'Adam', 48, True, 0.2),
    (10, 2, 8, 'SGD', 48, False, 0),
    (10, 2, 8, 'SGD', 48, True, 0.2),
    (10, 2, 8, 'Adam', 48, False, 0),
    (10, 2, 8, 'Adam', 48, True, 0.2),
    (15, 3, 12, 'SGD', 48, False, 0),
    (15, 3, 12, 'SGD', 48, True, 0.2),
    (15, 3, 12, 'Adam', 48, False, 0),
    (15, 3, 12, 'Adam', 48, True, 0.2),
    (20, 4, 4, 'SGD', 48, False, 0),
    (20, 4, 4, 'SGD', 48, True, 0.2)]

    return combinaciones

def entrenar_modelo_automatico (combinaciones, X, X_train, y_train, X_test, y_test) :
    
    np.random.seed(42)
    tf.random.set_seed(42)
    mejor_accuracy = 0
    mejor_modelo = None

    results_file = "resultadoModelos.txt"
    with open(results_file, "w") as f:
        f.write("Configuraciones probadas:\n\n")


    for i, config in enumerate(combinaciones, start=1): 
        print(f"Entrenando el modelo {i} de {len(combinaciones)}")
        epochs, cantidad_capas, neuronas, optimizer, batch_size, dropout, dropout_ratio = config

        modelo = Sequential()
        modelo.add(Dense(X.shape[1], activation='relu', input_shape=(X.shape[1],)))

        for _ in range(cantidad_capas):
            modelo.add(Dense(neuronas, activation='relu'))
            if dropout:
                modelo.add(Dropout(dropout_ratio))

        modelo.add(Dense(1, activation='sigmoid'))

        modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        coste_train, accuracy_train = modelo.evaluate(X_train, y_train)
        coste_test, accuracy_test = modelo.evaluate(X_test, y_test)

        if accuracy_test > mejor_accuracy:
            index = i
            mejor_accuracy = accuracy_test
            mejor_modelo = modelo

        dropout_text = f"Dropout {dropout_ratio}" if dropout else "sin Dropout"

        with open(results_file, "a") as f:
            f.write(f"{i}. Modelo con {neuronas} neuronas por capa, {cantidad_capas} capas, {dropout_text}, optimizador {optimizer}, batch {batch_size}: \n")
            f.write(f"Coste final: {coste_train:.4f} (entrenamiento), {coste_test:.4f} (test)\n")
            f.write(f"Precision final: {accuracy_train:.4f} (entrenamiento), {accuracy_test:.4f} (test)\n\n")

    with open(results_file, "a") as f:
        f.write(f"Se ha escogido el modelo {index}")

    mejor_modelo.save('mejor_modelo.keras')

def cargar_modelo(ruta):
    if not os.path.exists(ruta):
        print("Error: No se encontró el modelo guardado.")
        ruta = input("Introduce la ruta del modelo: ")
        return cargar_modelo(ruta)
    else:
        return load_model(ruta)

def pedir_datos_usuario() :
    entrada_usuario = []

    print("\nIntroduce los datos del cliente solicitante:\n")
    entrada_usuario.append(float(input("Edad: ")))
    entrada_usuario.append(float(input("Sueldo anual: ")))
    entrada_usuario.append(float(input("Cantidad solicitada: ")))
    entrada_usuario.append(float(input("Puntuación crediticia: ")))
    entrada_usuario.append(float(input("Número de meses empleado: ")))
    entrada_usuario.append(float(input("Número de líneas de crédito abiertas: ")))
    entrada_usuario.append(float(input("Tasa de interés: ")))
    entrada_usuario.append(float(input("Período del préstamo en meses: ")))
    entrada_usuario.append(float(input("Ratio DTI: ")))
    entrada_usuario.append(int(input("Educación (0 = instituto, 1 = universidad, 2 = master, 3 = doctorado): ")))
    entrada_usuario.append(int(input("Tipo de empleo (0 = desempleado, 1 = autónomo, 2 = tiempo parcial, 3 = tiempo completo): ")))
    entrada_usuario.append(int(input("Estado civil (0 = divorciado, 1 = soltero, 2 = casado): ")))
    entrada_usuario.append(int(input("Tiene hipoteca actualmente (0 = no, 1 = sí): ")))
    entrada_usuario.append(int(input("Tiene personas dependientes a su cargo (0 = no, 1 = sí): ")))
    loan_purpose = int(input("Propósito del préstamo (0 = casa, 1 = estudios, 2 = coche, 3 = negocios, 4 = otros): "))
    entrada_usuario.append(int(input("Tiene co-firmante (0 = no, 1 = sí): ")))
    
    loan_purpose_encoded = [0, 0, 0, 0, 0]
    loan_purpose_encoded[loan_purpose] = 1
    entrada_usuario.extend(loan_purpose_encoded)

    entrada_np = np.array([entrada_usuario])

    return entrada_np

def predecir_resultado(modelo, datos_usuario):
    print("\nAnalizando solicitud... \n")
    resultado = modelo.predict(datos_usuario)
    print(f"El resultado final del análisis es de: {resultado[0][0]:.3f}")
    print("El clientes es problable que incumpla el préstamo." if resultado[0][0] > 0.5 else "El cliente es problable que cumpla el préstamo.")

