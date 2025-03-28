# Predicción de Incumplimiento de Préstamos con Redes Neuronales

## Descripción del Proyecto

Este proyecto implementa una red neuronal multicapa para predecir la probabilidad de que un solicitante de préstamo incumpla su pago. Se basa en un conjunto de datos que contiene información relevante sobre los solicitantes y sus préstamos. El modelo se entrena con diferentes configuraciones y se selecciona automáticamente el que obtiene la mejor precisión.

## Estructura del Proyecto

El proyecto consta de los siguientes archivos:

* **modelo\_train.py**: Contiene el código para la carga y preprocesamiento de datos, la creación de la red neuronal y el proceso de entrenamiento y selección del mejor modelo.
* **modelo\_prueba.py**: Permite la carga del modelo entrenado y realiza predicciones con nuevos datos ingresados por el usuario.
* **utils.py**: Archivo con funciones auxiliares para la manipulación de datos, preprocesamiento y gestión de modelos.
* **loan\_modificado.csv**: Conjunto de datos utilizado para el entrenamiento.
* **mejor\_modelo.keras**: Archivo que almacena el modelo con mejor rendimiento durante el entrenamiento.
* **resultadoModelos.txt**: Registro de todas las configuraciones probadas y los resultados obtenidos.

## Dataset

El dataset utilizado contiene las siguientes columnas:

* **LoanID**: Identificador del préstamo (no relevante para el análisis).
* **Age**: Edad del solicitante.
* **Income**: Ingresos anuales.
* **LoanAmount**: Cantidad solicitada.
* **CreditScore**: Puntuación crediticia.
* **MonthsEmployed**: Meses trabajados.
* **NumCreditLines**: Número de líneas de crédito activas.
* **InterestRate**: Tasa de interés.
* **LoanTerm**: Duración del préstamo en meses.
* **DTIRatio**: Relación entre deuda e ingresos.
* **Education**: Nivel educativo.
* **EmploymentType**: Tipo de empleo.
* **MaritalStatus**: Estado civil.
* **HasMortgage**: Si el solicitante tiene hipoteca.
* **HasDependents**: Si tiene personas a su cargo.
* **LoanPurpose**: Propósito del préstamo.
* **HasCoSigner**: Si tiene un cofirmante.
* **Default**: Variable objetivo (1 = incumple, 0 = paga).

## Proceso de Entrenamiento

1. **Carga y preprocesamiento de datos**:
   * Se eliminan valores nulos.
   * Se realizan codificaciones one-hot y label encoding en variables categóricas.
   * Se normalizan las variables numéricas.
2. **Definición del modelo base**:
   * Capa de entrada con tantas neuronas como características tiene el dataset.
   * Dos capas ocultas con 4 neuronas cada una y activación ReLU.
   * Capa de salida con una única neurona y activación sigmoide para clasificación binaria.
3. **Ajuste de hiperparámetros**:
   * Se prueban distintas configuraciones variando el número de capas, neuronas, optimizadores y tamaños de batch.
   * Se automatiza el entrenamiento con todas las combinaciones definidas en *utils.py*.
   * Se guarda el mejor modelo según la precisión en test.

## Evaluación y Selección del Mejor Modelo

El archivo *resultadoModelos.txt* contiene los resultados de todas las configuraciones probadas. El modelo final seleccionado es el que obtuvo mejor precisión en el conjunto de test.

## Uso del Proyecto

1. **Entrenar el modelo**:

   ```
   python modelo_train.py
   ```

   Esto generará el archivo *mejor_modelo.keras*.
2. **Probar el modelo con nuevos datos**:

   ```
   python modelo_prueba.py
   ```

   Se solicitarán los datos del usuario por consola y se mostrará la predicción de si el préstamo se incumplirá o no.

## Requisitos

* Python 3.x
* TensorFlow
* NumPy
* Pandas
* scikit-learn

Para instalar las dependencias:

```
pip install tensorflow numpy pandas scikit-learn
```

## Contribución

Si deseas mejorar el modelo o probar nuevas configuraciones, puedes modificar *utils.py* para agregar más combinaciones de hiperparámetros y reentrenar el modelo.
