import numpy as np
import tensorflow as tf
import utils as utils


modelo = utils.cargar_modelo("mejor_modelo.keras")

datos_usuario = utils.pedir_datos_usuario()
utils.predecir_resultado(modelo, datos_usuario)
