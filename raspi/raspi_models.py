#import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
    print("Se importo tensorflow.lite as tflite")


class ModelosCNN:
    def __init__(self, model_index):
        self.modelList = ["MODELS/modelo1.h5", "MODELS/modelo2.tflite"]
        self.models = []
        self.interpreters = []
        self.model_index = model_index
        self.PredictionsList = [["Coca", "Pepsi", "Fanta"], ["Coca", "Pepsi", "Fanta"]] #debe ser correspondiente a los modelos puestos en modelList

        if self.modelList[self.model_index].endswith(".h5"):
            self.models.append(tf.keras.models.load_model(self.modelList[self.model_index]))
            self.interpreters.append(None)
        elif self.modelList[self.model_index].endswith(".tflite"):
            interpreter = tflite.Interpreter(model_path=self.modelList[self.model_index])
            interpreter.allocate_tensors()
            self.models.append(None)
            self.interpreters.append(interpreter)
        else:
            raise ValueError(f"[ERROR] Formato de modelo no soportado: {self.modelList[self.model_index]}")
    def preprocess(self, img):
        if self.model_index == 1:
            img = cv2.resize(img, (200,200))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        if self.model_index == 2:
            img = cv2.resize(img, (224,224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

    def Predict(self, image):
        """
        model_index: índice en self.modelList
        image: imagen BGR como array (ej. capturada con OpenCV)
        """
        input_data = self.preprocess(image)

        model = self.models[self.model_index]
        interpreter = self.interpreters[self.model_index]

        if model is not None:
            predictionArray = model.predict(input_data)
            predictionText = self.PredictionsList[self.model_index][np.argmax(predictionArray)]
            predictionConf = np.max(predictionArray)
        elif interpreter is not None:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictionArray = interpreter.get_tensor(output_details[0]['index'])
            predictionText = self.PredictionsList[self.model_index][np.argmax(predictionArray)]
            predictionConf = np.max(predictionArray)
        else:
            raise RuntimeError("[ERROR] Modelo no cargado correctamente.")

        return predictionArray, predictionText, predictionConf
