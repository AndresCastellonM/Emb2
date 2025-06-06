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
        self.modelList = ["MODELS/modelo_CNN.tflite", "MODELS/modelo_CNN.h5", "MODELS/modelo1.tflite", "MODELS/modelo2.tflite", "MODELS/modelo3.tflite"]
        self.modelH5 = None
        self.interpreterTF = None
        self.model_index = model_index
        self.PredictionsList = [["Coca", "Pepsi", "Fanta", "Nada"], ["Coca", "Pepsi", "Fanta", "Nada"], ["Coca", "Pepsi", "Fanta"], ["Coca", "Pepsi", "Fanta"], ["Coca", "Pepsi", "Fanta"]] #debe ser correspondiente a los modelos puestos en modelList

        if self.modelList[self.model_index].endswith(".h5"):
            self.modelH5 = load_model(self.modelList[self.model_index])
        elif self.modelList[self.model_index].endswith(".tflite"):
            self.interpreterTF = tflite.Interpreter(model_path=self.modelList[self.model_index])
            self.interpreterTF.allocate_tensors()
        else:
            raise ValueError(f"[ERROR] Formato de modelo no soportado: {self.modelList[self.model_index]}")
    def preprocess(self, img):
        if self.model_index == 0:
            img = cv2.resize(img, (200,200))
            img = cv2.GaussianBlur(img, (5,5), 0)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        elif self.model_index == 1:
            img = cv2.resize(img, (200,200))
            img = cv2.GaussianBlur(img, (5,5), 0)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        elif self.model_index == 2 or self.model_index==3 or self.model_index==4:
            img = cv2.resize(img, (224,224))
            img = cv2.GaussianBlur(img, (5,5), 0)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

    def Predict(self, image):
        """
        model_index: índice en self.modelList
        image: imagen BGR como array (ej. capturada con OpenCV)
        """
        input_data = self.preprocess(image)
        if self.modelH5 is not None:
            predictionArray = self.modelH5.predict(input_data)
            predictionText = self.PredictionsList[self.model_index][np.argmax(predictionArray)]
            predictionConf = np.max(predictionArray)
        elif self.interpreterTF is not None:
            input_details = self.interpreterTF.get_input_details()
            output_details = self.interpreterTF.get_output_details()
            self.interpreterTF.set_tensor(input_details[0]['index'], input_data)
            self.interpreterTF.invoke()
            predictionArray = self.interpreterTF.get_tensor(output_details[0]['index'])
            predictionText = self.PredictionsList[self.model_index][np.argmax(predictionArray)]
            predictionConf = np.max(predictionArray)
        else:
            raise RuntimeError("[ERROR] Modelo no cargado correctamente.")

        return predictionArray, predictionText, predictionConf
