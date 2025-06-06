import json
import smtplib
from email.message import EmailMessage
from datetime import datetime
import os
import re
import numpy as np

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_EMISOR = 'castellon.andresf@gmail.com'
EMAIL_CLAVE = 'vgaguqvdfduuavqp'
EMAIL_RECEPTOR = 'castellon.andresf@gmail.com'

class Send_data:
    def __init__(self, usuario, soda_brands):
        self.usuario = usuario
        self.date = datetime.now().strftime("%Y/%m/%d")
        self.soda_brands = soda_brands

        if np.all(np.array(list(soda_brands.values())) == 0):
            self.email_error()
        else:
            self.opcion()

    def save_json(self, base_name="logCount", carpeta="."):
        data = {
            "USER": self.usuario,
            "DATE": self.date,
            "Counting": self.soda_brands
        }
        archivos_existentes = os.listdir(carpeta)
        patron = re.compile(rf"{re.escape(base_name)}\((\d+)\)\.json")
        numeros = [int(match.group(1)) for archivo in archivos_existentes if (match := patron.match(archivo))]
        siguiente = max(numeros) + 1 if numeros else 1
        nombre_archivo = os.path.join(carpeta, f"{base_name}({siguiente}).json")
        with open(nombre_archivo, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Archivo guardado como: {nombre_archivo}")

    def email_log(self, asunto=None, mensaje=None):
        if mensaje is None:
            mensaje = f"Usuario: {self.usuario}\nFecha: {self.date}\n\nResultados de detección:\n"
            for brand, count in self.soda_brands.items():
                mensaje += f"- {brand}: {count} \n"

        msg = EmailMessage()
        msg["Subject"] = asunto or f"Resultados de detección - {self.usuario}"
        msg["From"] = EMAIL_EMISOR
        msg["To"] = EMAIL_RECEPTOR
        msg.set_content(mensaje)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_EMISOR, EMAIL_CLAVE)
                server.send_message(msg)
            print("Correo enviado correctamente.")
        except Exception as e:
            print(f"Error al enviar el correo: {e}")
            self.save_local(mensaje)

    def email_error(self):
        mensaje = f"Usuario: {self.usuario}\n\nNo bottles were detected on date {self.date}"
        msg = EmailMessage()
        msg["Subject"] = f"Resultados de detección - {self.usuario}"
        msg["From"] = EMAIL_EMISOR
        msg["To"] = EMAIL_RECEPTOR
        msg.set_content(mensaje)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_EMISOR, EMAIL_CLAVE)
                server.send_message(msg)
            print("Correo enviado correctamente.")
        except Exception as e:
            print("ERROR: ")
            print(mensaje)
            self.save_local(mensaje)

    def save_local(self, mensaje):
        base_name = "log"
        extension = ".txt"
        carpeta = "."
        archivos_existentes = os.listdir(carpeta)
        patron = re.compile(rf"{re.escape(base_name)}\((\d+)\){re.escape(extension)}")
        numeros = [int(match.group(1)) for archivo in archivos_existentes if (match := patron.match(archivo))]
        siguiente = max(numeros) + 1 if numeros else 1
        archivo = os.path.join(carpeta, f"{base_name}({siguiente}){extension}")

        try:
            with open(archivo, 'w') as f:
                f.write(mensaje)
            print(f"Log guardado en {archivo}")
        except Exception as e:
            print(f"Error al guardar log local: {e}")

    def descartar(self):
        print("Datos descartados.")

    def opcion(self):
        while True:
            ans = input("What do you want to do with the log? type email, json or discard: ")
            if ans == 'email':
                self.email_log()
                break
            elif ans == 'json':
                self.save_json()
                break
            elif ans == 'discard':
                self.descartar()
                break
            else:
                print("Invalid choice, please try again\n")


if __name__ == "__main__":
    datos_de_prueba = {
        "CocaCola": 0,
        "Pepsi": 0,
        "Sprite": 0
    }

    sender = Send_data("Andy", datos_de_prueba)
