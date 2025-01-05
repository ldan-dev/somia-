"""
LEONARDO DANIEL AVIÑA NERI
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Correo: ld.avinaneri@ugto.mx
DESCRIPCION: l presente modelo de inteligencia artificial ha sido desarrollado con el objetivo de asistir en la detección del cáncer de tiroides mediante el análisis de imágenes. Sin embargo, es importante destacar que este modelo no garantiza resultados infalibles ni debe ser considerado como un sustituto del juicio clínico profesional.
"""

import os
import sys
import tensorflow as tf
from PIL import Image
from tkinter import filedialog, Label, Button, Tk
import numpy as np

# Si la aplicación es empaquetada, los recursos estarán en sys._MEIPASS
def resource_path(relative_path):
    try:
        # Cuando la app se empaqueta
        base_path = sys._MEIPASS
    except Exception:
        # Durante el desarrollo o ejecución normal
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Cargar el modelo desde la carpeta de recursos
model = tf.keras.models.load_model(resource_path('modelo_tirads_regularizado.h5'))

# Función para cargar y predecir la imagen
def cargar_imagen():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Cargar la imagen, convertir a RGB y preprocesar
    img = Image.open(file_path).convert('RGB')
    img = img.resize((150, 150))  # Ajusta al tamaño de entrada del modelo
    img_array = np.array(img) / 255.0  # Normaliza la imagen
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir la clase
    prediccion = model.predict(img_array)
    clase_predicha = np.argmax(prediccion)

    resultado_label.config(text=f"TIRADS predicho: {clase_predicha}")

    # Mostrar la imagen cargada en la GUI
    img_tk = ImageTk.PhotoImage(img)
    imagen_label.config(image=img_tk)
    imagen_label.image = img_tk

# Crear la ventana con Tkinter
ventana = Tk()
ventana.title("Clasificador de TIRADS")

# Cargar el logo
logo = Image.open(resource_path('logo.png'))
logo = logo.resize((200, 200))  # Redimensionar si es necesario
logo_tk = ImageTk.PhotoImage(logo)

logo_label = Label(ventana, image=logo_tk)
logo_label.pack()

# Botón para cargar imagen
cargar_boton = Button(ventana, text="Cargar Imagen", command=cargar_imagen)
cargar_boton.pack()

# Etiqueta para mostrar el resultado de la predicción
resultado_label = Label(ventana, text="TIRADS predicho:")
resultado_label.pack()

# Etiqueta para mostrar la imagen cargada
imagen_label = Label(ventana)
imagen_label.pack()

# Iniciar la aplicación
ventana.mainloop()
