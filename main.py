"""
LEONARDO DANIEL AVIÑA NERI
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Correo: ld.avinaneri@ugto.mx
DESCRIPCION: l presente modelo de inteligencia artificial ha sido desarrollado con el objetivo de asistir en la detección del cáncer de tiroides mediante el análisis de imágenes. Sin embargo, es importante destacar que este modelo no garantiza resultados infalibles ni debe ser considerado como un sustituto del juicio clínico profesional.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Función para obtener la ruta correcta del archivo en un ejecutable PyInstaller
def resource_path(relative_path):
    """ Devuelve la ruta absoluta del archivo, maneja PyInstaller """
    try:
        # PyInstaller crea una carpeta temporal y almacena el archivo ahí
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Cargar el modelo entrenado usando resource_path para asegurar que lo encuentra en el .exe
try:
    model_path = resource_path('modelo_tirads_79.h5')
    print(f"Ruta del modelo cargada: {model_path}")  # Verifica la ruta en el .exe
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Función para preprocesar la imagen y predecir su clase
def cargar_imagen():
    file_path = filedialog.askopenfilename()
    
    # Verificar si el archivo existe
    if not file_path:
        resultado_label.config(text="Error: No se seleccionó ninguna imagen.")
        return

    try:
        # Intentar abrir la imagen y procesarla
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))  # Ajusta al tamaño de entrada esperado por tu modelo
        img_array = np.array(img) / 255.0  # Normalizar la imagen
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para el batch

        # Realizar predicción
        prediccion = model.predict(img_array)
        clase_predicha = np.argmax(prediccion)

        # Mostrar el resultado en la interfaz
        resultado_label.config(text=f"Es probable que la imagen se clasifique como TIRADS: {clase_predicha+1}")

        # Mostrar la imagen en la GUI
        img_tk = ImageTk.PhotoImage(img)
        imagen_label.config(image=img_tk)
        imagen_label.image = img_tk  # Esto evita que la imagen sea eliminada de la memoria
    except FileNotFoundError:
        resultado_label.config(text="Error: No se encontró el archivo seleccionado.")
    except AttributeError as e:
        resultado_label.config(text=f"Error de atributo: {str(e)}. Asegúrate de que el archivo de imagen es válido.")
    except Exception as e:
        resultado_label.config(text=f"Error inesperado: {str(e)}")

# Configuración de la ventana principal de Tkinter
ventana = tk.Tk()
ventana.title("Clasificador de TIRADS")
ventana.geometry("1200x900")  # Ancho x Alto

# Cargar el logo y mostrarlo usando resource_path para asegurar que lo encuentra en el .exe
try:
    logo_image = Image.open(resource_path('logo.png'))
    logo_image = logo_image.resize((100, 100))  # Ajustar el tamaño del logo
    logo_photo = ImageTk.PhotoImage(logo_image)
except Exception as e:
    print(f"Error al cargar el logo: {e}")

logo_label = Label(ventana, image=logo_photo)
logo_label.pack(pady=10)

# Texto de introducción centrado y con fuente personalizada
intro_label = Label(ventana, text="Bienvenido al modelo de detección de cáncer de tiroides.", 
                    font=("Helvetica", 14), anchor="w", justify="left")
intro_label.pack(fill="x", padx=20, pady=10)

# Botón para cargar imagen
cargar_boton = Button(ventana, text="Cargar Imagen", command=cargar_imagen, font=("Helvetica", 12))
cargar_boton.pack(pady=10)

# Etiqueta para mostrar el resultado de la predicción
resultado_label = Label(ventana, text="Es probable que la imagen se clasifique como TIRADS: ", font=("Helvetica", 12), anchor="w", justify="left")
resultado_label.pack(fill="x", padx=20, pady=5)

# Etiqueta para mostrar la imagen cargada
imagen_label = Label(ventana)
imagen_label.pack(pady=10)


# Label para los asuntos legales con texto más grande y justificado
ancho_pantalla = ventana.winfo_screenwidth()  # Obtener el ancho de la pantalla
legal_text = ("El presente modelo de inteligencia artificial ha sido desarrollado con el objetivo de asistir en la detección del cáncer de tiroides mediante el análisis de imágenes. Sin embargo, es importante destacar que este modelo no garantiza resultados infalibles ni debe ser considerado como un sustituto del juicio clínico profesional.")

legal_label = Label(ventana, text=legal_text, font=("Helvetica", 10), wraplength=ancho_pantalla // 2, 
                    anchor="w", justify="left")
legal_label.pack(padx=20, pady=20)

# Cargar el qr y mostrarlo usando resource_path para asegurar que lo encuentra en el .exe
qr_image = Image.open(resource_path('qr.png'))
qr_image = qr_image.resize((200, 200))  # Ajustar el tamaño del logo
qr_photo = ImageTk.PhotoImage(qr_image)

qr_label = Label(ventana, image=qr_photo)
qr_label.pack(pady=10)

# Firma del autor en la parte inferior
autor_label = Label(ventana, text="Desarrollado por Leonardo Daniel Aviña Neri", font=("Helvetica", 6), anchor="w", justify="left")
autor_label.pack(side="bottom", pady=10)

# Iniciar la aplicación
ventana.mainloop()
