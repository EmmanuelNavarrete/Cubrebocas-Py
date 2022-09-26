#Entrenar el modelo
import cv2
import os
import numpy as np #Para operaciones matriciales y vectores

direccion = 'C:\\Users\\emman\\Desktop\\ProyectosAI\\Cubrebocas-Py\\Tapabocas'
lista = os.listdir(direccion) #Lista de carpetas, para importar las fotos como si fueran un vector

etiquetas = [] #Etiquetas de las personas
rostros = [] #Rostros de las personas
 #Como es que la computadora nos entiende , con 0 y 1
cont = 0
 
for nameDir in lista:
     nombre = direccion + '\\' + nameDir
     
     for fileName in os.listdir(nombre): # Para cada archivo en la carpeta
         etiquetas.append(cont) #Agregamos la etiqueta
         rostros.append(cv2.imread(nombre + '\\' + fileName, 0)) #Agregamos el rostro
     cont = cont + 1
     
#Creamos el modelo
reconocimiento = cv2.face.LBPHFaceRecognizer_create() #importamos opencv-contrib-python para usar este modelo
 #Entrenamos el modelo
reconocimiento.train(rostros, np.array(etiquetas)) #Entrenamos el modelo con los rostros y las etiquetas
#Guardar el modelo
reconocimiento.write('modeloLBPHFace.xml') #Guardamos el modelo
print('Modelo entrenado')


