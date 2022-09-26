#Importar librerias
import cv2
import os
import numpy as np
import mediapipe as mp
import mtcnn.mtcnn 

#Importar los nombres de las carpetas

direccion = 'C:\\Users\\emman\\Desktop\\ProyectosAI\\Cubrebocas-Py\\Tapabocas'
etiquetas = os.listdir(direccion) #Lista de carpetas, para importar las fotos como si fueran un vector
print("Nombres: " + str(etiquetas))

#Importar el modelo
modelo= cv2.face.LBPHFaceRecognizer_create()

#Leer el modelo
modelo.read('modeloLBPHFace.xml')

#Capturamos el video en tiempo real
detector = mtcnn.mtcnn.MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False: break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()
    copia2 = gris.copy()
    
    caras = detector.detect_faces(copia)
    
    for i in range (len(caras)):
        x1, y1, ancho, alto = caras[i]['box']
        x2, y2 = x1 + ancho, y1 + alto
        cara_reg = copia2 [y1:y2, x1:x2]
        cara_rec = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC)
        resultado = modelo.predict(cara_rec)
        
        #Mostramos en pantalla los resultados
        if resultado[0] == 0:
            cv2.putText(frame, etiquetas[0], (x1,y1-5), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame,(x1,y1), (x1+ancho, y1+alto), (0,255,0), 2)
        elif resultado[0] == 1:
            cv2.putText(frame, etiquetas[1], (x1,y1-5), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame,(x1,y1), (x1+ancho, y1+alto), (0,255,0), 2)
    cv2.imshow('Reconocimiento', frame)
    
    t= cv2.waitKey(1)
    if t == 27:
        break
    
cap.release()
cv2.destroyAllWindows()