import cv2
import matplotlib.pyplot
import imutils
import mtcnn .mtcnn
import os
#Almacenaremos las imagenes para entrenar al modelo
#Recordemos que este modelo es inteligencia artificial con machine learning

direccion = 'C:\\Users\\emman\\Desktop\\ProyectosAI\\Cubrebocas-Py\\Tapabocas'
nombre = 'Persona_con_Tapabocas'
carpeta = direccion + '\\'+nombre

if not os.path.exists(carpeta):
    os.makedirs(carpeta)
    
#Tomar video en tiempo real
detector = mtcnn.mtcnn.MTCNN()
cap = cv2.VideoCapture(0)
count = 0

#El while se ejecutara para que se pasen los fotogramas
while True:
    ret, frame = cap.read()
    #Pasamos a escalas de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #copia en tiempo real
    copia = frame.copy()
    
    #detectar rostros
    caras= detector.detect_faces(copia)
    
    #Almacenara las caras de 0 a 299
    for i in range (len(caras)):
        x1, y1, ancho, alto = caras[i]['box'] #Coordenadas de la cara , los box son los cuadros que se dibujan, con valores de los ojos etc. coordenadas de esquina superior izquierda
        x2, y2 = x1 + ancho, y1 + alto #Coordenadas de la esquina inferior derecha
        cara_reg = frame [y1:y2, x1:x2] #Cara registrada
        
        cara_reg = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC) #Redimensionar la cara , para que sea de 160x160 y todas del mismo tamaño para el entrenamiento
        #Almacenar las imagenes
        cv2.imwrite(carpeta + "//rostro_{}.jpg".format(count), cara_reg) #Se guardan las imagenes en la carpeta
        count = count + 1
    cv2.imshow('Entrenamiento', frame) #Mostrar el video en tiempo real
    
    t = cv2.waitKey(1) #Esperar 1 milisegundo
    if t == 27 or count >= 100: #Si se presiona la tecla ESC o se toman 100 imagenes, se sale del while
        break
    
cap.release()
cv2.destroyAllWindows()









#Sistema de python que reconoce rostros con cubrebocas
