# packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # tomamos las dimensiones del marco y luego construimos un blob
    # con esas dimensiones
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    
    # pasamos el blob a través de la red y obtener las detecciones faciales
    faceNet.setInput(blob)
    detections = faceNet.forward()
   
    # inicializamos nuestra lista de caras, sus ubicaciones correspondientes,
    # y la lista de predicciones de nuestra red de mascarillas
    faces = []
	locs = []
	preds = []

    # ciclo for sobre las detecciones
    for i in range(0, detections.shape[2]):
        # extraer la confianza (es decir, la probabilidad) asociada con
        # la detección
        confidence = detections[0, 0, i, 2]

        # filtra las detecciones débiles asegurando que la confianza es
        # mayor que la confianza mínima
        if confidence > args["confidence"]:
            # introducimos las coordenadas (x,y)
            # para la caja
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

            # asegúrese de que los cuadros delimitadores caigan dentro de las dimensiones de
            # el marco
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extraer el ROI de la cara, convertirlo de BGR a canal RGB
            # pedidos, redimensionarlo a 224x224 y preprocesarlo
   			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

            # agregamos la cara y los cuadros delimitadores a sus respectivos
            # listas   
            faces.append(face)
			locs.append((startX, startY, endX, endY))      

    # prediction
    if len(faces) > 0
        # para una inferencia más rápida, haremos predicciones por lotes en * todos *
        # caras al mismo tiempo en lugar de predicciones una por una
        # en el bucle `for` anterior
        preds = maskNet.predict(faces)

    # devolver una tupla de 2 de las ubicaciones de la cara y sus correspondientes
    # ubicaciones
    return (locs, preds)

# construimos le argumento
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# cargamos el modelo del disco
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargamos el face mask detector model del disco
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# inicializamos la camara
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop sobre todos los frames del video
while True:
    # agarra el fotograma de la secuencia de video enhebrada y cambia el tamaño
    # para tener un ancho máximo de 400 píxeles
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

    # detectar caras en el marco y determinar si llevan puesto un
    # mascarilla o no
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # bucle sobre las ubicaciones de caras detectadas y sus correspondientes
    # ubicaciones
	for (box, pred) in zip(locs, preds):
		# bounding box unpack!
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		
        # determinar la etiqueta
		# y color de la caja
		label = "Con cubrebocas" if mask > withoutMask else "Sin cubrebocas"
		color = (0, 255, 0) if label == "Con cubrebocas" else (0, 0, 255)
		
        # porcentaje de probabilidad
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
        # mostrar label
        # y caja
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostrar output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# detener el loop si presionas q
	if key == ord("q"):
		break
# clean!
cv2.destroyAllWindows()
vs.stop()
