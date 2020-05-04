# utilizando el dataset de PRAJNA https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator
# packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# argumento para terminal
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# inicializacion learning rate, epochs de entrenamiento
# batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# tomamos la lista de imagenes del dataset
# list y clase
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# ciclo for
for imagePath in imagePaths:
    # extraemos la clase del archivo
    label = imagePath.split(os.path.sep)[-2]

    # cargamos la imagen en (224x224) y se procesa
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # actualizamos informacion y label list
    data.append(image)
    labels.append(label)

# convertimos la informacion a NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot enconding a las etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# dividimos los datos en divisiones de entrenamiento 75%
# los datos de entrenamiento y el restante en pruebas
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construimos el generador de imagenes 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# cargamos la red MobileNetV2
# asegurando de que las capas FC esten paradas
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construimos la cabecera del modelo que estara
# como modelo base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# colocamos el modelo FC principal sobre el modelo base (esto se convertirá
# el modelo real que entrenaremos)
model = Model(inputs=baseModel.input, outputs=headModel)


# ciclo for sobre todas las capas en el modelo base y congelarlas para que puedan
# *no* se actualizará durante el primer proceso de capacitación
for layer in baseModel.layers:
	layer.trainable = False

# compilamos el modelo
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# entrenamos la cabeza de la red
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# hacemos predicciones 
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# para cada imagen en el conjunto de pruebas necesitamos encontrar el índice de la
# etiqueta con la mayor probabilidad pronosticada correspondiente
predIdxs = np.argmax(predIdxs, axis=1)

# mostramos el reporte de clasificacion
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serializamos el modelo en el disco
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# trazar la pérdida de entrenamiento y la precisión
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
