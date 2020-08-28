import json
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as io2

# leer y abrir los objetos json
with open('images/Deadpool 006-007.json') as f:
    data = json.load(f)

# data['shapes'] para ver sus atributillos
# data['imageData'] imagen codificada como string dentro del json
img_bytes = base64.b64decode(data['imageData'])  # saco de la string los bytes en base 64
image = Image.open(io.BytesIO(img_bytes))  # los convierto a imagen
image_array = np.array(image)
height=image_array.shape[0]
width=image_array.shape[1]
detections_coords = data['shapes']
# creo una lista para almacenar solo las detecciones
detections = []
# creo una copia de la imagen original para modificar y dejar en blanco las detecciones
# para así después sacar el background
image_array_copy = image_array.copy()
# creamos una lista en la que guardar cuadrados del fondo
bg = []
# iteramos sobre todas las deteciones que tenga la imagen
for dc in detections_coords:
    # sacamos las coordenadas de la primera bounding box
    # remember: array de numpy es (fila, columna), pero las coordenadas de la bounding box
    # vienen con formato de imagen (columna, fila)
    ymin = round(dc["points"][0][1])
    ymax = round(dc["points"][1][1])
    xmin = round(dc["points"][0][0])
    xmax = round(dc["points"][1][0])
    # sacamos la deteccion actual y la añadimos a la lista de detecciones
    detections.append(image_array[ymin:ymax, xmin:xmax])
    # dejamos en negro las zonas de las detecciones
    image_array_copy[ymin:ymax, xmin:xmax] = (0, 255, 0)
# A continucación viene el proceso de extraer el fondo de la imagen,
# es decir, lo que no está dentro de las bounding boxes
# elegimos número de divisones del fondo
numparts=10
# dividimps según el tamaño de la imagen y el numparts elegido
for i in range(numparts):
    # dividimos a lo largo de ambos ejes cada vez, y lo apendeamos
    bg.append(np.array_split(np.array_split(image_array_copy,numparts,axis=0)[i],numparts,axis=1))
# como nos queda una lista de listas, la desenrollamos
bg=[item for sublist in bg for item in sublist]
# creamos una imagen del tamaño de los cachos de fondo toda a verde
reference=np.zeros(bg[0].shape)
reference[:,:,1]=255

# mostramos la imagen recortada, solo con la bounding box
# (es un numpy array, por eso usamos io2.show)
# io2.imshow(image_array[ymin:ymax,xmin:xmax])
