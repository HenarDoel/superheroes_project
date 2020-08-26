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
img_bytes = base64.b64decode(data['imageData']) # saco de la string los bytes en base 64
image = Image.open(io.BytesIO(img_bytes)) # los convierto a imagen
image_array=np.array(image)
detections_coords=data['shapes']
# creo una lista para almacenar solo las detecciones
detections = []
# iteramos sobre todas las deteciones que tenga la imagen
for dc in detections_coords:
    # sacamos las coordenadas de la primera bounding box
    # remember: array de numpy es (fila, columna), pero las cooridnadas de la bounding box
    # vienen con formato de imagen (columna, fila)
    ymin = round(dc["points"][0][1])
    ymax = round(dc["points"][1][1])
    xmin = round(dc["points"][0][0])
    xmax = round(dc["points"][1][0])
    # sacamos la deteccion actual y la añadimos a la lista de detecciones
    detections.append(image_array[ymin:ymax, xmin:xmax])

# mostramos la imagen recortada, solo con la bounding box
# (es un numpy array, por eso usamos io2.show)
#io2.imshow(image_array[ymin:ymax,xmin:xmax])