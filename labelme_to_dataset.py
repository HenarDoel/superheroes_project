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
detections=data['shapes']

# sacamos las coordenadas de la primera bounding box
# remember: array de numpy es (fila, columna), pero las cooridnadas de la bounding box
# vienen con formato de imagen (columna, fila)
ymin=round(detections[0]["points"][0][1])
ymax=round(detections[0]["points"][1][1])
xmin=round(detections[0]["points"][0][0])
xmax=round(detections[0]["points"][1][0])

# mostramos la imagen recortada, solo con la bounding box
# (es un numpy array, por eso usamos io2.show)
io2.imshow(image_array[ymin:ymax,xmin:xmax])