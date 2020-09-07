import json
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as io2
import cv2
import glob
import os

# creamos una lista con los paths a los json
paths_json = []
paths_json = glob.glob("images/*.json")
# creamos una lista con los paths a los png
# paths_jpg = []
# paths_jpg = glob.glob("images/*.jpg")

# creamos los directorios donde irán el bg y las detecciones (bounding bpxes)
os.makedirs('images/bg')
os.makedirs('images/bounding_boxes')

# bucle principal
# en él se irán abriendo los json y sacando fragmentos del bg y las bounding boxes
# para guardarlas como imágenes y entrenar con ellas después
for p in paths_json:
    # leer y abrir los objetos json
    with open(p) as f:
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
    for i, dc in enumerate(detections_coords):
        # sacamos las coordenadas de la primera bounding box
        # remember: array de numpy es (fila, columna), pero las coordenadas de la bounding box
        # vienen con formato de imagen (columna, fila)
        ymin = int(min(dc["points"][0][1],dc["points"][1][1]))
        ymax = int(max(dc["points"][0][1],dc["points"][1][1]))
        xmin = int(min(dc["points"][0][0],dc["points"][1][0]))
        xmax = int(max(dc["points"][0][0],dc["points"][1][0]))
        # sacamos la deteccion actual y la añadimos a la lista de detecciones
        if dc["label"] == 'bg':
            # bg.append(image_array[ymin:ymax, xmin:xmax]) si queremos guardar los bg de esta img una lista
            img = Image.fromarray(image_array[ymin:ymax, xmin:xmax], 'RGB')
            # creamos el path y el nombre con el que guardaremos la imagen
            path = 'images/bg/' + p.split(os.sep)[1].split('.')[0] + '_' + str(i) + '.jpg'
            img.save(path)
        else:
            # detections.append(image_array[ymin:ymax, xmin:xmax]) si queremos guardar las bb de esta img n¡en una lista
            img = Image.fromarray(image_array[ymin:ymax, xmin:xmax], 'RGB')
            # creamos el path y el nombre con el que guardaremos la imagen
            path = 'images/bounding_boxes/' +p.split('\\')[1].split('.')[0] + '_' + dc["label"] + '_' + str(i) + '.jpg'
            img.save(path)

# # A continucación viene el proceso de extraer el fondo de la imagen,
# # es decir, lo que no está dentro de las bounding boxes
# # elegimos número de divisones del fondo
# numparts=10
# # dividimps según el tamaño de la imagen y el numparts elegido
# for i in range(numparts):
#     # dividimos a lo largo de ambos ejes cada vez, y lo apendeamos
#     bg.append(np.array_split(np.array_split(image_array_copy,numparts,axis=0)[i],numparts,axis=1))
# # como nos queda una lista de listas, la desenrollamos
# bg=[item for sublist in bg for item in sublist]
# # creamos una imagen del tamaño de los cachos de fondo toda a verde
# reference=np.zeros(bg[0].shape)
# reference[:,:,1]=255
# # comprobamos si más de un 60% de los pixeles de cada cuadrado son verdes (no fondo)
# size_bg=bg[0].size
# # if (sum(sum(sum(bg[1]==reference)))/size_bg >= 0.6)
# # mostramos la imagen recortada, solo con la bounding box
# # (es un numpy array, por eso usamos io2.show)
# # io2.imshow(image_array[ymin:ymax,xmin:xmax])
