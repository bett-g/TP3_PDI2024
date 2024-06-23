"""
1)a) Desarrollar un algoritmo para detectar automaticamene el carril por el cual 
esta circulando el auto, mediante la detección de las dos lineas que lo delimitan. 
En la figura 2 se muestra un ejemplo del resultado esperado

Para detectar líneas, puede utilizar la función cv2.HoughLinesP(img, rho, theta, threshold, lines, 
minLineLength, maxLineGap). 

Los parámetros img, rho, theta y threshold tienen el mismo significado que 
para cv2.HoughLines(), el parámetro lines no lo utilizaremos (le asignamos el valor np.array([])), minLineLength 
representa la longitud mínima de línea permitida y maxLineGap representa la distancia máxima permitida entre puntos 
de una misma línea para enlazarlos. A diferencia de cv2.HoughLines(), cada línea encontrada se representa mediante 
un vector de 4 elementos (x1,y1,x2,y2), donde (x1,y1) e (x2,y2) son los puntos finales de cada segmento de línea detectado.

cv2.HoughLines detecta líneas completas y las representa en el espacio de Hough, mientras que cv2.HoughLinesP 
detecta segmentos de línea y los representa mediante sus puntos finales, ofreciendo más flexibilidad 
y eficiencia para ciertas aplicaciones.

1)b) Generar videos donde se muestren las lineas que definen el carril resaltadas en color azul.

Se puede crea una máscara para definir una ROI de forma poligonal utilizando cv2.fillPoly(img, points, color), 
donde img es la imagen donde se dibujará el polígono (por ejemplo np.zeros((H,W),dtype=np.uint8)), 
points son los vértices que representan al polígono y color es el color con el cual se rellena el polígono 
(por ejemplo 255).

"""
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

# Rutas de los videos de entrada
video1_filename = 'videos/ruta_1.mp4'
video2_filename = 'videos/ruta_2.mp4'

# Abro los videos de entrada
cap1 = cv2.VideoCapture(video1_filename)
cap2 = cv2.VideoCapture(video2_filename)

# Verifico que los videos se abrieron correctamente
if not cap1.isOpened():
    print(f'Error al abrir el video: {video1_filename}')
    exit()

if not cap2.isOpened():
    print(f'Error al abrir el video: {video2_filename}')
    cap1.release()
    exit()

# Obtengo las propiedades de los videos
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
n_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
n_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

# Creo los objetos para escribir los videos de salida
out1_filename = 'output_ruta_1.mp4'
out2_filename = 'output_ruta_2.mp4'

out1 = cv2.VideoWriter(out1_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps1, (width1, height1))
out2 = cv2.VideoWriter(out2_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps2, (width2, height2))

# Verifico que los archivos de salida se hayan creado correctamente
if not out1.isOpened():
    print(f'Error al abrir el archivo de salida: {out1_filename}')
    cap1.release()
    cap2.release()
    exit()

if not out2.isOpened():
    print(f'Error al abrir el archivo de salida: {out2_filename}')
    cap1.release()
    out1.release()
    cap2.release()
    exit()

# Defino las regiones de interés para ambos videos
roi_vertices1 = np.array([
    [(0, height1), 
     (width1 // 2 - 100, height1 // 2 + 40), 
     (width1 // 2 + 100, height1 // 2 + 40), 
     (width1, height1)]
], dtype=np.int32)

roi_vertices2 = np.array([
    [(0, height2), 
     (width2 // 2 - 100, height2 // 2 + 40), 
     (width2 // 2 + 100, height2 // 2 + 40), 
     (width2, height2)]
], dtype=np.int32)

# Proceso ambos videos simultáneamente
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    if ret1:
        # Procesamiento del primer video
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        edges1 = cv2.Canny(blur1, 50, 150)

        roi_edges1 = region_of_interest(edges1, roi_vertices1)

        lines1 = cv2.HoughLinesP(roi_edges1, rho=1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=100)

        if lines1 is not None:
            longest_lines1 = sorted(lines1, key=lambda x: np.sqrt((x[0][2] - x[0][0]) ** 2 + (x[0][3] - x[0][1]) ** 2), reverse=True)[:2]

            for line in longest_lines1:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out1.write(frame1)
        cv2.imshow('Frame 1', frame1)
        # Visualización de bordes detectados
 #       cv2.imshow('Edges1', edges1)
    if ret2:
        # Procesamiento del segundo video
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        edges2 = cv2.Canny(blur2, 50, 200)

        roi_edges2 = region_of_interest(edges2, roi_vertices2)

        lines2 = cv2.HoughLinesP(roi_edges2, rho=1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=100)

        if lines2 is not None:
            longest_lines2 = sorted(lines2, key=lambda x: np.sqrt((x[0][2] - x[0][0]) ** 2 + (x[0][3] - x[0][1]) ** 2), reverse=True)[:2]

            for line in longest_lines2:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out2.write(frame2)
        cv2.imshow('Frame 2', frame2)
        # Visualización de bordes detectados
 #       cv2.imshow('Edges2', edges2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libero los recursos
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()