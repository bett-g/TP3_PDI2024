import cv2
import numpy as np

# Abro los videos
cap1 = cv2.VideoCapture(r'C:\Users\solki\OneDrive\Documentos\TP3_PDI\TP3_PDI2024\videos\ruta_1.mp4')
cap2 = cv2.VideoCapture(r'C:\Users\solki\OneDrive\Documentos\TP3_PDI\TP3_PDI2024\videos\ruta_2.mp4')

if not cap1.isOpened() or not cap2.isOpened():
    print("Error al abrir uno de los videos")
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

# Defino la función para extraer la región de interés
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

# Defino la función para dibujar las líneas en azul
def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Proceso ambos videos simultáneamente
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    if ret1:
        # Procesamiento del primer video
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Ajuste de los parámetros de Canny
        edges1 = cv2.Canny(gray1, 50, 150)
        
        # Aplicación de la región de interés
        roi_edges1 = region_of_interest(edges1, roi_vertices1)
        
        # Detección de líneas con HoughLinesP
        lines1 = cv2.HoughLinesP(roi_edges1, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        
        # Filtrado de líneas basado en la pendiente y posición
        if lines1 is not None:
            filtered_lines1 = []
            for line in lines1:
                x1, y1, x2, y2 = line[0]
                # Calcular la pendiente (evitar divisiones por cero)
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    # Filtrar líneas con pendientes y posiciones específicas
                    if 0.5 < abs(slope) < 2.0 and min(y1, y2) > height1 // 2:
                        filtered_lines1.append(line)
            
            # Dibujar las líneas filtradas
            draw_lines(frame1, filtered_lines1)

        out1.write(frame1)
        cv2.imshow('Frame 1', frame1)

    if ret2:
        # Procesamiento del segundo video
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Ajuste de los parámetros de Canny
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # Aplicación de la región de interés
        roi_edges2 = region_of_interest(edges2, roi_vertices2)
        
        # Detección de líneas con HoughLinesP
        lines2 = cv2.HoughLinesP(roi_edges2, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        
        # Filtrado de líneas basado en la pendiente y posición
        if lines2 is not None:
            filtered_lines2 = []
            for line in lines2:
                x1, y1, x2, y2 = line[0]
                # Calcular la pendiente (evitar divisiones por cero)
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    # Filtrar líneas con pendientes y posiciones específicas
                    if 0.5 < abs(slope) < 2.0 and min(y1, y2) > height2 // 2:
                        filtered_lines2.append(line)
            
            # Dibujar las líneas filtradas
            draw_lines(frame2, filtered_lines2)

        out2.write(frame2)
        cv2.imshow('Frame 2', frame2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libero los recursos
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
