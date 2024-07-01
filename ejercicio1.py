import cv2
import numpy as np

# Función para procesar cada video
def procesar_video(input_filename, output_filename):
    cap = cv2.VideoCapture(input_filename)
    if not cap.isOpened():
        print(f"Error al abrir el video: {input_filename}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Definición de los parámetros de las regiones de interés
    roi_vertices = np.array([
        [int(width * 0.05), height],
        [int(width * 0.95), height],
        [int(width * 0.55), int(height * 0.6)],
        [int(width * 0.45), int(height * 0.6)]
    ], dtype=np.int32)

    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        print(f"Error al abrir el archivo de salida: {output_filename}")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesamiento del frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        roi_edges = region_de_interes(edges, roi_vertices)
        lines = cv2.HoughLinesP(roi_edges, rho=3, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=200)
        
        # Filtrado y dibujo de líneas
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if 0.5 < abs(slope) < 2.0 and min(y1, y2) > height // 2:
                        filtered_lines.append(line)
            dibujar_lineas(frame, filtered_lines)

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

# Función para definir la región de interés
def region_de_interes(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)

# Función para dibujar líneas
def dibujar_lineas(img, lines, color=(255, 0, 0), thickness=3):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Procesamiento de ambos videos
procesar_video('videos/ruta_1.mp4', 'output_ruta_1.mp4')
procesar_video('videos/ruta_2.mp4', 'output_ruta_2.mp4')

cv2.destroyAllWindows()

