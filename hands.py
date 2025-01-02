import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import pygame
from random import randint

# Inicialización de MediaPipe y variables
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None

def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detection_result
    detection_result = result

def draw_landmarks_on_image(rgb_image, detection_result):
    # Dibuja las landmarks en la imagen
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return annotated_image

# Configuración de Pygame para ventana de 1920x1080
pygame.init()
screen_width, screen_height = 1920, 1080  # Tamaño fijo de la ventana
screen = pygame.display.set_mode((screen_width, screen_height))  # Ventana de tamaño fijo
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Variables del juego
score = 0
square_size = 80  # Tamaño del cuadrado más grande
square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
square_timer = 0
square_lifetime = 10  # Ahora el cuadrado vive durante 10 segundos

# Crear la bola que se moverá con la mano
ball_radius = 40  # Radio de la bola más grande
ball_x, ball_y = screen_width // 2, screen_height // 2  # Posición inicial de la bola en el centro de la pantalla

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

# Inicialización de la cámara
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    running = True
    while cap.isOpened() and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Obtener las dimensiones de la imagen de la cámara
        camera_height, camera_width, _ = image.shape

        # Redimensionar la imagen de la cámara para que encaje en la ventana de 1920x1080
        image_resized = cv2.resize(image, (screen_width, screen_height))

        # Procesar la imagen de la cámara
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_resized)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Convertir la imagen de OpenCV (BGR) a Pygame (RGB)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        if detection_result is not None and len(detection_result.hand_landmarks) > 0:
            landmarks = detection_result.hand_landmarks[0]  # Solo usamos la primera mano detectada

            # Obtener coordenadas de los puntos del pulgar y meñique
            thumb_tip = landmarks[4]  # Pulgar (landmark 4)
            pinky_tip = landmarks[20]  # Meñique (landmark 20)
            
            # Calcular el centro de la mano (promedio de las coordenadas del pulgar y el meñique)
            center_x = (thumb_tip.x + pinky_tip.x) / 2
            center_y = (thumb_tip.y + pinky_tip.y) / 2

            # Convertir las coordenadas normalizadas a la resolución de la pantalla
            screen_x = int(center_x * screen_width)
            screen_y = int(center_y * screen_height)

            # Dibujar las landmarks sobre la imagen
            image_rgb = draw_landmarks_on_image(image_rgb, detection_result)

            # Mover la bola con la mano (posicionar la bola en el centro de la mano)
            ball_x = screen_width - screen_x  # Invertir la coordenada x para compensar el espejo
            ball_y = screen_y

            # Verificar si la bola colide con el cuadrado
            if (square_position[0] <= ball_x <= square_position[0] + square_size and
                square_position[1] <= ball_y <= square_position[1] + square_size):
                score += 1
                square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
                square_timer = 0  # Reiniciar el tiempo del cuadrado

        # Actualizar el temporizador del cuadrado
        square_timer += clock.get_time() / 1000
        if square_timer >= square_lifetime:
            square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
            square_timer = 0

        # Convertir la imagen procesada a formato Pygame
        frame_surface = pygame.surfarray.make_surface(np.rot90(image_rgb))

        # Dibujar elementos en pantalla
        screen.blit(frame_surface, (0, 0))  # Imagen de la cámara como fondo
        pygame.draw.rect(screen, (255, 0, 0), (*square_position, square_size, square_size))  # Cuadrado rojo
        pygame.draw.circle(screen, (0, 0, 255), (ball_x, ball_y), ball_radius)  # Bola azul

        score_text = font.render(f"Puntos: {score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        clock.tick(60)

cap.release()
pygame.quit()
