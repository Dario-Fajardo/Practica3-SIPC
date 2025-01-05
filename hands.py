import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import pygame
from random import randint, choice
import math

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

# Configuración de Pygame para ventana ajustada al tamaño de la pantalla
pygame.init()
screen_info = pygame.display.Info()  # Obtener información de la pantalla
screen_width, screen_height = screen_info.current_w, screen_info.current_h  # Resolución de la pantalla
screen = pygame.display.set_mode((screen_width, screen_height))  # Ventana en modo ventana, no pantalla completa
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Variables del juego
score = 0
square_size = 80  # Tamaño de la forma
square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
square_timer = 0
square_lifetime = 10  # Ahora la forma vive durante 10 segundos

# Crear la bola que se moverá con la mano
ball_radius = 40  # Radio de la bola más grande
ball_x, ball_y = screen_width // 2, screen_height // 2  # Posición inicial de la bola en el centro de la pantalla

# Generar figura aleatoria (puede ser 'circle', 'square' o 'triangle')
def generate_random_shape():
    return choice(['circle', 'square', 'triangle'])

# Inicializar forma aleatoria para la bola y el cuadrado
current_shape = generate_random_shape()

# Función para calcular el ángulo entre dos puntos
def calculate_angle(p1, p2):
    # Usamos la fórmula para calcular el ángulo entre dos puntos
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))  # Convertimos el ángulo a grados
    return angle

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

# Inicialización de la cámara
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    running = True

    # Ángulo de rotación de la figura estática (aleatorio y fijo por iteración)
    static_angle = randint(0, 360)  # Ángulo estático aleatorio

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

        # Redimensionar la imagen de la cámara para que encaje en la ventana
        image_resized = cv2.resize(image, (screen_width, screen_height))

        # Procesar la imagen de la cámara
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_resized)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Convertir la imagen de OpenCV (BGR) a Pygame (RGB)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        angle = 0

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

            # Calcular el ángulo de rotación basado en el pulgar y el meñique
            angle = calculate_angle((thumb_tip.x, thumb_tip.y), (pinky_tip.x, pinky_tip.y))

            # Verificar si la bola colide con la forma
            if (square_position[0] <= ball_x <= square_position[0] + square_size and
                square_position[1] <= ball_y <= square_position[1] + square_size):
                # Comprobar si los ángulos coinciden (con tolerancia de margen para la rotación)
                if abs(angle - static_angle) < 5:  # Si la diferencia de ángulo es menor a 5 grados
                    score += 1
                    # Cambiar la figura aleatoriamente
                    current_shape = generate_random_shape()
                    square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
                    square_timer = 0  # Reiniciar el tiempo de la forma
                    static_angle = randint(0, 360)  # Generar un nuevo ángulo aleatorio para la siguiente iteración

        # Actualizar el temporizador del cuadrado
        square_timer += clock.get_time() / 1000
        if square_timer >= square_lifetime:
            # Cambiar la figura aleatoriamente cuando se agota el temporizador
            current_shape = generate_random_shape()
            square_position = [randint(0, screen_width - square_size), randint(0, screen_height - square_size)]
            square_timer = 0

        # Convertir la imagen procesada a formato Pygame
        frame_surface = pygame.surfarray.make_surface(np.rot90(image_rgb))

        # Dibujar elementos en pantalla
        screen.blit(frame_surface, (0, 0))  # Imagen de la cámara como fondo
        
        # Dibujar la forma estática en la pantalla con su ángulo fijo
        if current_shape == 'circle':
            rotated_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            pygame.draw.circle(rotated_surface, (255, 0, 0), (square_size // 2, square_size // 2), square_size // 2)
            rotated_surface = pygame.transform.rotate(rotated_surface, static_angle)
            screen.blit(rotated_surface, (square_position[0] - rotated_surface.get_width() // 2,
                                          square_position[1] - rotated_surface.get_height() // 2))
        elif current_shape == 'square':
            rotated_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            pygame.draw.rect(rotated_surface, (255, 0, 0), (0, 0, square_size, square_size))
            rotated_surface = pygame.transform.rotate(rotated_surface, static_angle)
            screen.blit(rotated_surface, (square_position[0] - rotated_surface.get_width() // 2,
                                          square_position[1] - rotated_surface.get_height() // 2))
        elif current_shape == 'triangle':
            points = [(square_size // 2, 0), (0, square_size), (square_size, square_size)]
            rotated_surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            pygame.draw.polygon(rotated_surface, (255, 0, 0), points)
            rotated_surface = pygame.transform.rotate(rotated_surface, static_angle)
            screen.blit(rotated_surface, (square_position[0] - rotated_surface.get_width() // 2,
                                          square_position[1] - rotated_surface.get_height() // 2))

        # Dibujar la bola (con la misma forma que la figura)
        if current_shape == 'circle':
            rotated_ball = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(rotated_ball, (0, 0, 255), (ball_radius, ball_radius), ball_radius)
            rotated_ball = pygame.transform.rotate(rotated_ball, angle)
            screen.blit(rotated_ball, (ball_x - rotated_ball.get_width() // 2, ball_y - rotated_ball.get_height() // 2))
        elif current_shape == 'square':
            rotated_ball = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
            pygame.draw.rect(rotated_ball, (0, 0, 255), (0, 0, ball_radius * 2, ball_radius * 2))
            rotated_ball = pygame.transform.rotate(rotated_ball, angle)
            screen.blit(rotated_ball, (ball_x - rotated_ball.get_width() // 2, ball_y - rotated_ball.get_height() // 2))
        elif current_shape == 'triangle':
            points_ball = [(ball_radius, 0), (0, ball_radius * 2), (ball_radius * 2, ball_radius * 2)]
            rotated_ball = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
            pygame.draw.polygon(rotated_ball, (0, 0, 255), points_ball)
            rotated_ball = pygame.transform.rotate(rotated_ball, angle)
            screen.blit(rotated_ball, (ball_x - rotated_ball.get_width() // 2, ball_y - rotated_ball.get_height() // 2))

        # Mostrar el puntaje
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))

        # Actualizar pantalla
        pygame.display.flip()

        # Limitar los FPS
        clock.tick(60)

    cap.release()
    pygame.quit()
