import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math
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

# Inicializar pygame y sonidos
pygame.init()

# Cargar música de fondo y sonido de éxito
pygame.mixer.music.load('background_music.wav')
pygame.mixer.music.play(-1, 0.0)  # Reproducir en bucle (-1) desde el principio

success_sound = pygame.mixer.Sound('success_sound.wav')

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
        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return annotated_image

# Configuración de Pygame para ventana de 1280x720
screen_width, screen_height = 1280, 720  # Tamaño fijo de la ventana
screen = pygame.display.set_mode((screen_width, screen_height))  # Ventana de tamaño fijo
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Variables del juego
score = 0
square_size = 120  # Tamaño de los cuadrados
max_squares = 3  # Número máximo de cuadrados simultáneos
squares = []  # Lista de cuadrados, cada cuadrado tendrá su posición, velocidad y tiempo de aparición
square_size_player = 120  # Tamaño del cuadrado controlado por el jugador
game_timer = 30  # 30 segundos para el juego
last_square_time = 0  # Tiempo del último cuadrado generado
angle = 0  # Ángulo entre el pulgar y el meñique
new_square_interval = 2  # Intervalo en segundos para generar nuevos cuadrados

# Gravedad
gravity = 0.5
bounce_factor = 0.8  # Factor de rebote

# Función para mostrar la pantalla de "Game Over"
def show_game_over_screen(final_score, high_score):
    game_over = True
    while game_over:
        screen.fill((255, 255, 255))  # Fondo blanco
        game_over_text = font.render("Game Over", True, (0, 0, 0))
        score_text = font.render(f"Puntaje final: {final_score}", True, (0, 0, 0))
        high_score_text = font.render(f"Mejor puntuación: {high_score}", True, (0, 0, 0))
        repeat_text = font.render("Presiona R para repetir", True, (0, 0, 0))
        quit_text = font.render("Presiona Q para salir", True, (0, 0, 0))

        screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, screen_height // 4))
        screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, screen_height // 3))
        screen.blit(high_score_text, (screen_width // 2 - high_score_text.get_width() // 2, screen_height // 2))
        screen.blit(repeat_text, (screen_width // 2 - repeat_text.get_width() // 2, screen_height // 1.5))
        screen.blit(quit_text, (screen_width // 2 - quit_text.get_width() // 2, screen_height // 1.8))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # Repetir el juego
                if event.key == pygame.K_q:
                    return False  # Salir del juego
    return False  # Si no presionó "R" ni "Q", devuelve False

# Función para mostrar el menú de pausa
def show_pause_menu():
    menu_running = True
    while menu_running:
        screen.fill((255, 255, 255))  # Fondo blanco
        pause_text = font.render("Pausa", True, (0, 0, 0))
        resume_text = font.render("Presiona 'R' para continuar", True, (0, 0, 0))
        quit_text = font.render("Presiona 'Q' para salir", True, (0, 0, 0))

        screen.blit(pause_text, (screen_width // 2 - pause_text.get_width() // 2, screen_height // 4))
        screen.blit(resume_text, (screen_width // 2 - resume_text.get_width() // 2, screen_height // 2))
        screen.blit(quit_text, (screen_width // 2 - quit_text.get_width() // 2, screen_height // 1.5))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # Reanudar el juego
                if event.key == pygame.K_q:
                    pygame.quit()

    return False  # Si no se presionó ninguna tecla, devuelve False

# Función para leer la mejor puntuación desde un archivo
def read_high_score():
    try:
        with open('high_score.txt', 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # Si el archivo no existe, retorna 0 como la puntuación inicial

# Función para guardar la mejor puntuación en un archivo
def save_high_score(high_score):
    with open('high_score.txt', 'w') as f:
        f.write(str(high_score))

# Definir las opciones de HandLandmarker antes de usarlo
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result
)

# Función para detectar la colisión entre dos cuadrados
def check_collision(square1, square2):
    # Revisar si hay colisión en los bordes
    return (square1['position'][0] < square2['position'][0] + square_size and
            square1['position'][0] + square_size > square2['position'][0] and
            square1['position'][1] < square2['position'][1] + square_size and
            square1['position'][1] + square_size > square2['position'][1])

# Función para calcular el ángulo entre dos puntos
def calculate_angle(p1, p2):
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    return math.atan2(delta_y, delta_x)  # Devuelve el ángulo en radianes

# Función principal del juego
def run_game():
    global score, squares, last_square_time
    game_running = True  # Variable que controla si el juego sigue corriendo

    # Leer la mejor puntuación desde el archivo
    high_score = read_high_score()

    # Mostrar el menú principal antes de comenzar
    if not show_main_menu(high_score):
        pygame.quit()
        exit()  # Si el jugador decide salir, cerramos el juego

    # Bucle principal para reiniciar el juego
    while game_running:
        squares = []  # Reiniciar la lista de cuadrados
        score = 0  # Reiniciar el puntaje
        start_time = time.time()  # Tiempo de inicio del juego
        last_square_time = start_time  # Tiempo de aparición del primer cuadrado

        # Crear el primer cuadrado
        squares.append({
            'position': [randint(0, screen_width - square_size), 0],  # Posición en la parte superior
            'velocity': [0, 0],  # Velocidad inicial (0, 0)
            'start_time': time.time()
        })

        with HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)
            running = True
            while cap.isOpened() and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:  # Tecla ESC para pausar
                            if not show_pause_menu():
                                running = False

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Obtener las dimensiones de la imagen de la cámara
                camera_height, camera_width, _ = image.shape

                # Redimensionar la imagen de la cámara para que encaje en la ventana de 1280x720
                image_resized = cv2.resize(image, (screen_width, screen_height))

                # Procesar la imagen de la cámara
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_resized)
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, frame_timestamp_ms)

                # Convertir la imagen de OpenCV (BGR) a Pygame (RGB)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

                # Inicializarplayer_square_x y player_square_y por defecto si no hay detección
                player_square_x, player_square_y = screen_width // 2, screen_height // 2  # Valores predeterminados (centro de la pantalla)

                angle = 0  # Ángulo predeterminado

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

                    # Calcular el ángulo entre el pulgar y el meñique
                    angle = math.degrees(calculate_angle(thumb_tip, pinky_tip))

                    # Dibujar las landmarks sobre la imagen
                    image_rgb = draw_landmarks_on_image(image_rgb, detection_result)

                    # Mover la bola con la mano (posicionar la bola en el centro de la mano)
                    player_square_x = screen_width - screen_x  # Invertir la coordenada x para compensar el espejo
                    player_square_y = screen_y

                    # Verificar si la bola colide con algún cuadrado
                    for square in squares[:]:
                        if (square['position'][0] <=player_square_x <= square['position'][0] + square_size and
                            square['position'][1] <= player_square_y <= square['position'][1] + square_size):
                            time_elapsed = time.time() - square['start_time']  # Tiempo que tardó en tocar el cuadrado
                            score += max(0, 10 - int(time_elapsed * 2))  # Restar 2 puntos por cada segundo de demora
                            squares.remove(square)  # Eliminar el cuadrado tocado
                            # Agregar un nuevo cuadrado
                            squares.append({
                                'position': [randint(0, screen_width - square_size), 0],  # Aparece en la parte superior
                                'velocity': [0, 0],  # Velocidad inicial (0, 0)
                                'start_time': time.time()
                            })

                            # Reproducir sonido de éxito
                            success_sound.play()

                # Actualizar el temporizador de los cuadrados y eliminar los que hayan pasado 5s
                squares = [square for square in squares if time.time() - square['start_time'] < 5]

                # Generar un nuevo cuadrado cada 2 segundos si hay menos de 3
                if len(squares) < max_squares and time.time() - last_square_time >= new_square_interval:
                    squares.append({
                        'position': [randint(0, screen_width - square_size), 0],  # Aparece en la parte superior
                        'velocity': [0, 0],  # Velocidad inicial
                        'start_time': time.time()
                    })
                    last_square_time = time.time()  # Actualizar el tiempo de aparición del último cuadrado

                # Actualizar la física de los cuadrados (gravedad y rebote)
                for square in squares:
                    # Aplicar gravedad
                    square['velocity'][1] += gravity

                    # Actualizar posición del cuadrado
                    square['position'][1] += square['velocity'][1]

                    # Colisiones con el borde inferior
                    if square['position'][1] + square_size > screen_height:
                        square['position'][1] = screen_height - square_size  # Ajustar posición al borde inferior
                        square['velocity'][1] *= -bounce_factor  # Rebote

                    # Colisiones con los bordes izquierdo y derecho
                    if square['position'][0] < 0:
                        square['position'][0] = 0
                        square['velocity'][0] *= -bounce_factor  # Rebote
                    if square['position'][0] + square_size > screen_width:
                        square['position'][0] = screen_width - square_size
                        square['velocity'][0] *= -bounce_factor  # Rebote

                    # Verificar las colisiones entre los cuadrados
                    for other_square in squares:
                        if square != other_square and check_collision(square, other_square):
                            # Colisión detectada, intercambiar velocidades (rebote)
                            square['velocity'][0], other_square['velocity'][0] = other_square['velocity'][0], square['velocity'][0]
                            square['velocity'][1], other_square['velocity'][1] = other_square['velocity'][1], square['velocity'][1]

                # Retraso de 2 segundos antes de empezar a contar el tiempo
                if time.time() - start_time > 2:
                    # Actualizar el temporizador del juego
                    elapsed_game_time = time.time() - start_time - 2  # Restar los 2 segundos de retraso
                    if elapsed_game_time >= game_timer:
                        running = False  # Finalizar el juego después de 30 segundos

                # Convertir la imagen procesada a formato Pygame
                frame_surface = pygame.surfarray.make_surface(np.rot90(image_rgb))

                # Dibujar elementos en pantalla
                screen.blit(frame_surface, (0, 0))  # Imagen de la cámara como fondo

                # Dibujar el cuadrado controlado por la mano
                square_image = pygame.Surface((square_size_player, square_size_player), pygame.SRCALPHA)  # Crear una superficie para el cuadrado
                square_image.fill((0, 255, 0))  # Rellenar de verde
                rotated_square = pygame.transform.rotate(square_image, angle)  # Rotar el cuadrado según el ángulo
                square_rect = rotated_square.get_rect(center=(player_square_x, player_square_y))  # Obtener el rectángulo del cuadrado rotado
                screen.blit(rotated_square, square_rect)  # Dibujar el cuadrado rotado en pantalla
                
                # Dibujar los cuadrados
                for square in squares:
                    pygame.draw.rect(screen, (255, 0, 0), (*square['position'], square_size, square_size))

                # Mostrar el puntaje en pantalla
                score_text = font.render(f"Puntaje: {score}", True, (0, 0, 0))
                screen.blit(score_text, (10, 10))

                # Mostrar el temporizador en pantalla
                timer_text = font.render(f"Tiempo: {max(0, game_timer - int(elapsed_game_time))}", True, (0, 0, 0))
                screen.blit(timer_text, (screen_width - timer_text.get_width() - 10, 10))

                pygame.display.flip()  # Actualizar la pantalla

                clock.tick(60)  # Limitar el frame rate a 60 FPS

            cap.release()  # Liberar la cámara

        # Al final del juego, mostrar la pantalla de Game Over y guardar el high score
        if score > high_score:
            high_score = score
            save_high_score(high_score)

        # Mostrar Game Over
        if not show_game_over_screen(score, high_score):
            game_running = False

    pygame.quit()

# Función para mostrar el menú principal
def show_main_menu(high_score):
    menu_running = True
    while menu_running:
        screen.fill((255, 255, 255))  # Fondo blanco
        title_text = font.render("Catch the Box", True, (0, 0, 0))
        high_score_text = font.render(f"Mejor puntuación: {high_score}", True, (0, 0, 0))
        start_text = font.render("Presiona Enter para comenzar", True, (0, 0, 0))
        quit_text = font.render("Presiona Q para salir", True, (0, 0, 0))

        screen.blit(title_text, (screen_width // 2 - title_text.get_width() // 2, screen_height // 4))
        screen.blit(high_score_text, (screen_width // 2 - high_score_text.get_width() // 2, screen_height // 3))
        screen.blit(start_text, (screen_width // 2 - start_text.get_width() // 2, screen_height // 2))
        screen.blit(quit_text, (screen_width // 2 - quit_text.get_width() // 2, screen_height // 1.5))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Tecla Enter para iniciar el juego
                    return True
                if event.key == pygame.K_q:  # Tecla Q para salir
                    return False

    return False  # Si no se presionó ninguna tecla, devuelve False

# Iniciar el juego
run_game()
