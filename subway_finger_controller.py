# file: hand_runner_with_camera.py

import cv2
import mediapipe as mp
import pygame
import sys
import random
import numpy as np

# Game window setup
pygame.init()
game_width, game_height = 640, 480
total_width = game_width * 2  # Split screen
screen = pygame.display.set_mode((total_width, game_height))
pygame.display.set_caption("Finger Game + Webcam View")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Player
player_size = 50
player_y = game_height - player_size - 10
player_x = game_width // 2
player_color = (0, 255, 0)

# Obstacles
obstacle_size = 50
obstacle_color = (255, 0, 0)
obstacle_speed = 5
obstacles = []

# Score
score = 0

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Main loop
running = True
while running:
    screen.fill((0, 0, 0))

    # Handle quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read and process camera
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Detect finger and move player
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index = hand_landmarks.landmark[8]
            fx = int(index.x * game_width)
            player_x = max(0, min(game_width - player_size, fx))

    # Draw player
    player_rect = pygame.Rect(player_x, player_y, player_size, player_size)
    pygame.draw.rect(screen, player_color, player_rect)

    # Spawn and draw obstacles
    if random.randint(1, 30) == 1:
        ox = random.randint(0, game_width - obstacle_size)
        obstacles.append(pygame.Rect(ox, 0, obstacle_size, obstacle_size))

    for obs in obstacles[:]:
        obs.y += obstacle_speed
        pygame.draw.rect(screen, obstacle_color, obs)
        if obs.colliderect(player_rect):
            running = False
        if obs.y > game_height:
            obstacles.remove(obs)
            score += 1

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    # Show camera feed (right side)
    cam_resized = cv2.resize(frame, (game_width, game_height))
    cam_surface = pygame.surfarray.make_surface(np.rot90(cam_resized))
    screen.blit(cam_surface, (game_width, 0))  # Right side

    pygame.display.flip()
    clock.tick(60)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
