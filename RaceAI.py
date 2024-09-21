import pygame
from random import randint
import math
import time

# COLORS
bg = (0, 0, 0)
track = (255, 255, 255)

# WINDOW SETTINGS
pygame.init()
(width, height) = (800, 600)
GD = pygame.display.set_mode((width, height))
pygame.display.set_caption('Race AI')
GD.fill(bg)
clock = pygame.time.Clock()
image = pygame.transform.scale(pygame.image.load(r"C:\Users\jaimy\Pictures\Roblox\RobloxScreenShot20240823_175739378.png"), (40, 22))

# CAR SETTINGS
KEYS = []
START_POS = [400, 100]
maxSpeed = 5
acceleration = 0.10
deacceleration = 0.05
drag = 0.05
turnSpeed = 3

# AI SETTINGS
currentGen = []

# KEYS
# w - 1, s - 0, a - 3, d - 2

def spawnGen():
    currentGen.append(Car(image, START_POS[0], START_POS[1], 0, [], 0))

class Car():
    def __init__(self, image, x, y, speed, brain, fitness):
        self.image = image
        self.x = x
        self.y = y
        self.speed = speed
        self.brain = brain
        self.fitness = fitness
        self.angle = 0  
        self.rect = self.image.get_rect(center=(self.x, self.y))  

    def move(self):
        if 0 in KEYS:
            self.speed += deacceleration
        if 1 in KEYS:
            self.speed -= acceleration
   
        if self.speed != 0:
            if 3 in KEYS:  
                if self.speed > 0:
                    self.angle += turnSpeed  
                else:
                    self.angle -= turnSpeed 
           
            if 2 in KEYS:  
                if self.speed > 0:
                    self.angle -= turnSpeed 
                else:
                    self.angle += turnSpeed  

        #CALC ANGLE
        radians = math.radians(self.angle)
        self.x += self.speed * math.cos(radians)
        self.y += self.speed * math.sin(radians)

        if self.speed > 0:
            if self.speed > maxSpeed:
                self.speed = maxSpeed
            if (self.speed <= drag):
                self.speed = 0
            else:
                self.speed -= drag

        elif self.speed < 0:
            if self.speed < -maxSpeed:
                self.speed = -maxSpeed
            self.speed += drag

        # UPDATE LOCATION
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def draw(self):
        # ROTATE SPRITE
        rotated_image = pygame.transform.rotate(self.image, -self.angle)  # Draai tegen de klok in
        rotated_rect = rotated_image.get_rect(center=self.rect.center)  # Zorg dat het draait om het midden
        GD.blit(rotated_image, rotated_rect.topleft)

spawnGen()

running = True
while running:
    start = time.time()
    GD.fill(bg)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                KEYS.append(1)
            if event.key == pygame.K_s:
                KEYS.append(0)
            if event.key == pygame.K_a:
                KEYS.append(3)
            if event.key == pygame.K_d:
                KEYS.append(2)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                KEYS.remove(1)
            if event.key == pygame.K_s:
                KEYS.remove(0)
            if event.key == pygame.K_a:
                KEYS.remove(3)
            if event.key == pygame.K_d:
                KEYS.remove(2)

    for i in range(len(currentGen)):
        currentGen[i].draw()
        currentGen[i].move()

    pygame.display.update()
    clock.tick(60)
