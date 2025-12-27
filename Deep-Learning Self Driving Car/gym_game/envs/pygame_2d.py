import pygame
import os
import math
import numpy as np

SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016


class PyGame2D():
    def __init__(self):
        super().__init__()

        self.SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.TRACK = pygame.image.load(os.path.join("Assets","track.png"))
        self.SCREEN.blit(self.TRACK, (0, 0))

        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 2
        self.alive = True
        self.radars = [[0,0] for i in range(5)]
        self.reward = 0
        self.CheckPoints = [
            # Bottom Right
            [(700, 750),(700, 900)],
            [(900, 750),(900, 900)],
            
            # Right
            [(950, 650), (1100, 650)],
            [(900, 400), (1100, 400)],

            # Top
            [(900, 350), (900, 150)],
            [(700, 300), (700, 150)],
            [(500, 300), (500, 150)],
            [(300, 350), (300, 150)],

            # Left
            [(150, 400), (400, 400)],
            [(150, 650), (400, 650)],
            # bottom left
            [(350, 750), (350, 900)],
            # Finish
            [(550, 750), (550, 900)],
        ]
        self.currentCheckPointIndex = 0

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        if self.SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or self.SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False
            self.reward -= 500
        
    def action(self, action):
        self.reward = 0
        self.direction = action

        
        self.drive()
        self.rotate()
        self.collision()
        for i,radar_angle in enumerate([-60, -30, 0, 30, 60]):
            self.radar(i,radar_angle)
        

    def rotate(self):
        if self.direction == 0:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == 1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)


        # Test Check points
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

        #pygame.draw.line(self.SCREEN,(0, 255, 0) , self.CheckPoints[i][0], self.CheckPoints[i][1], 1)
        if self.rect.clipline(self.CheckPoints[self.currentCheckPointIndex][0], self.CheckPoints[self.currentCheckPointIndex][1]): 
            self.currentCheckPointIndex += 1
            self.reward += 100
            #print(self.currentCheckPointIndex)
            if (self.currentCheckPointIndex >= len(self.CheckPoints)):
                self.reward += 5000
                self.alive = False


    def radar(self,Index,radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        
        #while not self.rgb_array[x][y] == [2, 105, 31] and length < 200:
        while not self.SCREEN.get_at((x,y)) == pygame.Color(2, 105, 31,255) and length < 200:
                length += 1
                x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            # Draw Radar
            #pygame.draw.line(self.SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
            #pygame.draw.circle(self.SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                                + math.pow(self.rect.center[1] - y, 2)))

        self.radars[Index] = [radar_angle, dist]


    def resetGame (self):
        self.rect = self.image.get_rect(center=(490, 820))
        self.direction = 2
        self.alive = True
        self.radars = [[0,0] for i in range(5)]
        self.currentCheckPointIndex = 0


    def isRoundFinished(self):
        return not self.alive


    def draw(self):
        self.SCREEN.blit(self.TRACK, (0, 0))
        self.SCREEN.blit(self.image,self.rect.topleft)

        if (self.alive and self.currentCheckPointIndex < 12):
            pygame.draw.line(self.SCREEN,(0, 255, 0) , self.CheckPoints[self.currentCheckPointIndex][0], self.CheckPoints[self.currentCheckPointIndex][1], 1)

        

        for radar_angle in [-60, -30, 0, 30, 60]:
            length = 0
            x = int(self.rect.center[0])
            y = int(self.rect.center[1])
            while not self.SCREEN.get_at((x,y)) == pygame.Color(2, 105, 31,255) and length < 200:
                    length += 1
                    x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
                    y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            pygame.draw.line(self.SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
            pygame.draw.circle(self.SCREEN, (0, 255, 0, 0), (x, y), 3)

    def view(self):
        self.draw()
        pygame.display.update()

    def evaluate(self):
        return self.reward

    def observe(self):
        obs = np.zeros(shape=(8,),dtype=np.int32)
        for i, radar in enumerate(self.radars):
            obs[i] = int(radar[1])

        obs[5] = int(self.rect.center[0])
        obs[6] = int(self.rect.center[1])
        obs[7] = self.currentCheckPointIndex
        return obs
