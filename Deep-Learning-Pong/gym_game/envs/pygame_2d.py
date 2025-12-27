# Libraries
import pygame
import numpy as np
import os


import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)




class Paddle:
    COLOR = WHITE
    VEL = 4

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    MAX_VEL = 5
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1




class PyGame2D:
    def __init__(self):

        pygame.init()
        self.WIDTH, self.HEIGHT = 700, 500
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong")

        self.FPS = 60

        PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
        BALL_RADIUS = 7

        self.SCORE_FONT = pygame.font.SysFont("comicsans", 50)
        self.WINNING_SCORE = 10

        self.run = True
        self.clock = pygame.time.Clock()

        self.left_paddle = Paddle(10, self.HEIGHT//2 - PADDLE_HEIGHT //
                            2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = Paddle(self.WIDTH - 10 - PADDLE_WIDTH, self.HEIGHT //
                            2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, BALL_RADIUS)


        self.left_score = 0
        self.right_score = 0
        self.reward = 0

    
    #
    def isWinnerExist(self):
        if self.left_score >= self.WINNING_SCORE:
            return True
        elif self.right_score >= self.WINNING_SCORE:
            return True
        else:
            return False

    def resetGame (self):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0


    def action(self, action):
        self.reward = 0

        middle_y = self.right_paddle.y + self.right_paddle.height / 2
        if ( (action == 0 and self.ball.y < middle_y ) or (action == 1 and self.ball.y > middle_y ) or (action == 2 and self.ball.y == middle_y)) :
            self.reward += 10
        else:
            self.reward -= 10

        self.handle_paddle_movement(action)
        self.ball.move()
        self.handle_collision()

    
        if self.ball.x < 0:
            self.right_score += 1
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.reward += 2000
        elif self.ball.x > self.WIDTH:
            self.left_score += 1
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.reward -= 4000

    
    def view(self):
        self.draw()
        self.clock.tick(self.FPS)

    def draw(self):
        self.win.fill(BLACK)

        left_score_text = self.SCORE_FONT.render(f"{self.left_score}", 1, WHITE)
        right_score_text = self.SCORE_FONT.render(f"{self.right_score}", 1, WHITE)
        self.win.blit(left_score_text, (self.WIDTH//4 - left_score_text.get_width()//2, 20))
        self.win.blit(right_score_text, (self.WIDTH * (3/4) -
                                    right_score_text.get_width()//2, 20))

        for paddle in [self.left_paddle,self.right_paddle]:
            paddle.draw(self.win)

        for i in range(10, self.HEIGHT, self.HEIGHT//20):
            if i % 2 == 1:
                continue
            pygame.draw.rect(self.win, WHITE, (self.WIDTH//2 - 5, i, 10, self.HEIGHT//20))

        self.ball.draw(self.win)
        pygame.display.update()


    def handle_collision(self):
        if self.ball.y + self.ball.radius >= self.HEIGHT:
            self.ball.y_vel *= -1
        elif self.ball.y - self.ball.radius <= 0:
            self.ball.y_vel *= -1

        if self.ball.x_vel < 0:
            if self.ball.y >= self.left_paddle.y and self.ball.y <= self.left_paddle.y + self.left_paddle.height:
                if self.ball.x - self.ball.radius <= self.left_paddle.x + self.left_paddle.width:
                    self.ball.x_vel *= -1

                    middle_y = self.left_paddle.y + self.left_paddle.height / 2
                    difference_in_y = middle_y - self.ball.y
                    reduction_factor = (self.left_paddle.height / 2) / self.ball.MAX_VEL
                    y_vel = difference_in_y / reduction_factor
                    self.ball.y_vel = -1 * y_vel
                    self.reward -= 200

        else:
            if self.ball.y >= self.right_paddle.y and self.ball.y <= self.right_paddle.y + self.right_paddle.height:
                if self.ball.x + self.ball.radius >= self.right_paddle.x:
                    self.ball.x_vel *= -1

                    middle_y = self.right_paddle.y + self.right_paddle.height / 2
                    difference_in_y = middle_y - self.ball.y
                    reduction_factor = (self.right_paddle.height / 2) / self.ball.MAX_VEL
                    y_vel = difference_in_y / reduction_factor
                    self.ball.y_vel = -1 * y_vel
                    self.reward += 100


    def handle_paddle_movement(self,action):
        middle_y = self.left_paddle.y + self.left_paddle.height / 2
        if (self.ball.y < middle_y and self.left_paddle.y - self.left_paddle.VEL >= 0 ):
            self.left_paddle.move(up=True)
        elif (self.ball.y > middle_y and self.left_paddle.y + self.left_paddle.VEL + self.left_paddle.height <= self.HEIGHT ):
            self.left_paddle.move(up=False)

        if action == 0 :
            self.right_paddle.move(up=True)
        if action == 1 :
            self.right_paddle.move(up=False)

    def evaluate(self):
        return self.reward

    def observe(self):
        obs = np.zeros(shape=(6,),dtype=np.int32)
        obs[0] = self.right_paddle.y + self.right_paddle.height / 2
        obs[1] = self.ball.x
        obs[2] = self.ball.y
        obs[3] = self.ball.x_vel
        obs[4] = self.ball.y_vel
        obs[5] = self.left_paddle.y + self.left_paddle.height / 2
        return obs



