import gym
import gym_game
import os

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    env.reset()

    #check_env(env)

    # Test
    
    """
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)

        env.render()
        if terminated:
            env.reset()
    """
    


    # Train
    
    if (not True):
        #model1 = A2C('MlpPolicy', env, verbose=1 , tensorboard_log="logs")
        model2 = DQN('MlpPolicy', env, verbose=1 , tensorboard_log="logs")

        models_list = [model2]
        #models_list = [model1,model2]
        #models_dir = ['A2C','DQN']
        models_dir = ['DQN']
        TIMESTEPS = 10000

        for dir in models_dir:
            if not os.path.exists("models/"+dir):
                os.makedirs("models/"+dir)

        for i in range(len(models_list)):
            for j in range(1,10):
                models_list[i].learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False , tb_log_name=models_dir[i])
                models_list[i].save(f"models/{models_dir[i]}/{TIMESTEPS*j}")


    # Test with model

    if (True):
        model = DQN.load("models/DQN/1000000" , env = env)
        for ep in range(5):
            obs = env.reset()
            done = False
            score = 0
            while not done:
                # pass observation to model to get predicted action
                action, _ = model.predict(obs ,deterministic=True)
                # pass action to env and get info back
                obs, rewards, done, info = env.step(action)

                # show the environment on the screen
                score += rewards
                env.render()

            print(f"Run Number :{ep} Score: {score}")

    env.close()