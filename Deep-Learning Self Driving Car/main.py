import gym
import gym_game
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    env.reset()

    #check_env(env)

    # Test
    
    """
    for ep in range(5):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            # pass observation to model to get predicted action
            action = env.action_space.sample()
            # pass action to env and get info back
            obs, rewards, done, info = env.step(action)

            # show the environment on the screen
            score += rewards
            env.render()

        print(f"Run Number :{ep} Score: {score}")
    """
    
    # Train
    
    if (not True):
        #model = PPO('MlpPolicy', env, verbose=1 ,tensorboard_log="logs")

        TIMESTEPS = 20000

        if not os.path.exists("models/PPO"):
            os.makedirs("models/PPO")

        for j in range(1,11):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False , tb_log_name="PPO")
            model.save(f"models/PPO/{TIMESTEPS*j}")


    # Test with model

    if (True):
        model = PPO.load("models/PPO/100000" , env = env)
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