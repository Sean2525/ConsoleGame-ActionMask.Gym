import os
import sys
sys.path.append('./')

from env.Snake.action_mask_env import ActionMaskEnv
from stable_baselines import PPO2
from stable_baselines.common.vec_env import  DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from examples.MouseWalkingMaze.map1.custom_policy import CustomCnnLnLstmPolicy

model_folder = './models/Snake/action_mask/'
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

env = DummyVecEnv([lambda: ActionMaskEnv(10, 10)])

model_tag = ''
if len(sys.argv) > 1:
    model_tag = '_' + sys.argv[1]

model = PPO2.load(model_folder + "PPO2" + model_tag)

done = False
states = None
action_masks = []
obs = env.reset()

while not done:
    action, states = model.predict(obs, states, action_mask=action_masks)
    obs, _, done, infos = env.step(action)
    env.render()
    action_masks.clear()
    for info in infos:
        env_action_mask = info.get('action_mask')
        action_masks.append(env_action_mask) 