import os.path
import os
from sympy.physics.vector.printing import params
#from sympy.plotting.pygletplot.util import scale_value
from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import numpy as np
import random
import ale_py
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.autograd import Variable
import snntorch as snn
from snntorch import spikegen
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

#Selecting GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DRL algorithm parameters
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(4e5)
MIN_REPLAY_SIZE=10000
EPSILON_START=1.0
EPSILON_END=0.1
NUM_ENV=4
EPSILON_DECAY=int(1e6)
TARGET_UPDATE_FREQ = 10000//NUM_ENV
LR= 5e-5

# Saving approach parameters
SAVE_PATH= './atari_model.pack'
SAVE_INTERVAL= 10000
LOG_DIR='./logs/atari_vanilla'
LOG_INTERVAL=1000


# Define DSQN architecture using snn.leaky
class Network(nn.Module):
    def __init__(self, env,device,depths,final_layer,beta, threshold,num_steps ):
        super().__init__()
        self.num_actions = env.action_space.n
        self.device=device
        self.depths=depths
        self.final_layer=final_layer
        self.beta=beta
        self.beta=beta
        self.threshold=threshold
        self.num_steps=num_steps

        "defining the size of the input for the network"
        n_input_channels = env.observation_space.shape[0]

        self.fc1=nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4)
        self.fc2 = snn.Leaky(beta=beta, threshold=self.threshold)
        self.fc3=nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2)
        self.fc4 = snn.Leaky(beta=beta, threshold=self.threshold)
        self.fc5=nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1)
        self.fc6 = snn.Leaky(beta=beta, threshold=self.threshold)
        self.fc7=nn.Flatten()
        self.fc8=nn.Linear(3136,final_layer)
        self.fc9 = snn.Leaky(beta=beta, threshold=self.threshold)
        self.fc10=nn.Linear(final_layer, self.num_actions)


    def forward(self, x):

        mem1=self.fc2.reset_mem()
        mem2 = self.fc4.reset_mem()
        mem3 = self.fc6.reset_mem()
        mem4 = self.fc9.reset_mem()
        spk_out_rec = []

        "Normalizing pixel values"
        x=x/255

        "Rate encoding of the input"
        x=spikegen.rate(x, self.num_steps).to(device=device)

        for step in range(self.num_steps):

            cur1 = self.fc1(x[step])
            spk1, mem1 = self.fc2(cur1, mem1)
            cur2 = self.fc3(spk1)
            spk2, mem2 = self.fc4(cur2, mem2)
            cur3 = self.fc5(spk2)
            spk3, mem3 = self.fc6(cur3, mem3)
            cur4=self.fc8(self.fc7(spk3))
            spk4, mem4 = self.fc9(cur4, mem4)
            cur5 = self.fc10(spk4)
            spk_out_rec.append(cur5)
        return th.stack(spk_out_rec, dim=0).sum(dim=0)

    "selecting the optimal action"
    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        "epsilon-greedy policy"
        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions-1)

        return actions

    def compute_loss(self, transitions, target_net):

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    "save function for saving models during training"
    def save(self, epoch):
        print('model saved')
        # th.save(self.eval_net,'training_models/etdqn_'+str(epoch)+'.pkl')
        th.save(self.state_dict(), 'training_models/dsqn_' + str(epoch) + '.pth')

    "load function for loading the saved model"
    def load(self, epoch):
        print('load model')
        # self.eval_net = th.load('training_models/etdqn_'+str(epoch)+'.pkl')
        self.load_state_dict(th.load('training_models/dsqn_' + str(epoch) + '.pth'))


# making the vectorized gym environment
make_env= make_atari_env('BreakoutNoFrameskip-v4',n_envs=4, seed=0)
env= VecFrameStack(make_env, n_stack=4)
env=VecTransposeImage(env)

# defining reply buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)

# Defining the buffer to store episode information for processing
epinfos_buffer = deque([], maxlen=100)

episode_count = 0
summary_writer = SummaryWriter(LOG_DIR)

online_net = Network(env,device=device, depths=(32, 64,64), final_layer=512,beta=.9, threshold=1,num_steps=20)
target_net = Network(env,device=device, depths=(32, 64,64), final_layer=512,beta=.9, threshold=1,num_steps=20)

online_net= online_net.to(device)
target_net=target_net.to(device)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# Initialize replay buffer
obses = env.reset()

for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for _ in range(NUM_ENV)]
    new_obses, rews, dones,_ = env.step(actions)
    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
    obses = new_obses

# Defining the buffer to store reward information for processing
rews_buffer_=[[],[],[],[]]


# Main Training Loop
obses = env.reset()

for step in itertools.count():
    epsilon = np.interp(step * NUM_ENV, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()
    actions = online_net.act(obses, epsilon)
    new_obses, rews, dones, infos = env.step(actions)

    # reply buffer update
    i=0
    for obs, action, rew, done, new_obs, infos in zip(obses, actions, rews, dones, new_obses, infos):
        rews_buffer_[i].append(rew)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        if done:
            eprew = sum(rews_buffer_[i])
            eplen = len(rews_buffer_[i])
            epinfo = {"r": round(eprew, 6), "l": eplen}
            epinfos_buffer.append(epinfo)
            rews_buffer_[i]=[]
            episode_count+=1
    i+=1
    obses = new_obses

    transitions = random.sample(replay_buffer, BATCH_SIZE)
    loss = online_net.compute_loss(transitions, target_net)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Net
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % LOG_INTERVAL == 0:
        rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0
        print()
        print('Step:', step)
        print('Avg Rew:', rew_mean)
        print('Avg Ep len:', len_mean)
        print('Epizode:', episode_count)

        summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
        summary_writer.add_scalar('AvgLen', len_mean, global_step=step)
        summary_writer.add_scalar('Epizode', episode_count, global_step=step)

    #saving
    if step % SAVE_INTERVAL==0 and step !=0:
        print('Saving...')
        online_net.save(step)












