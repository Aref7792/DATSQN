import os, random

import glfw
import numpy as np
import torch
from sympy.physics.control.control_plots import matplotlib
#from pyglet.canvas.cocoa import CocoaScreen
from torch import nn
import itertools
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames
import gymnasium as gym
import time
import pygame
from pygame import display
import matplotlib.pyplot as plt
import os

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()
import numpy as np
import torch as th
import torch.nn as nn
from snntorch import spikegen
from torch.utils.tensorboard import SummaryWriter
LOG_DIR='./logs/DATQN_test'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#please enter the game id that you have entered in DSQN_trian for training
env_id= 'Breakout'

#dtermine seed
seed=0

#determine number of test episodes
TC = 10

#NUmber of lives in each game
if env_id == 'Breakout':
    DTER = 5
elif env_id == 'BeamRider':
    DTER = 3
elif env_id == 'CrazyClimber':
    DTER = 5
elif env_id == 'Gopher':
    DTER = 3
elif env_id == 'SpaceInvaders':
    DTER = 3
elif env_id == 'Jamesbond':
    DTER = 6
else:
    DTER =1

class PseudoSpikeRect(th.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input_,x):
        ctx.save_for_backward(input_,)

        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                2
                / 2
                / (1 + (th.pi / 2 * 2 * input_).pow_(2))
                * grad_input
        )
        return grad, None

def atan():
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    def inner(x,x_):
        return PseudoSpikeRect.apply(x,x_)
    return inner

# define a neuron
class IF(nn.Module):
    def __init__(self, beta, vth):
        super(IF, self).__init__()
        # neuron settings
        self.vth = vth
        self.beta = th.tensor(beta)
        # neuron dynamics
        self.pseudo_spike = atan()
        self._init_mem()
        #self.state_function = self._base_state_function
        self.state_function=self._base_sub

    def reset_mem(self,vmem):
        mem_shift = (vmem - self.vth)
        vmem_sign=1
        reset=self.pseudo_spike(mem_shift, vmem_sign).clone().detach()
        return reset

    def _init_mem(self):
        vmem = th.zeros(0)
        self.register_buffer("vmem", vmem, False)

    def mem_reset(self):
        self.vmem = th.zeros_like(self.vmem, device=device)
        return self.vmem

    def fire(self,vmem):
        mem_shift = (vmem - self.vth)
        vmem_sign = 1
        spk = self.pseudo_spike(mem_shift, vmem_sign)
        return spk

    def forward(self, input_, vmem=None):
        if not vmem == None:
            self.vmem = vmem
        if self.vmem.shape != input_.shape:
            self.vmem=th.zeros_like(input_).to(device=device)
        self.reset = self.reset_mem(self.vmem)
        self.vmem = self.state_function(input_)
        spk = self.fire(self.vmem)
        return spk, self.vmem

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0,1) * self.vmem + input_
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.vth


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
        n_input_channels = env.observation_space.shape[0]
        self.fc1=nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4)
        self.fc2 = IF(beta=beta, vth=self.threshold)
        self.fc3=nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2)
        self.fc4 = IF(beta=beta, vth=self.threshold)
        self.fc5=nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1)
        self.fc6 = IF(beta=beta, vth=self.threshold)
        self.fc7=nn.Flatten()
        self.fc8=nn.Linear(3136,final_layer)
        self.fc9 = IF(beta=beta, vth=self.threshold)
        self.fc10=nn.Linear(final_layer, self.num_actions)

    def forward(self, x):
        mem1=self.fc2.mem_reset()
        mem2 = self.fc4.mem_reset()
        mem3 = self.fc6.mem_reset()
        mem4 = self.fc9.mem_reset()
        spk_out_rec = []
        x=x/255
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

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions-1)
        return actions

    def load(self, epoch):
        print('load model')
        self.load_state_dict(th.load('training_models/dsqn_' + str(epoch) + '.pth'))

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
env= make_atari_env(env_id+'NoFrameskip-v4',n_envs=1, seed=seed)

summary_writer = SummaryWriter(LOG_DIR)
env=VecTransposeImage(env)
env = BatchedPytorchFrameStack(env, k=4)

net = Network(env, device, depths=(32, 64,64), final_layer=512,beta=.9, threshold=1,num_steps=20)
net = net.to(device)


for k in range(100):

    net.load(int((k+1) * 10000))
    obs = env.reset()

    beginning_episode = True
    reww=0
    dter=0
    Rew = np.zeros((10,1))

    tc =0
    for t in itertools.count():
        if isinstance(obs[0], PytorchLazyFrames):
            act_obs = np.stack([o.get_frames() for o in obs])
            action = net.act(act_obs, 0.0)
        else:
            action = net.act(obs, 0.0)
        if beginning_episode:
            action = [1]
            beginning_episode = False
        obs, rew, done, _= env.step(action)
        reww = reww + rew

        ##watch the game

        env.render()
        time.sleep(0.01)

        if done[0]:
            dter = dter + 1



        if dter==DTER:
            Rew[tc] = reww
            obs = env.reset()
            beginning_episode = True
            reww = 0
            dter = 0
            tc= tc+1

        if tc==TC:
            R = np.average(Rew)
            V = np.var(Rew) **.5
            Rew = np.zeros((10,1))
            summary_writer.add_scalar('AvgRew', R, global_step=(k+1) * 10000)
            summary_writer.add_scalar('VAR', V, global_step=(k + 1) * 10000)
            break