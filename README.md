# DATSQN
# Improving Performance of Spike-based Deep Q-Learning using Ternary Neurons

## This code evaluates the performance of deep spiking Q networks utilizing binary spiking neurons, ternary spiking neurons, and asymmetric ternary spiking neurons in playing Atari games in the Gym environment. 

## How to Run: 

Install all the required packages "pip install gymnasium; pip install stable_baselines3; pip install ale_py; pip install snntorch; pip install msgpack-numpy; pip install tensorboard", 

To train each of the three RL agents, run:  DATSQN_train.py, DTSQN_train.py, and DSQN_train.py
```
DATSQN_train.py
```
You can see the results by running TensorBoard in the terminal:

```
tensorboard --logdir /logs
```

### To test a trained model:

After finishing the training, run the corresponding test environment. For example, if you ran DATSQN_train.py, run DATSQN_test.py.
If you want to watch the game to observe the scores, please uncomment "env.render()" in the testing loop (lines 248 of DATSQN_test, 235 of DSQN_test, and 238 of DTSQN_test).  
Please ensure that the environment ID is consistent for test and train files. 

```
DATSQN-test.py
```

You can see the results by running TensorBoard in the terminal:

```
tensorboard --logdir /logs
```
