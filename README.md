# DATSQN
Improving Performance of Spike-based Deep Q-Learning using Ternary Neurons

This code evaluates the performance of deep spiking Q networks utilizing binary spiking neurons, ternary spiking neurons, and asymmetric ternary spiking neurons in playing Atari games in the Gym environment. 

How to Run: 

Install all the required packages, 

To train each of the three RL agents run:  DATSQN-train.py, DTSQN-train.py, and DSQN-train.py 

To test a trained model:

Place the trained model in the training_models directory.

Open the corresponding test environment script: DATSQN-test.py, DTSQN-test.py, or DSQN-test.py.

Locate the net.load() function (line 255 in DATSQN-test.py and DTSQN-test.py, line 170 in DSQN-test.py).

Enter the numeric portion of the model name (the digits following "dqn") as the argument for net.load().

Run the script to execute the test.   
