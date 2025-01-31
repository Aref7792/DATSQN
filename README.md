# DATSQN
Improving Performance of Spike-based Deep Q-Learning using Ternary Neurons

This code evaluates the performance of deep spiking Q networks utilizing binary spiking neurons, ternary spiking neurons, and asymmetric ternary spiking neurons in playing Atari games in the Gym environment. 

How to Run: 
\newline Install all the required packages, 
To train each of the three RL agents run:  DATSQN-train.py, DTSQN-train.py, and DSQN-train.py 
To test a trained model: put the model in the training_models directory. Open the related test environment: DATSQN-test.py or DTSQN-test.py or DSQN-test.py. Enter the digit part of the model name (digits after "dqn") to "net.load()" function (line 255 in DATSQN and DTSQN, line 170 in DSQN). Run the code.   
