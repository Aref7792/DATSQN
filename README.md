
# 🧠 DATSQN  
## Improving Performance of Spike-based Deep Q-Learning using Ternary Neurons

This repository evaluates the performance of **SNNs** for **Q-learning** using:

- **Binary spiking neurons** (DSQN)  
- **Ternary spiking neurons** (DTSQN)  
- **Asymmetric ternary spiking neurons** (DATSQN)  

These models are trained and tested on **Atari games** using the **Gymnasium** framework.

---

## 📦 Installation

Install all required Python packages:

```bash
pip install gymnasium
pip install stable-baselines3
pip install ale_py
pip install snntorch
pip install msgpack-numpy
pip install tensorboard
```

---

## 🚀 How to Run

### 🧠 Training

To train each of the three reinforcement learning agents, run the corresponding script:

```bash
python DATSQN_train.py    # Asymmetric ternary spiking neurons
python DTSQN_train.py     # Ternary spiking neurons
python DSQN_train.py      # Binary spiking neurons
```

### 📊 Monitor Training

To visualize the training process using TensorBoard, run:

```bash
tensorboard --logdir logs/
```

---

## 🧪 Testing a Trained Model

Once training is complete, run the corresponding test script:

```bash
python DATSQN_test.py
# or
python DTSQN_test.py
# or
python DSQN_test.py
```

### 👀 Watch the Agent Play

To render the game environment and observe the agent in action:

1. Open the appropriate test script.
2. Uncomment the `env.render()` line in the main test loop:
   - Line **248** in `DATSQN_test.py`
   - Line **238** in `DTSQN_test.py`
   - Line **235** in `DSQN_test.py`

This enables real-time visualization of the agent during evaluation.

### ⚠️ Important

Ensure the **environment ID** (e.g., `'Breakout'`) is **consistent** between training and testing scripts to avoid errors or unexpected behavior.

---

## 📈 Visualize Testing Results

You can also view evaluation metrics through TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## 📁 Project Structure

```
├── DATSQN_train.py        # Train using asymmetric ternary neurons
├── DATSQN_test.py         # Test DATSQN model
├── DTSQN_train.py         # Train using ternary neurons
├── DTSQN_test.py          # Test DTSQN model
├── DSQN_train.py          # Train using binary neurons
├── DSQN_test.py           # Test DSQN model
├── pytorch_wrappers.py    # Utility modules and helper functions
└── logs/                  # TensorBoard logs
```


