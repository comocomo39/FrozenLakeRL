# config.py
"""
Centralized configuration for RL agents and experiments.
"""
SEED = 48  # Random seed for reproducibility

# General training parameters
MAX_EPISODE_STEPS = 50
NUM_EPISODES = 300

# MCTS hyperparameters
NUM_SIMULATIONS = 1000
ROLLOUT_DEPTH = 36

# Q-Learning hyperparameters (defaults)
QL_ALPHA = 0.1
QL_GAMMA = 0.99
QL_EPSILON = 1.0
QL_EPSILON_MIN = 0.01
QL_EPSILON_DECAY = 0.999

# Penalties
GOAL_REWARD = 1
HOLE_PENALTY = -1
STEP_PENALTY = -0.01


# ------------------------------------------------
# Hyper-parametri per il Deep Q-Learning (DQN)
# ------------------------------------------------
DQN_LR = 5e-4
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 10000
DQN_TARGET_UPDATE = 200

# Numero minimo di transizioni raccolte prima di iniziare l’allenamento
DQN_MIN_REPLAY = 500   # da 1_000 a 100

# Fattore di sconto
DQN_GAMMA = 0.99

# Epsilon iniziale, finale e decay per epsilon-greedy
DQN_EPS_START = 0.8
DQN_EPS_END   = 0.01
DQN_EPS_DECAY = 5000  # da 10_000 a 1_000

# ------------------------------------------------
# MCTS hyper-parameters
# ------------------------------------------------
C_PARAM = 1.0

