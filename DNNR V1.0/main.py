'''
 # @ Author: Ki Seki
 # @ Create Time: 2023-01-16 20:19:45
 # @ Modified by: Ki Seki
 # @ Modified time: 2023-01-16 22:18:55
 # @ Description:
 '''

import random

import task_definition as td


# ─── Hyper-parameter Setting ──────────────────────────────────────────────────

random.seed(2)

# Dataset-related

size = 999  # Number of the samples in the dataset
lower = -100  # Lower bound of X of the dataset
upper = 100
split = [0.6, 0.3, 0.1]  # Train/Validate/Test splition ratio

# Training-related

iterations = 10 # Number of iterations

# ─── Dataset Preparation ──────────────────────────────────────────────────────

X = [random.randrange(lower, upper) for _ in range(size) ]
Y = [td.f(X[i]) for i in range(size) ]

s1, s2, s3 = round(size * split[0]), round(size * split[1]), round(size * split[2])
train = [X[:s1], Y[:s1]]
validate = [X[s1:s1+s2], Y[s1:s1+s2]]
test = [X[s1+s2:s1+s2+s3], Y[s1+s2:s1+s2+s3]]

# ─── Parameter Initialization ─────────────────────────────────────────────────

w1_1 = random.random()
w1_2 = random.random()  # 2nd weight of the 1st layer
b1 = random.random()    # Bias of the 1st layer

w2_1 = random.random()  # 1st weight of 2nd layer
w2_2 = random.random()
b2 = random.random()    # Bias of the 2nd layer

# ─── Training ─────────────────────────────────────────────────────────────────

for epoch in range(iterations):
    # ─── Forward Propagation ──────────────────────────────────────────────

    pass

    # ─── Backward Propagation ─────────────────────────────────────────────

    pass

    # ─── Evaluation ───────────────────────────────────────────────────────

    pass

# ─── Testing ──────────────────────────────────────────────────────────────────

