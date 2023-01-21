'''
 # @ Author: Ki Seki
 # @ Create Time: 2023-01-16 20:19:45
 # @ Modified by: Ki Seki
 # @ Modified time: 2023-01-21 15:04:36
 # @ Description:
 '''

import random

import task_definition as td
import util


# ─── Hyper-parameter Setting ──────────────────────────────────────────────────

random.seed()

# Dataset-related

size = 999  # Number of the samples in the dataset
lower = -100  # Lower bound of X of the dataset
upper = 100
split = [0.6, 0.3, 0.1]  # Train/Validate/Test splition ratio

# Training-related

iterations = 10000  # Number of iterations
alpha = 0.2  # Learning rate

# ─── Dataset Preparation ──────────────────────────────────────────────────────

X = [random.randrange(lower, upper) for _ in range(size) ]
Y = [td.f(X[i]) for i in range(size) ]

s1, s2, s3 = round(size * split[0]), round(size * split[1]), round(size * split[2])
train = [X[:s1], Y[:s1]]
validate = [X[s1:s1+s2], Y[s1:s1+s2]]
test = [X[s1+s2:s1+s2+s3], Y[s1+s2:s1+s2+s3]]

# ─── Parameter Initialization ─────────────────────────────────────────────────

w1_1 = random.random()  # 1st weight of the 1st layer
w1_2 = random.random()
b1_1 = random.random()
b1_2 = random.random()  # 2nd bias of the 1st layer

w2_1 = random.random()  # 1st weight of 2nd layer
w2_2 = random.random()
b2 = random.random()    # Bias of the 2nd layer

# ─── Training ─────────────────────────────────────────────────────────────────

for epoch in range(iterations):

    Dw1_1, Dw1_2, Db1_1, Db1_2, Dw2_1, Dw2_2, Db2 = [0] * 7  # Initialization

    for x, y in zip(train[0], train[1]):  # For each sample in training set

        # ─── Forward Propagation ──────────────────────────────────────
        
        a0 = x
        a1_1 = util.sigmoid(w1_1 * x + b1_1)
        a1_2 = util.sigmoid(w1_2 * x + b1_2)
        # a1_1 = 1 / (1 + math.e ** -(w1_1 * x + b1_1))
        # a1_2 = 1 / (1 + math.e ** -(w1_2 * x + b1_2))
        a2 = w2_1 * a1_1 + w2_2 * a1_2 + b2
        y_hat = a2

        # ─── Derivative Calculation ───────────────────────────────────

        dy_hat = (y_hat - y)
        dw2_1 = a1_1 * dy_hat
        dw2_2 = a1_2 * dy_hat
        db2 = dy_hat

        da1_1 = w2_1 * dy_hat
        da1_2 = w2_2 * dy_hat
        dz1_1 = a1_1 * (1 - a1_1) * da1_1
        dz1_2 = a1_2 * (1 - a1_2) * da1_2

        dw1_1 = a0 * dz1_1
        dw1_2 = a0 * dz1_2
        db1_1 = dz1_1
        db1_2 = dz1_2

        # ─── Derivative Accumulation ──────────────────────────────────

        Dw1_1 += dw1_1
        Dw1_2 += dw1_2
        Db1_1 += db1_1
        Db1_2 += db1_2
        Dw2_1 += dw2_1
        Dw2_2 += dw2_2
        Db2 += db2
        
    # ─── Backward Propagation ─────────────────────────────────────────────

    # Divided by the size of training set

    Dw1_1 /= s1
    Dw1_2 /= s1
    Db1_1 /= s1
    Db1_2 /= s1
    Dw2_1 /= s1
    Dw2_2 /= s1
    Db2 /= s1

    w2_1 = w2_1 - alpha * Dw2_1
    w2_2 = w2_2 - alpha * Dw2_2
    b2 = b2 - alpha * Db2
    w1_1 = w1_1 - alpha * Dw1_1
    w1_2 = w1_2 - alpha * Dw1_2
    b1_1 = b1_1 - alpha * Db1_1
    b1_2 = b1_2 - alpha * Db1_2

    # ─── Evaluation On Validation Set ─────────────────────────────────────

    loss = 0
    for x, y in zip(validate[0], validate[1]):
        a0 = x
        a1_1 = util.sigmoid(w1_1 * x + b1_1)
        a1_2 = util.sigmoid(w1_2 * x + b1_2)
        a2 = w2_1 * a1_1 + w2_2 * a1_2 + b2
        y_hat = a2
        loss += 1/2 * (y_hat - y) ** 2
    loss /= s2  # == 1/2 MSE

    print(f'{epoch + 1 = }, {loss = }')

    # ─── Save Parameters Bring Best Performance ───────────────────────────

    pass

# ─── Testing ──────────────────────────────────────────────────────────────────

pass
