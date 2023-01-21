'''
 # @ Author: Ki Seki
 # @ Create Time: 2023-01-16 20:19:45
 # @ Modified by: Ki Seki
 # @ Modified time: 2023-01-21 15:04:36
 # @ Description:
 '''

import random

import matplotlib.pyplot as plt
from tqdm import tqdm

import task_definition as td
import util


# ─── Hyper-parameter Setting ──────────────────────────────────────────────────

random.seed(1)

# Dataset-related

size = 999  # Number of the samples in the dataset
lower = -100  # Lower bound of X of the dataset
upper = 100
split = [0.6, 0.3, 0.1]  # Train/Validate/Test splition ratio

# Training-related

iterations = 100  # Number of iterations
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

# ─── Values Need To Be Saved ──────────────────────────────────────────────────

best_w1_1, best_w1_2, best_b1_1, best_b1_2, best_w2_1, best_w2_2, best_b2 = [None] * 7
best_epoch = 0
loss_history = [0] * iterations

# ─── Training ─────────────────────────────────────────────────────────────────

pbar = tqdm(range(iterations))  # Progress bar
for epoch in pbar:

    Dw1_1, Dw1_2, Db1_1, Db1_2, Dw2_1, Dw2_2, Db2 = [0] * 7  # Initialization of accumulated variables

    for x, y in zip(train[0], train[1]):  # For each sample in training set

        # ─── Forward Propagation ──────────────────────────────────────
        
        a0 = x
        a1_1 = util.sigmoid(w1_1 * x + b1_1)
        a1_2 = util.sigmoid(w1_2 * x + b1_2)
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

    # Update parameters

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
    loss_history[epoch] = loss
    del_loss = loss - (0 if epoch == 0 else loss_history[epoch-1])

    # ─── Output and Drawing ───────────────────────────────────────────────

    pbar.set_description(f'Epoch = {epoch:5d}, loss = {loss:.3e}, Δloss = {del_loss:.3e}')  # Progress bar
    plt.scatter(epoch+1, loss, color='blue')
    plt.pause(0.001)  # Make the real-time plotting possible

    # ─── Save Parameters Bring Best Performance ───────────────────────────

    if loss_history[epoch] < loss_history[best_epoch]:
        best_w1_1 = w1_1
        best_w1_2 = w1_2
        best_b1_1 = b1_1
        best_b1_2 = b1_2
        best_w2_1 = w2_1
        best_w2_2 = w2_2
        best_b2 = b2
        best_epoch = epoch
print(f'The minimum loss is at epoch {best_epoch}, its loss is {loss_history[best_epoch]}')  # TODO
plt.show()


# ─── Testing ──────────────────────────────────────────────────────────────────

loss = 0
for x, y in zip(test[0], test[1]):
    a0 = x
    a1_1 = util.sigmoid(w1_1 * x + b1_1)
    a1_2 = util.sigmoid(w1_2 * x + b1_2)
    a2 = w2_1 * a1_1 + w2_2 * a1_2 + b2
    y_hat = a2
    loss += 1/2 * (y_hat - y) ** 2
loss /= s2  # == 1/2 MSE

print(f"Loss on the testing set: {loss}")