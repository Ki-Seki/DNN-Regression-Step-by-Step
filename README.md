# DNN Regression Step by Step

This repo is a beginner tutorial on making DNN step by step.

If we have a good command of the inside mechanism of Deep Neural Network (DNN), then we can be much better at Deep Learning frameworks with high-level abstraction, like TensorFlow, Pytorch, etc.

## Goal

* Use pure Python (with NumPy at most) to implement a DNN model that can fit any given mathematical function. 
* Complete this task within several version iterations.

## Task

Given the labeled dataset which is generated from $f:x \mapsto y$, 

such as $\mathbf{x}=[x_1, x_2, ..., x_m]$ and its corresponding $\mathbf{y}=[y_1, y_2, ..., y_m]$, 

the model should try to fit the function, that is $f': x \mapsto \hat{y}$, 

with a relatively lower Mean Squared Error (MSE), that is $\sum_{i=1}^{m}(\hat{y_i}-y_i)^2$.

## Version History

| Version | Network Architecture | Description            |
| ------- | -------------------- | ---------------------- |
| V1.0    | 2 Layers: 1 x 2 x 1  | Fit a linear function. |
