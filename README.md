# DNN Regression Step by Step

Make a framework-free DNN Regression model step by step.

If we have a good command of the inside mechanism of Deep Neural Network (DNN), then we can be much better at machine learning frameworks with high-level abstraction, like TensorFlow, PyTorch, etc.

## Task

Given the labeled dataset which is generated from $f:x \mapsto y$, 

such as $\mathbf{x}=[x_1, x_2, ..., x_m]$ and its corresponding $\mathbf{y}=[y_1, y_2, ..., y_m]$, 

the model should try to fit the original function by a learned function,  $f': x \mapsto \hat{y}$, 

with a relatively lower ½ Mean Squared Error (MSE) Loss, that is $\mathcal{L}=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y_i}-y_i)^2$.

## Conventions

* Use relatively pure Python to implement a DNN model that can fit any given mathematical function.
* No machine learning frameworks, eg. TensorFlow, PyTorch, scikit-learn, Keras, etc.
* Complete this task from imperfectly to perfectly within several version iterations.
* Regardless of efficiency.

## Notations

| Math                                      | Code   | Description                            |
| ----------------------------------------- | ------ | -------------------------------------- |
| $x_{i \_ j}$                              | `xi_j` | the ith weight/output of the jth layer |
| $\partial{\mathcal{L}} \over \partial{w}$ | `dw`   | (Partial) derivative                   |

The notations follow the conventions in Andrew Ng's Machine Learning course.

## Version History

| Version | Network Architecture      | Target                                                   |
| ------- | ------------------------- | -------------------------------------------------------- |
| V1      | (1+)2 Layers: (1 x) 2 x 1 | Use the simplest approach to implement a DNN regression. |
| V2      |                           | Introduce OOP                                            |