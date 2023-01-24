# DNN Regression Version 1.0

## Target

Use the simplest approach to implement a DNN regression.

## Network Design

[ ![DNNR V1.0 Architecture][svg-path] ][diagram-url]

Click the diagram to edit in diagrams.net

[svg-path]: ./DNNR%20V1.0%20Architecture.drawio.svg
[diagram-url]: https://app.diagrams.net/?src=about#Uhttps://raw.githubusercontent.com/Ki-Seki/DNN-Regression-Step-by-Step/master/DNNR%20V1.0/DNNR%20V1.0%20Architecture.drawio.svg

## Convention

* No vector manipulation
* No OOP
* Input is 1-dimensional

## Some Math

Loss function: $\mathcal{L(w, b)}=\frac{1}{m}\sum_{i=1}^{m}{\ell _i}$
, where $\ell _i=\frac{1}{2}(\hat{y_i}-y_i)^2$

Sigmoid activation function: $\sigma(x)=\frac{1}{1-e^{-x}}$

### Forward Propagation

*Layer 0*

$$
a^{[0]} = x
$$

*Layer 1*

$$
\begin{align*}
z_1^{[1]} &= w_1^{[1]}a^{[0]}+b_1^{[1]} \\
z_2^{[1]} &= w_2^{[1]}a^{[0]}+b_2^{[1]} \\
a_1^{[1]} &= \sigma(z_1^{[1]}) \\
a_2^{[1]} &= \sigma(z_2^{[1]})
\end{align*}
$$

*Layer 2*

$$
\begin{align*}
\mathbf{w}^{[2]} &\coloneqq \begin{pmatrix} w_1^{[2]} & w_2^{[2]} \end{pmatrix} \\
a^{[2]} &= w_1^{[2]}a_1^{[1]}+w_2^{[2]}a_2^{[1]}+b^{[2]}
\end{align*}
$$

*Loss*

$$
\begin{align*}
\hat{y} &= a^{[2]} \\
\ell &= \frac{1}{2}(\hat{y}-y)^2
\end{align*}
$$

### Backward Propagation

*Loss*

$$
\begin{align*}
\frac{d\ell}{d\hat{y}} &= \hat{y}-y = a^{[2]} -y
\end{align*}
$$

*Layer 2*

$$
\begin{align*}
\frac{d\ell}{dw_1^{[2]}} &= \frac{d\ell}{d\hat{y}} \cdot \frac{d\hat{y}}{dw_1^{[2]}} = (a^{[2]}-y) \cdot a_1^{[1]} \\
\frac{d\ell}{dw_2^{[2]}} &= \frac{d\ell}{d\hat{y}} \cdot \frac{d\hat{y}}{dw_2^{[2]}} = (a^{[2]}-y) \cdot a_2^{[1]} \\
\frac{d\ell}{db^{[2]}} &= \frac{d\ell}{d\hat{y}} \cdot \frac{d\hat{y}}{db^{[2]}} = (a^{[2]}-y) \cdot 1 \\
\frac{d\ell}{da_1^{[1]}} &= \frac{d\ell}{d\hat{y}} \cdot \frac{d\hat{y}}{da_1^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \\
\frac{d\ell}{da_2^{[1]}} &= \frac{d\ell}{d\hat{y}} \cdot \frac{d\hat{y}}{da_2^{[1]}} = (a^{[2]}-y) \cdot w_2^{[2]}
\end{align*}
$$

*Layer 1*

$$
\begin{align*}
\frac{d\ell}{dz_1^{[1]}} &= \frac{d\ell}{da_1^{[1]}} \cdot \frac{da_1^{[1]}}{dz_1^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_1^{[1]}(1-a_1^{[1]}) \\
\frac{d\ell}{dz_2^{[1]}} &= \frac{d\ell}{da_2^{[1]}} \cdot \frac{da_2^{[1]}}{dz_1^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_2^{[1]}(1-a_2^{[1]}) \\
\frac{d\ell}{dw_1^{[1]}} &= \frac{d\ell}{dz_1^{[1]}} \cdot \frac{dz_1^{[1]}}{dw_1^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_1^{[1]}(1-a_1^{[1]}) \cdot a^{[0]} \\
\frac{d\ell}{dw_2^{[1]}} &= \frac{d\ell}{dz_2^{[1]}} \cdot \frac{dz_2^{[1]}}{dw_2^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_2^{[1]}(1-a_2^{[1]}) \cdot a^{[0]} \\
\frac{d\ell}{db_1^{[1]}} &= \frac{d\ell}{dz_1^{[1]}} \cdot \frac{dz_1^{[1]}}{db_1^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_1^{[1]}(1-a_1^{[1]}) \cdot 1 \\
\frac{d\ell}{db_2^{[1]}} &= \frac{d\ell}{dz_2^{[1]}} \cdot \frac{dz_2^{[1]}}{db_2^{[1]}} = (a^{[2]}-y) \cdot w_1^{[2]} \cdot a_2^{[1]}(1-a_2^{[1]}) \cdot 1
\end{align*}
$$

### Batch Gradient Descent

The forward and backward propagation above is within one epoch. The parameters to be learned in this model will accumulate all the derivatives in every epoch and then be divided by the #epoch. Finally, use the fixed learning rate $\alpha$ to calculate the new parameters.

$$
\begin{align*}
&\text{Let } \theta \in \{ w_1^{[1]}, w_2^{[1]}, w_1^{[2]}, w_2^{[2]}, b_1^{[1]}, b_2^{[1]}, b^{[2]} \} \\
&{d\ell \over d\theta} = {1 \over m}\sum_{i=1}^m{d\ell \over d\theta_i} \\
&\theta \coloneqq \theta - \alpha {d\ell \over d\theta}
\end{align*}
$$