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

Loss function: $\mathcal{L(w, b)}=\frac{1}{m}\sum_{i=1}^{m}{l_i}$, where $l_i=\frac{1}{2}(\hat{y_i}-y_i)^2$

Sigmoid activation function: $\sigma(x)=\frac{1}{1-e^{-x}}$

### Forward Propagation

$$
\begin{align*}

a^{[0]} &= x \\ \\

z_1^{[1]} &= w_1^{[1]}a^{[0]}+b_1^{[1]} \\
z_2^{[1]} &= w_2^{[1]}a^{[0]}+b_2^{[1]} \\
a_1^{[1]} &= \sigma(z_1^{[1]}) \\
a_2^{[1]} &= \sigma(z_2^{[1]}) \\ \\

\mathbf{w}^{[2]} &\coloneqq
\begin{pmatrix} w_1^{[2]} & w_2^{[2]} \end{pmatrix} \\
a^{[2]} &= w_1^{[2]}a_1^{[1]}+w_2^{[2]}a_2^{[1]}+b^{[2]} \\ \\

\hat{y} &= a^{[2]} \\
l &= \frac{1}{2}(\hat{y}-y)^2

\end{align*}
$$

### Backward Propagation

$$
\text{TODO}
$$