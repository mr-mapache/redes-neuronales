- $\vec{x} \in (\mathbb{R}^*)^d$
- $W \in  (\mathbb{R}^*)^d \otimes  (\mathbb{R}^*)^l$

$$
\vec{x} = \begin{bmatrix} x_1 &x_2 &\cdots &x_d \end{bmatrix}
$$

$$ W = \begin{bmatrix}
	w^1{}_1 & w^1{}_2 & \cdots & w^1{}_d \\
	w^2{}_1 & w^2{}_2 & \cdots & w^2{}_d \\
    \vdots & \vdots & \ddots & \vdots \\
	w^l{}_1 & w^l{}_2 & \cdots & w^l{}_d \\
\end{bmatrix}
$$

$$
W^T = \begin{bmatrix}
	w_1{}^1 & w_1{}^2 & \cdots & w_l{}^1 \\
	w_1{}^2 & w_2{}^2 & \cdots & w_k{}^2 \\
    \vdots & \vdots & \ddots & \vdots \\
	w_1{}^d & w_2{}^d & \cdots & w_l{}^d \\
\end{bmatrix}
$$

$$
\frac{\partial}{\partial w^{\mu}{}_{\nu}} = \begin{bmatrix}  & & \cdots  \\ & \vdots  & \ddots & \\ & & & 1 & \vdots \\ & & & \cdots\end{bmatrix} \qquad \frac{\partial}{\partial x^{\mu}} = \begin{bmatrix} & \cdots & 1 & \cdots &  \cdots & \end{bmatrix}
$$


$$
\vec{y} = \vec{x} W^T
$$

$$
y_{\nu} = (\vec{x} W^T)_{\nu} = [\vec{x}]_{\mu} [W^T]^{\mu}{}_{\nu} = x_{\mu} w^{\alpha}{}_{\beta} g_{\alpha \nu} g^{\beta \mu} 
$$

$$
\frac{\partial y_{\nu}}{\partial w^{\alpha}{}_{\beta}} = x_{\mu} g_{\alpha \nu} g^{\beta \mu} = g_{\alpha \nu} x^{\beta}
$$

$$
\frac{\partial \mathcal{L}}{\partial y_{\nu}}\frac{\partial y_{\nu}}{\partial w^{\alpha}{}_{\beta}} = 
j^{\nu}
g_{\alpha \nu} x^{\beta} = j_{\alpha} x^{\beta}
$$
 
$$
\vec{\nabla}_{W} \mathcal{L} = g^{\alpha \mu} g_{\beta \nu} \frac{\partial \mathcal{L}}{\partial y_{\nu}}\frac{\partial y_{\nu}}{\partial w^{\alpha}{}_{\beta}}\frac{\partial}{\partial w^{\mu}{}_{\nu}}
$$
  $$
\vec{\nabla}_{W} \mathcal{L} = g^{\alpha \mu} g_{\beta \nu} j_{\alpha} x^{\beta} \frac{\partial}{\partial w^{\mu}{}_{\nu}} = j^{\mu}x_{\nu}\frac{\partial}{\partial w^{\mu}{}_{\nu}} = \vec{j}^T \vec{x}
$$

$$
\frac{\partial y_{\nu}}{\partial x_{\mu}} = w^{\alpha}{}_{\beta} g_{\alpha \nu} g^{\beta \mu} 
$$

$$
\frac{\partial \mathcal{L}}{\partial y_{\nu}} \frac{\partial y_{\nu}}{\partial x_{\mu}} = j^{\nu} w^{\alpha}{}_{\beta} g_{\alpha \nu} g^{\beta \mu} = j_{\alpha}w^{\alpha \mu}
$$

$$
\vec{\nabla}_{\vec{x}}\mathcal{L} = g_{\mu \nu} \frac{\partial \mathcal{L}}{\partial x_{\mu}} \frac{\partial}{\partial x_{\nu}}
$$

$$
\vec{\nabla}_{\vec{x}}\mathcal{L}  = g_{\mu \nu}j_{\alpha}w^{\alpha \mu}\frac{\partial}{\partial x_{\nu}} = j_{\alpha}w^{\alpha}{}_{\nu} \frac{\partial}{\partial x_{\nu}} = \vec{j} W
$$

