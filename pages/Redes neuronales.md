
El componente más básico de una red neuronal artificial es la **neurona**, inspirada en las neuronas biológicas. Cada neurona realiza dos operaciones fundamentales: la **preactivación** y la **activación** (o "disparo").

La preactivación $z_i$ de una neurona es una agregación lineal de señales $s_j$ en donde cada señal se pondera con un peso $w^j_i$  se sesga por un bias $b_i$.

$$ 
z_i(s) = b_i + \sum_{j = 1}^d w_i^j s_j
$$

Luego cada neurona se puede *disparar* de acuerdo al valor de la preactivación, produciendo una activación:

$$
a_i = \sigma(z_i(s)) 
$$

La función $\sigma$ se denomina **función de activación** y actúa de forma independiente sobre cada preactivación.

Puestas juntas, $l$ neuronas forman una **capa**. Luego podemos describir la acción de una red neuronal sobre un vector de señales $\vec{s}\in \mathbb{R}^d$ como:

$$
\vec{a} =  \sigma(\begin{bmatrix}
	w_1^1 & w_2^1 & \cdots & w_d^1 \\
    w_1^2 & w_2^2 & \cdots & w_d^2 \\
    \vdots & \vdots & \ddots & \vdots \\
    w_1^h & w_2^h & \cdots & w_d^h \\
\end{bmatrix} \begin{bmatrix}
	s^1 \\ s^2 \\ \vdots \\ s^d
\end{bmatrix} +  \begin{bmatrix}
	b^1 \\ \vdots \\ b^l
\end{bmatrix})
$$

Esta capa transforma un vector de señales $\vec{s} \in \mathbb{R}^d$ en un vector de activación $\vec{a}\in \mathbb{R}^l$ como:






producen un vector $l$-dimensional de activación. 


Y la activación es simplemente:

$$
\vec{a} = \sigma(\vec{z})
$$

Utilizando estas capas, se pueden "apilar" de forma tal que las activaciones de una capa se conviertan en las señales de entrada de la siguiente.  La organización de estas capas se denomina la arquitectura de la red neuronal. La cantidad de capas $L$ de una red se denomina profundidad de la red. 






$$ 
\vec{z} = \begin{bmatrix} s_1 & s_2 &...& s_d \end{bmatrix} \begin{bmatrix}
	w_1^1 & w_1^2 & \cdots & w_1^d \\
    w_1^2 & w_2^2 & \cdots & w_2^d \\
    \vdots & \vdots & \ddots & \vdots \\
    w_1^l & w_2^l & \cdots & w_d^l \\
\end{bmatrix}^T + \begin{bmatrix} b_1 & b_2 & \cdots & b_l \end{bmatrix}
$$

Y la activación es simplemente:

$$
\vec{a} = \sigma(\vec{z})
$$

Utilizando estas capas, se pueden "apilar" de forma tal que las activaciones de una capa se conviertan en las señales de entrada de la siguiente.  La organización de estas capas se denomina la arquitectura de la red neuronal. La cantidad de capas $L$ de una red se denomina profundidad de la red. 