
El componente más básico de una red neuronal artificial es la **neurona**, inspirada en las neuronas biológicas. Una neurona biológica recibe señales eléctricas a través de sus **dendritas**, que la conectan con otras neuronas, el mecanismo por el cual llegan estas señales se denomina **sinapsis**. 

Una sinapsis es una conexión especializada entre neuronas, en la cual se transporta energía química mediante sustancias denominadas **neurotransmisores**, la señales se integran en la neurona mediante un proceso que modelaremos como una **agregación ponderada** y es denominado **preactivación**.

Si la agregación de estas señales supera un determinado **umbral**, se genera un impulso eléctrico denominado **activación**, que viaja a través del **axón**, una larga extensión encargada de transmitir la señal hacia las **dendritas** de otras neuronas.

Dicho esto, la tarea de una neurona puede resumirse en reaccionar a señales y transmitirlas. La intensidad de una señal viene determinada por realización de una observación $\omega$ de un conjunto de estados $\Omega$ a los cuales la neurona reacciona, y es de naturaleza estocástica, ya que está sujeta a la incertidumbre inherente al entorno. Podemos entonces representar a una señal como a una variable aleatoria $X: \Omega \rightarrow \mathbb{R}$ y cuya intensidad para una observación $\omega \in \Omega$ esta dada por $X(\omega)$. Si una neurona puede aceptar un conjunto de $d$ señales $X_1, X_2,..., X_d$, podemos describir su potencial de activación como una nueva variable aleatoria $Z$ dada por:

$$
Z = \sum_{\mu=1}^d X_{\mu}w^{\mu} + w^0
$$

En donde los coeficientes $w^1, w^2, ..., w^d$ representan a los pesos sinápticos que modulan las señales y $w^0$ representa la contribución de la misma neurona al potencial de activación. Estos pesos considerados parámetros, y pueden ser configurados en una neurona de manera tal que la respuesta de esta, ante las señales que recibe, sea óptima para la tarea que deba realizar. De esta forma una sola neurona puede ser vista como un regresor lineal durante la fase de preactivación. 

Supongamos que **lote** $\{\omega^1, ..., \omega^N \} \subset \Omega$  de observaciones. La neurona debe ajustar sus parámetros a modo de generar potenciales $\{  z^1, ..., z^N  \}$ **objetivos** para estas observaciones. Apoyados en el teorema del limite central, podemos asumir que los objetivos son realizaciones de una variable aleatoria cuya perturbación es un error gaussiano, es decir:

$$
|Z(\omega^{\kappa}) - z^{\kappa}| \sim \exp \Big[-\frac{\big(Z(\omega^{\kappa})-z^{\kappa}\big)^2}{2\sigma^2} \Big] \qquad \kappa = 1, 2,...,N
$$

En donde $\sigma^2$ es la varianza de la distribución. Podemos medir que tan bien el mapa $Z: \Omega \rightarrow \mathbb{R}$ explica estas observaciones mediante la **verosimilitud**, la cual podemos estimar como:

$$
\text{L}[Z] \sim \prod_{\kappa=1}^N  \exp \Big(-\frac{(Z(\omega^{\kappa}) - z^{\kappa})^2}{2\sigma^2} \Big)
$$

Dicho esto, el modelo que maximiza la probabilidad de obtener las predicciones observadas es aquel que maximiza la verosimilitud. Como el logaritmo es una función monótona y creciente, maximizar la verosimilitud es equivalente a maximizar su logaritmo. Tomando logaritmo en ambos miembros:

$$
\log \text{L}[Z] \sim - \frac{1}{2\sigma^2}\sum_{\kappa=1}^N \big(Z(\omega^{\kappa}) - z^{\kappa} \big)^2
$$

El ultimo termino corresponde a la suma de los errores cuadráticos entre las predicciones para las muestras del lote de datos y sus respectivos objetivos. Esto implica que la transformación $Z$  que maximiza la verosimilitud es aquella minimiza la suma de los cuadrados de la función de error sobre un lote, y nos define un **criterio** para entrenar una neurona, y lo expresamos mediante la función:

$$
\text{MSE}(\vec{z}, \vec{z}') = \frac{1}{N} \sum_{\kappa=1}^N \big( z^{\kappa}  - z'^{\kappa} \big)^2
$$

Esta función corresponde al **error cuadrático medio** y sirve como criterio para ajustar los pesos de una neurona.














Por otra parte, podemos agrupar a las señales de entrada en un vector de variables aleatorias $\vec{X}: \Omega \rightarrow \mathcal{X}$. El vector $\vec{X}$ toma elementos $\omega$ de un espacio muestral $\Omega$ representando a un conjunto de datos y devuelve realizaciones $\vec{x}$ de un espacio $\mathcal{X}$ al cual denominaremos **espacio de características**. 

Luego una






Estas señales se pueden agrupar en un vector de variables aleatorias $\vec{X}: \Omega \rightarrow \mathcal{X}$, de forma que $\vec{X}$ toma elementos $\omega \in \Omega$ y devuelve realizaciones en un espacio $\mathcal{X}$ al cual denominamos **espacio de características**. Los elementos de este espacio se conocen como **vectores de características** y podemos representarlos como:

$$
\vec{X}(\omega) = \vec{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_d \end{bmatrix} \in \mathcal{X}
$$










Por otra parte, podemos agrupar neuronas de forma tal que los condominios de cada una, correspondan a proyecciones de un nuevo espacio de características $\mathcal{X}'$. Esto es lo que se conoce como capa de neuronas.

Este espacio de características puede representar tanto como al dominio otra capa, como a un dominio de características de interés  al cual denominaremos espacio de objetivos. 





Consideramos un conjunto de muestras $\{\omega_1, ..., \omega_N \} \subset \Omega$ para el cual se conocen un mapa $Y: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{Y}$ a un espacio de objetivos real $\mathcal{Y} \subset \mathbb{R}$ y un mapa $X: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{X}$ a un espacio de características $\mathcal{X}$.

En este caso la agregación de señales en una neurona puede interpretarse como una regresión lineal, es decir, si el espacio $\Omega$ es plano, la neurona puede aprender a generalizar la composición $Y \circ X^{-1}: \mathcal{X} \rightarrow \mathcal{Y}$ y actuar como un mapa $\psi: \mathcal{X} \rightarrow \mathcal{Y}$ que permita inferir un objetivo $y \in \mathbb{R}$ fuera del conjunto de muestras a partir de una característica $\vec{x} \in \mathcal{X}$.  

$$
\psi(\vec{x}) = \sum_{i=1}^d w^{i}x_{i} = y
$$

Si $y'$ es el objetivo conocido:

$$
y = Y \circ X^{-1} (\vec{x})
$$






Si una red neuronal realiza 

$$

$$




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















$$
\psi= \sum_{i=1}^d w^{i}X_{i}
$$
$$
\int_{\mathcal{X}}p_{X_i}(\vec{x}) \delta\vec{x} = 1
$$


$$
\text{E}[X_i] = \int_{\mathcal{X}} p_{X_i}(\vec{x}) \vec{x}\delta\vec{x}
$$

$$
\text{Var}[X_i] = \int_X (\vec{x}-\text{E}[X_i])(\vec{x}-\text{E}[X_i])^T p_{X_i}(\vec{x}) \delta \vec{x}
$$

$$
\text{H}[X_{i}] = \int_{\mathcal{X}} p_{X_{i}}(\vec{x}) \log p_{X_i}(\vec{x}) \delta\vec{x}
$$

$$
\mathcal{N}(\vec{x} | \vec{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} \exp\Big[ -\frac{1}{2}\sum_{i,j}(x_i -\mu_i) \Sigma^{ij} (x_j - \mu_j) \Big]
$$





#### GLU

Las **Gated Linear Units ($\text{GLU}$)** son capas neuronales que regulan el flujo de información mediante un mecanismo de compuerta. Este mecanismo decide dinámicamente qué parte de la señal debe propagarse a través de la capa principal.

Sean:

- $W \in \mathbb{R}^d \otimes (\mathbb{R}^l)^*$ la matriz de pesos de la transformación principal y $\vec{b} \in \mathbb{R}^l$ su respectivo vector de bias. 
- $G \in \mathbb{R}^d \otimes (\mathbb{R}^l)^*$ La transformación de la puerta lógica con $\vec{c} \in \mathbb{R}^l$ su respectivo vector de bias.

La operación de una capa $\text{GLU}$ se define como:

$$
\text{GLU}(\vec{s}) = (W\vec{s} +\vec{b})\odot \sigma(G \vec{s} +\vec{c})
$$

En donde:

- $\odot$ denota el producto de Hadamard.
- $\sigma$ es la función sigmoide, que proyecta los valores de la compuerta en el rango $[0,1]$.

Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with gated convolutional networks. _Proceedings of the 34th International Conference on Machine Learning (ICML 2017)_, 70, 933-941. [https://arxiv.org/abs/1612.08083](https://arxiv.org/abs/1612.08083)