
El componente más básico de una red neuronal artificial es la **neurona**, inspirada en las neuronas biológicas. Una neurona biológica recibe señales eléctricas a través de sus **dendritas**, que la conectan con otras neuronas, el mecanismo por el cual llegan estas señales se denomina **sinapsis**. 

Una sinapsis es una conexión especializada entre neuronas, en la cual se transporta energía química mediante sustancias denominadas **neurotransmisores**, la señales se integran en la neurona mediante un proceso que modelaremos como una **agregación ponderada** y es denominado **preactivación**.

Si la agregación de estas señales supera un determinado **umbral**, se genera un impulso eléctrico denominado **activación**, que viaja a través del **axón**, una larga extensión encargada de transmitir la señal hacia otras neuronas.

Podemos modelar a la preactivación de una neurona como una variable aleatoria $Y: \Omega \rightarrow \mathbb{R}$ definida sobre un espacio $\Omega$ que representa al conjunto de posibles estímulos capaces de generar una respuesta en esta, y puede verse como un espacio muestral. Cada realización $Y(\omega)$ dado un estimulo $\omega \in \Omega$ representa al nivel de activación. 

Una neurona individual puede ajustarse para actuar como un regresor lineal durante la fase de preactivación. Supongamos que ante un estimulo $\omega$ una neurona alcanza un nivel de activación $Y(\omega)$. Si se establece que el nivel de activación requerido para la neurona es de $y \in \mathbb{R}$, puede escribirse una expresión del error en la inferencia como:

$$
\xi = |Y(\omega) - y| 
$$

Si asumimos que el error $\xi$ sigue una **distribución normal** $\mathcal{N}(0, \sigma^2)$, podemos escribir escribir a la distribución del error como:

$$
\rho_{\xi} \sim \exp \Big[-\frac{\big(Y(\omega)-y\big)^2}{2\sigma^2} \Big]
$$

En donde $\sigma^2$ es la varianza de la distribución. Esta asunción se apoya en el Teorema del Límite Central, ya que el error suele ser resultado de múltiples fuentes independientes de variabilidad. 

Tomando un conjunto de muestras o **lote** $\{\omega^1, ..., \omega^N \} \subset \Omega$ para el cual se conocen un conjunto de observaciones correspondientes a sus respectivos objetivos $\{  y^1, ..., y^N  \}$, podemos medir que tan bien un mapa $Y: \Omega \rightarrow \mathcal{Y}$ explica estas observaciones mediante la **verosimilitud**, la cual podemos estimar como:

$$
\text{L}[Y] \sim \prod_{i=1}^N  \exp \Big(-\frac{(Y(\omega^i) - y^i)^2}{2\sigma^2} \Big)
$$

Dicho esto, el modelo que maximiza la probabilidad de obtener las predicciones establecidas es aquel que maximiza la verosimilitud. Si tomamos el logaritmo en ambos miembros:

$$
\log \text{L}[Y] \sim - \frac{1}{2\sigma^2}\sum_{i=1}^N \big(Y(\omega^i) - y^i \big)^2
$$

Se puede ver que el mapa $Y$ que maximiza la verosimilitud es aquel minimiza la suma de los cuadrados de la función de error del lote de datos, esto nos define un **criterio** para entrenar una neurona. 

Podemos modelar, las conexiones sinápticas de la neurona, mediante un vector de pesos con elementos $w^1, w^2, ..., w^d$ que modulan las señales, las cuales son realizaciones de un conjunto de variables aleatorias $X_1, X_2,..., X_d$. Esta agregación produce una nueva variable aleatoria $Y$ que podemos escribir como:

$$
Y = \sum_{i=1}^d w^{i}X_{i} + w^0
$$

En donde $w^0$ es un parámetro denominado **sesgo** y permite regular el umbral de activación de una neurona.

Se pueden agrupar las señales de entrada en un vector de variables aleatorias $\vec{X}: \Omega \rightarrow \mathcal{X}$, de forma que $\vec{X}$ toma elementos $\omega \in \Omega$ y devuelve realizaciones en un espacio $\mathcal{X}$ al cual denominaremos **espacio de características**.  
 
Por otra parte, una neurona puede ajustar sus pesos para aprender a predecir valores de un dominio $\mathcal{Y}$ al cual denominaremos **espacio de resultados**, a partir de características conocidas del espacio $\mathcal{X}$. 










dado un conjunto de muestras $\{\omega_1, ..., \omega_N \} \subset \Omega$ para el cual se conocen un mapa $Y: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{Y} \subset \mathbb{R}$ y un mapa $X: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{X}$ podemos definir la transformación de las


Consideramos un conjunto de muestras $\{\omega_1, ..., \omega_N \} \subset \Omega$ para el cual se conocen un mapa $Y: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{Y}$ a un espacio de objetivos real $\mathcal{Y} \subset \mathbb{R}$ y un mapa $X: \{\omega_1, ..., \omega_N \} \rightarrow \mathcal{X}$ a un espacio de características $\mathcal{X}$.












 la neurona puede aprender a generalizar la composición $Y \circ X^{-1}: \mathcal{X} \rightarrow \mathcal{Y}$ y actuar como un mapa $\psi: \mathcal{X} \rightarrow \mathcal{Y}$ que permita inferir un objetivo $y \in \mathbb{R}$ fuera del conjunto de muestras a partir de una característica $\vec{x} \in \mathcal{X}$.  

$$
\psi(\vec{x}) = \sum_{i=1}^d w^{i}x_{i} = y
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
 \begin{bmatrix}
	w_1{}^1 & w_2{}^1 & \cdots & w_l{}^1 \\
	w_1{}^2 & w_2{}^2 & \cdots & w_l{}^2 \\
    \vdots & \vdots & \ddots & \vdots \\
	w_1{}^d & w_2{}^d & \cdots & w_l{}^d \\
	b_1 & b_2 & \cdots & b_l
\end{bmatrix}
$$