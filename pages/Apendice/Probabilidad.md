En esta sección se repasan conceptos de probabilidad. La idea de esta sección no es brindar un marco de trabajo riguroso de la teoría de la probabilidad, sino mas bien lo contrario, tomar definiciones un tanto abstractas como las $\sigma$-algebras y extraer de estas conceptos útiles que se utilizaran a lo largo de este trabajo como el **espacio de eventos**.

Se asume que las funciones de esta sección tienen buen comportamiento (e.g. continuas, integrables o diferenciables en donde se necesite a menos que se especifique lo contrario). Además algunos conceptos se encuentran simplificados, por ejemplo definimos las integrales en términos de "elementos de volumen" $\delta \vec{x}$ y evitamos entrar en temas mas abstractos como la teoría de la medida. 

### Espacios de probabilidad

Un espacio de probabilidad $(\Omega, \mathcal{F}, P)$ es una construcción que se utiliza para modelar procesos aleatorios  y consiste en:

- Un **espacio de muestral** $\Omega$ el cual consiste el conjunto de todos los posibles resultados $\omega$ de un proceso aleatorio.  
- Un **espacio de eventos** $\mathcal{F}$ es el conjunto de todos los posibles eventos. Un evento $A$ es un subconjunto del espacio muestral.
- Una **función de probabilidad** $P$ es una función que asigna a cada evento $A$ del espacio de probabilidad un numero entre $0$ y $1$. 

Una variable aleatoria es una función $X: \Omega \rightarrow \mathcal{X}$ del espacio muestral a un espacio $\mathcal{X}$ al que denominaremos **espacio de características**. La probabilidad de que $X$ tome un subconjunto de valores $\mathcal{S}$ de $\mathcal{X}$ se escribe como:

$$
P(X \in S) = P(\{ \omega \in \Omega : X(\omega) \in \mathcal{S} \})
$$

Dada una variable aleatoria $X$, su **densidad de probabilidad** $p_X$ representa cuán probable es que la variable $X$ tome valores cercanos a $\vec{x}$. La probabilidad de que $X$ tome un valor dentro de una región $\mathcal{S}$ se aproxima como $p_X(\vec{x}) \delta \vec{x}$ cuando $\delta \vec{x}$ es un volumen pequeño que contiene a $\vec{x}$.

$$
P(X \in \mathcal{S}) = \int_{\mathcal{S}} p_X(\vec{x}) \delta \vec{x}
$$

La densidad de probabilidad debe ser no negativa, es decir $p_X(\vec{x}) \ge 0$ y la probabilidad de tomar un elemento del espacio de características total debe ser uno, es decir  $\int_{\mathcal{X}} p_X(\vec{x}) \delta\vec{x} = 1$.

Se hace necesario también definir los conceptos de **esperanza** o valor esperado y la **varianza**. Ambos son casos de particulares de lo que se denominan *momentos de una distribución*. Dada una variable aleatoria $X$ y $p_X$ su distribución de probabilidad, su esperanza viene dada por:

$$
\text{E}[X] = \int_\mathcal{X} \vec{x} p_X(\vec{x}) \delta{\vec{x}}
$$

La esperanza puede interpretarse como lo que esperamos obtener si repetimos un experimento aleatorio repetidas veces. Por otra parte podemos escribir la varianza usando el producto de Kronecker.

$$
\text{Var}[X] = \int_{\mathcal{X}} \Big[(\vec{x} - \text{E}[X]) \otimes (\vec{x} - \text{E}[X]) \Big] p_X(\vec{x}) \delta{\vec{x}}
$$

 Luego si los vectores $\vec{x}$ son tensores del tipo $(0,1)$, la varianza es un tensor del tipo $(0,2)$ cuyos elementos pueden interpretarse como una medida de cuanto varían las componentes de la variable aleatoria $X$, cada uno con respecto al resto.   

Todas estas definiciones pueden ser extendidas de forma sencilla a probabilidades discretas usando la **distribución de Dirac**, la cual permite representar distribuciones concentradas en puntos gracias a la siguiente propiedad:

$$
\int_{\mathcal{S}} \delta(\vec{x} - \vec{x}^i) \delta\vec{x} = \begin{cases}\vec{x}^i \text{ si }\vec{x}^i \in \mathcal{S} \\ 0 \text{ de lo contrario.}\end{cases}
$$

Si una variable aleatoria $X$ puede tomar valores $\{ \vec{x}^1, \vec{x}^2 ...,  \}$ con probabilidades $\{ p_1, p_2,... \}$. Podemos escribir su distribución como:

$$
p_X(\vec{x}) = \sum_ip_i \delta(\vec{x} - \vec{x}^i)
$$

Si tomamos particionamos el espacio de características $\mathcal{X}$ en un conjunto de regiones $\{ \mathcal{S}_i \}_{i\in I}$ tales que cada $\vec{x}^i$ se encuentran en una única región $S_i$, entonces la probabilidad de que la variable aleatoria tome un valor $\vec{x}_i$ será:

$$
P(X \in \mathcal{S}_i) = \int_{\mathcal{S}_i} p_X(\vec{x})\delta(\vec{x} - \vec{x}^i) \delta \vec{x} = p_i 
$$

De la misma forma podemos expresar la esperanza y la varianza en términos de probabilidades discretas:

$$
\text{E}[X] = \sum_{i\in I} p_i \vec{x}^i \qquad \text{Var}[X] = \sum_{i\in I} p_i (\vec{x}^i - \text{E}[X]) \otimes (\vec{x}^i - \text{E}[X])
$$

### Verosimilitud

Hasta ahora, hemos definido la **densidad de probabilidad** como una función que describe cuán probable es observar un determinado valor $\vec{x}$ si conocemos la distribución de la variable aleatoria $X$. 

Sin embargo, en muchos problemas de inferencia estadística la situación es inversa: conocemos (u observamos) un conjunto de datos, y lo que queremos es obtener información sobre la distribución que los generó, o más concretamente, sobre sus **parámetros**.

Supongamos que una variable aleatoria $X$ tiene una densidad de probabilidad $p_X(\vec{x} \mid \theta)$ que depende de un parámetro (o conjunto de parámetros) $\theta$. Si observamos un valor $\vec{x}$, definimos la **función de verosimilitud** como:

