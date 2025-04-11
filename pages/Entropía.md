En el contexto del aprendizaje profundo, los modelos aprenden a predecir distribuciones de probabilidad sobre posibles resultados. Evaluar la calidad de estas predicciones requiere una medida que cuantifique cuánta incertidumbre o error hay entre las probabilidades predichas y las verdaderas. Para esto, conceptos de la teoría de la información como la **entropía**  juegan un rol fundamental. 

Definimos al espacio muestral $\Omega$ como al conjunto de todos los posibles resultados de un experimento aleatorio. Dado un experimento, nos interesa la ocurrencia de determinados **eventos**, es decir subconjuntos $A$ de $\Omega$ que agrupan uno o más resultados posibles del experimento. 

Buscamos una forma de cuantificar la incertidumbre asociada a la ocurrencia de un evento. Para ello definimos la **entropía** como una medida de la incerteza. 

Supongamos que tenemos una serie de $n$ eventos con probabilidad de ocurrencia $p_1, p_2, ..., p_n$ respectivamente. 

- La entropía es continua para cada $p_i$.
- Si $p_i = 1$ entonces la entropía debe ser nula, no hay incertidumbre si el resultado es seguro.
- Si $p_1 = p_2 = ... = p_n$ entonces la entropía, debe ser máxima, ya que la ocurrencia de cada evento es igual de probable. 
- Si la ocurrencia de un evento se descompone en la ocurrencia de dos eventos sucesivos, la entropía total debe ser la suma ponderada de las entropías correspondientes.

Luego, la función que satisface estas condiciones puede escribirse como:

$$ 
S = - \sum_{i=1}^n p_i \log p_i
$$

Esta expresión es conocida como la entropía de Shannon. Si tenemos un modelo que predice probabilidades $q_1, q_2, ..., q_n$ para estos eventos, el modelo no conoce las probabilidades reales de ocurrencia, por lo que definiremos la entropía según el modelo como: 

$$ 
S = - \sum_{i=1}^n q_i \log q_i
$$

De esta forma podemos diferenciar la incertidumbre intrínseca de los datos verdaderos, es decir, cuán impredecible es el resultado si conociéramos la verdadera distribución de probabilidad y cual impredecible es un resultado según la distribución de probabilidad que predice el modelo.

Esta distinción entre la entropía de los datos y la entropía del modelo nos permiten definir la **entropía cruzada** como:

$$
S = - \sum_{i=1}^n p_i \log q_i
$$

En donde $p_i$ son las probabilidades reales de ocurrencia de los eventos y $-\log q_i$ los valores de entropía individúales para las predicciones del modelo. 

La entropía cruzada, mide discrepancia entre la distribución real y la predicha. Representa la cantidad promedio de información adicional necesaria para codificar muestras verdaderas utilizando la distribución del modelo como referencia.

Notemos que las probabilidades $q_1, q_2, ..., q_n$ predichas por el modelo representan una distribución de probabilidad sobre posibles eventos disjuntos, por lo que, por definición, deben sumar 1:

$$
\sum_{i=1}^n q_i = 1
$$

Esto impone una restricción sobre la entropía cruzada. Veamos ahora que sucede si buscamos minimizamos esta entropía teniendo en cuenta esta restricción. Usando el método de los multiplicadores de Lagrange proponemos:

$$ 
S = - \sum_{i=1}^n p_i \log q_i + \lambda \Big(\sum_{i=1}^n q_i - 1 \Big)
$$

Derivando esta expresión con respecto a $q_i$ para $i = 1, 2, ...,n$ obtenemos:

$$
\frac{\partial S}{\partial q_i} = -\frac{p_i}{q_i} + \lambda = 0 \Rightarrow q_i =  \frac{p_i}{\lambda} \qquad i= 1,2,...,n
$$

Reemplazando en la restricción:

$$
\sum_{i=1}^n q_i = \sum_{i=1}^n \frac{p_i}{\lambda} = 1 \Rightarrow \sum_{i=1}^n p_i = \lambda
$$

Luego si los eventos son disjuntos, la suma de sus probabilidades de ocurrencia $p_1, p_2, ..., p_n$ debe también ser uno, por lo que $\lambda = 1$. Luego las probabilidades predichas por el modelo serán iguales a las probabilidades conocidas para los eventos:

$$ 
q_i = p_i
$$

La interpretación de esto es sencilla, para que un modelo arroje buenos resultados, se debe reducir el valor de la entropía cruzada sobre sus resultados. Esto nos permite definir la **divergencia de Kullback-Leibler** como la diferencia entre la entropía cruzada y la entropía verdadera:

$$ 
D_{KL} =  \sum_{i=1}^n p_i \log \frac{p_i}{q_i}
$$

La divergencia KL captura cuán **ineficiente** es el modelo para representar la verdadera distribución. Es siempre no negativa y se anula si, y solo si, $p_i=q_i$ para todos los $i=0,1,...,n$

Shannon, C. E. (1948). _A mathematical theory of communication_. **Bell System Technical Journal, 27**(3), 379-423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

**Jaynes, E. T. (1957).** Information theory and statistical mechanics. _Physical Review_, **106**(4), 620–630. https://doi.org/10.1103/PhysRev.106.620
