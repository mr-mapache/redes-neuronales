
Un espacio de probabilidad $(\Omega, \mathcal{F}, P)$ es una construcción que se utiliza para modelar procesos aleatorios  y consiste en:

- Un **espacio de resultados** $\Omega$ el cual consiste el conjunto de todos los posibles resultados $\omega$ de un proceso aleatorio.  
- Un **espacio de eventos** $\mathcal{F}$ es el el conjunto de todos los posibles eventos. Un evento $A$ es un subconjunto del espacio de resultados.
- Una **función de probabilidad** $P$ es una función que asigna a cada evento $A$ del espacio de probabilidad un numero entre $0$ y $1$. 

Una variable aleatoria es una función $X: \Omega \rightarrow E$ del espacio de resultados a un espacio mensurable $E$ y la probabilidad de que $X$ tome un subconjunto de valores $S$ de $E$ se escribe como:

$$
P(X \in S) = P(\{ \omega \in \Omega : X(\omega) \in S \})
$$

Una **densidad de probabilidad** es una función que describe cómo se distribuye la probabilidad de una **variable aleatoria continua** sobre su espacio de valores y cuya integral nos da la probabilidad de que la variable aleatoria tome valores en $S$, es decir: 

$$
P(X \in S) = \int_S p_X(\vec{x}) d\vec{x}
$$
