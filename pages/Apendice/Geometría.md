### Variedad 

Una **variedad** (manifold) es una generalización del concepto de superficie en el espacio. Podemos modelar a una variedad como a un **espacio topológico** al cual podemos "cubrir" con subconjuntos del mismo**, de forma tal que, localmente, se comporten como espacios euclidianos. Este procedimiento de cubrir a un dicho espacio se denomina la **cobertura del atlas**.
 
 Veamos esto de forma mas rigurosa. Sea $M$ un espacio topológico, podemos decir que $M$ es una variedad si cumple con las siguientes propiedades:

- El espacio $M$ esta formado por un conjunto de subconjuntos $\{O_{\alpha}\}_{\alpha\in I}$ de $M$ que lo cubren, es decir, si $p$ es un punto de $M$ entonces $p$ pertenece a algún subconjunto $O_{\alpha}$.     

- Para cada sub-conjunto $O_{\alpha}$ de $M$, existe un mapa biyectivo y continuo $\psi_{\alpha}: O_{\alpha} \rightarrow U_{\alpha}$ donde $U_{\alpha}$  es un subconjunto abierto de $\mathbb{K}^n$.  Este mapa asegura que en cada uno de esos subconjuntos $O_{\alpha}$, se puede "localmente" describir sus puntos como si fueran puntos en el espacio $\mathbb{K}^{n}$. Estos mapas son lo que conocemos como *sistemas de coordenadas*.

-  Si dos sub-conjuntos $O_{\alpha}$ y $O_{\beta}$ se superponen, entonces sus mapas deben esta relacionados de manera coherente, es decir, existe una composición de los mapas $\psi_{\alpha}^{-1} \circ \psi_{\beta}$ que mapea la proyección de la intersección $O_{\alpha} \cap O_{\beta}$ a través de $\psi_{\beta}$, hacia la proyección de la misma intersección bajo $\psi_{\alpha}$.


![[20250225174334.png]]


En la física clásica, se asume que el espacio tiene una estructura natural de un espacio vectorial tridimensional. Esto significa que se puede tomar dos desplazamientos (vectores) en el espacio y sumarlos de una manera que siga las reglas matemáticas conocidas para los espacios vectoriales. Además, un punto del espacio se puede tomar como el **origen**, y los demás puntos se describen por sus desplazamientos desde allí.

Sin embargo, cuando la geometría del espacio es curva esto cambia. No se puede simplemente sumar dos puntos de la esfera y esperar obtener un tercer punto dentro de la misma. En geometrías curvas, no existe una estructura de espacio vectorial global que se mantenga en toda la geometría. A pesar de esto, es posible recuperar el concepto de "vector" y la estructura vectorial del espacio a través de la noción de "desplazamientos infinitesimales".

### Espacio tangente 

Un **vector tangente** en un punto de una variedad se puede entender como un operador lineal que actúa sobre funciones suaves, estos operadores, que son equivalentes a las **derivadas direccionales** y son una forma de describir vectores en espacios curvados.

Sea $\mathcal{F}$ a la colección de las funciones sobre una variedad $M$ en $\mathbb{R}$.  Dado un punto $p$ de $M$, decimos que el mapa $v: \mathcal{F} \rightarrow \mathbb{R}$, es un vector tangente en un punto $p$ si $v$ es lineal y obedece las reglas de Leibnitz:

- $v(af+bg) = av(f)+bv(g)$ para cada función $f, g$ de $\mathcal{F}$ y $a,b$ en $\mathbb{R}$.
- $v(fg) = fv(g) + gv(f)$  para cada función $f, g$ de $\mathcal{F}$.  

Es sencillo probar que la colección $V_p$ de todos los vectores tangentes al punto $p$ tiene una estructura de espacio vectorial. 

Una segunda propiedad muy importante relaciona la dimensión de una variedad con la dimensión del espacio tangente.  Dado un punto $p$ de una variedad $M$ de dimensión $n$, la dimensión de la colección $V_p$ también será de $n$. Veamos esto construyendo una base para el espacio $V_p$.

Dijimos que una variedad puede "cubrirse", con cartas que se comportan como espacios euclídeos, es decir que para algún $p$ de $M$ existe una carta $O$ que lo contiene y que existe un mapa $\psi: O \rightarrow U \subset \mathbb{R}^n$ que lo proyecta a un espacio euclídeo.  

La tercera propiedad de las variedades nos dice que si $f \in \mathcal{F}$, entonces por definición entonces la composición $f \circ \psi^{-1}:$ mapea los puntos del espacio $U \subset \mathbb{R}^n$ al espacio $\mathbb{R}$.


![[20250312142101.png]]

Se puede probar que dada una función $F: \mathbb{R}^n \rightarrow \mathbb{R}$, podemos escribir:

$$
F(x) = F(a) + \sum_{\mu=1}^n(x^{\mu} - a^{\mu})H_{\mu}(x) \qquad \text{En donde } H_{\mu}(a) = \frac{\partial F}{\partial x^{\mu}}\Big|_{x=a}
$$

Si tomamos $F = f \circ \psi^{-1}$, $a = \psi(p)$ y $q \in M$ tal que $x = \psi(q)$, entonces:

$$
f(q) = f(a) + \sum_{\mu=1}^n(x^{\mu} - a^{\mu}) H_{\mu}(\psi(q))
$$

Dijimos que $v$ era una transformación que tomaba elementos del espacio de funciones, luego si evaluamos $v$ en $f(q)$:

$$
v(f(q)) = \sum_{\mu=1}^n v(x^{\mu})H_{\mu}(\psi(q)) + (x^{\mu} - a^{\mu})v(H_{\mu}(\psi(q))
$$

Los términos constantes se anulan, por las reglas de Leibniz. Haciendo $q \rightarrow p$ se puede obtener:

$$
v(f(p)) = \sum_{\mu=1}^n v(x^{\mu}) \frac{\partial}{\partial x^{\mu}} (f\circ\psi^{-1})\Big|_{\psi(p)}
$$

Para $\mu = 1,2,...,n$ podemos entonces definir un "vector tangente"  $X_{\mu}: \mathcal{F} \rightarrow \mathbb {R}$ como:

$$ 
X_{\mu}(f) = \frac{\partial}{\partial x^{\mu}}(f\circ \psi^{-1}) \big|_{\psi(p)}
$$

Luego la acción del vector $v$ sobre una función $f \in \mathcal{F}$ es la de:

$$
v(f) = \sum_{\mu = 1}^{n} v(x^{\mu}) X_{\mu}(f)
$$

Esta base $\{X_{\mu}\}$ de $V_p$ es denominada *base coordenada* y suele denotarse simplemente como $\partial/\partial x^{\mu}$. El vector $v$ suele expresarse simplemente como:

$$
v = \sum_{\mu = 1}^{n} v^{\mu} \frac{\partial}{\partial x^{\mu}}
$$


Si elegimos un mapa distinto $\psi'$ se hubiera obtenido una base coordenada distinta $\{X'_{\mu}\}$. Se puede expresar la base $\{X_{\mu}\}$ en términos de otra base $\{X'_{\mu}\}$. Utilizando la regla de la cadena se pueden relacionar ambas bases como:

$$ 
X_{\mu} = \sum_{\nu =1}^n \frac{\partial x'^{\nu}}{\partial x^{\mu}} \Big|_{\psi(p)} X'_{\nu} 
$$

En donde $x'^{\nu}$ corresponde a las componentes del mapa $\psi' \circ \psi^{-1}$.  También un vector puede expresarse un nuevo sistemas de coordenadas de forma:

$$
v'^{\nu} = \sum_{\mu=1}^n v^{\mu} \frac{\partial x'^{\nu}}{\partial x^{\mu}}
$$





### Curvas

Una curva suave $C$ sobre una variedad $M$ es simplemente un mapa de $\mathbb{R}$ en $M$. En cada punto $p \in M$ podemos asociar con la curva $C$ al vector tangente $v \in V_p$ como sigue:

Para cada $f \in \mathcal{F}$ tomamos que $v(f)$ sea igual a la derivada de la función $f \circ \gamma: \mathbb{R} \rightarrow \mathbb{R}$ evaluada en $p$, es decir:

$$
v(f) = \frac{d(f\circ C)}{dt} \Big|_{C(t) =p}
$$

SI elegimos un mapa $\psi$ para mapear los puntos de $M$ al espacio $\mathbb{R}^n$, tal que al mapear la curva al rededor de $p$ como $\psi \circ C$, obtenemos una proyección sobre una sola dirección $x^{\mu}$ entonces podemos escribir el vector tangente $v$ como: 

$$
v(f) = \frac{d}{dt}(f\circ C) = \frac{d}{dt}(f\circ\psi^{-1} \circ \psi \circ C) = \sum_{\mu=1}^n \frac{\partial (f\circ\psi^{-1})}{\partial x^{\mu}} \frac{d x^{\mu}}{dt} \
$$

$$
v(f) = \sum_{\mu=0}^n \frac{dx^{\mu}}{dt}X_{\mu}(f)
$$

Luego, en cualquier base de coordenadas, las componentes del vector tangente a la curva estarán dadas por:

$$
v^{\mu} = \frac{dx^{\mu}}{dt}
$$

De esta forma se describe a los vectores tangentes como desplazamientos infinitesimales. 




Wald, R. M. (1984). _General relativity_. University of Chicago Press.