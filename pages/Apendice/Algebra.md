
El objetivo de esta sección no es redefinir los conceptos de álgebra lineal, sino establecer la notación y aclarar algunos conceptos no triviales que se utilizarán a lo largo del texto.

### Vectores

Dado un cuerpo $\mathbb{K}$ (por ejemplo, $\mathbb{R}$ o $\mathbb{C}$), consideraremos por el momento únicamente los vectores que son elementos de un espacio vectorial $\mathbb{K}^n$, a los cuales representamos como:

$$
\vec{v} = \begin{bmatrix} v^1 \\ v^2 \\ \vdots \\ v^n \end{bmatrix} \quad \text{con } v^i \in \mathbb{K} 
$$

Los índices **superiores** indican que estamos ante una base de coordenadas **contravariantes**, y nos referimos a estos vectores como **vectores columna**.

El conjunto de todas las aplicaciones lineales $f: \mathbb{K}^n \rightarrow \mathbb{K}$ se denomina el espacio **dual** de $\mathbb{K}^n$ y se denota por $\left(\mathbb{K}^n\right)^*$. Sus elementos se denominan **covectores** y pueden representarse como:

$$
\vec{\omega} = \begin{bmatrix} \omega_1 & \omega_2 & \cdots & \omega_n \end{bmatrix} \quad \text{con } \omega_i \in \mathbb{K} 
$$

Los índices **inferiores** indican una transformación **covariante**, y nos referimos a ellos como **vectores fila**. El resultado de evaluar un covector $\vec{\omega}$ sobre un vector $\vec{v}$ es:

$$
\vec{\omega}(\vec{v}) = \begin{bmatrix} \omega_1 & \omega_2 & \cdots & \omega_n \end{bmatrix} \begin{bmatrix} v^1 \\ v^2 \\ \vdots \\ v^n \end{bmatrix} = \sum_{i=1}^n \omega_i v^i \in \mathbb{K}
$$

Podemos desarrollar los elementos de un espacio primal $V$ en términos de una base $\{ \vec{e}_1, \vec{e}_2,..., \vec{e}_n\}$ y a los miembros del espacio dual $V^*$ por su respectiva base dual $\{ \vec{e}^1, \vec{e}^2,..., \vec{e}^n\}$ como:

$$
\vec{v} = \sum_{i=1}^n v^i\vec{e}_i \qquad \vec{\omega} = \sum_{i=1}^n \omega_i \vec{e}^i
$$

En donde se verifica que $\vec{e}^i(\vec{e}_j) = \delta^i_j$, con $\delta^i_j$ la delta de Kronecker. Este tipo de vectores son solamente casos particulares de transformaciones más generales. De hecho, algunos vectores ni siquiera admiten una representación matricial, ya que no todos los espacios vectoriales son de dimensión finita. Por ello, es fundamental comprender los vectores y covectores como **transformaciones**. 

Dado un espacio vectorial $V$ sobre un cuerpo $\mathbb{K}$, su espacio dual $V^*$ se define como el conjunto de todas las transformaciones lineales $T: V \rightarrow \mathbb{K}$, bajo este contexto, un covector es un ejemplo de un tensor del tipo $(0,1)$.  De la misma forma, un vector puede interpretarse como un tensor de tipo $(1,0)$ con propiedades geométricas que no se discutirán por el momento. 

#### Matrices

Es necesario también revisar el concepto de matriz. A primera instancia, podemos definir a una matriz $A$ simplemente como una colección de números, organizados en filas y columnas:

$$
A =  \begin{bmatrix}
	a_1^1 & a_2^1 & \cdots & a_n^1 \\
    a_1^2 & a_2^2 & \cdots & a_n^2 \\
    \vdots & \vdots & \ddots & \vdots \\
    a_1^m & a_2^m & \cdots & a_n^m \\
\end{bmatrix}
$$

Sin embargo, esta noción inicial, aunque útil, es limitada, ya que no explica como esta puede operar sobre los vectores de un espacio, por lo que se vuelve necesario profundizar más sobre que son realmente las matrices. 

Podemos decir que una matriz $A \in \mathbb{K}^{m\times n}$ representa a una transformación lineal:

$$ 
A: \mathbb{K}^n \rightarrow \mathbb{K}^m 
$$

En donde $\mathbb{K}^n$ y $\mathbb{K}^m$ son espacios vectoriales sobre el campo $\mathbb{K}$.  Esta definición de matriz como transformación lineal implica que una matriz $A$ toma un vector $\vec{v}$ de $\mathbb{K}^n$ y lo mapea en un vector $\vec{v}'$ de $\mathbb{K}^m$.  Cada fila de $A$ "extrae" una componente del vector de salida, por lo que las filas de $A$ son funcionales del espacio dual $(\mathbb{K}^n)^*$. Dicho esto, expresar entonces a la matriz $A$ como un elemento de un producto tensorial:

$$ 
A \in \mathbb{K}^m \otimes (\mathbb{K}^n)^*
$$

En donde $(\mathbb{K}^n)^*$ es el dual de $\mathbb{K}^n$.  Esta formulación de $A$ como elemento del espacio $\mathbb{K}^m \otimes (\mathbb{K}^n)^*$ sugiere interpretar a la matriz $A$ como un vector columna de dimensión $m$, cuyas filas son covectores:

$$
A = \begin{bmatrix} \vec{a}^1 \\ \vec{a}^2 \\ \vdots \\ \vec{a}^m \end{bmatrix}, \qquad \text{con} \quad \vec{a}^i = \begin{bmatrix} a_1^i & a_2^i & \cdots a_n^i\end{bmatrix} \in (\mathbb{K}^n)^*
$$

A partir de la interpretación de la matriz $A$ puede como una colección de $m$ vectores fila, se puede recuperar su representación original:

$$
A = \begin{bmatrix} \vec{a}^1 \\ \vec{a}^2 \\ \vdots \\ \vec{a}^m \end{bmatrix} =  \begin{bmatrix}
	a_1^1 & a_2^1 & \cdots & a_n^1 \\
    a_1^2 & a_2^2 & \cdots & a_n^2 \\
    \vdots & \vdots & \ddots & \vdots \\
    a_1^m & a_2^m & \cdots & a_n^m \\
\end{bmatrix}
$$

De la misma forma, podemos definir a la matriz transpuesta $A^T \in \mathbb{K}^{n \times m}$ como a la transformación dual:

$$ 
A^T: (\mathbb{K}^m)^* \rightarrow (\mathbb{K}^n)^*
$$

Luego, bajo esta idea, podemos escribir a la matriz transpuesta $A^T$:

$$
A^T =  \begin{bmatrix}
	a_1^1 & a_1^2 & \cdots & a_1^m \\
    a_2^1 & a_2^2 & \cdots & a_2^m \\
    \vdots & \vdots & \ddots & \vdots \\
    a_n^1 & a_n^2 & \cdots & a_n^m \\
\end{bmatrix}
$$

Luego si $\vec{\omega}$ es un vector del dual $(\mathbb{K}^n)^*$, es ahora $\vec{\omega}$ quien toma cada vector columna de $A^T$ y extrae una componente de un vector dual de salida $\vec{\omega}'$ de $(\mathbb{K}^m)^*$.

Esta interpretación de una matriz transpuesta no solo clarifica su acción sobre vectores, sino que también simplifica entender la composición de transformaciones lineales.

Veamos ahora el producto de matrices. Sean ahora las matrices $A \in \mathbb{K}^m \otimes (\mathbb{K}^n)^*$ y $B \in \mathbb{K}^n \otimes (\mathbb{K}^p)^*$, luego el producto de matrices $AB$ es simplemente la composición:

$$
AB: \mathbb{K}^m \rightarrow (\mathbb{K}^p)^*
$$

Esta operación es un caso particular de la **contracción entre tensores** en donde un tensor de $\mathbb{K}^m \otimes (\mathbb{K}^n)^*$ se contrae con un tensor de $B \in \mathbb{K}^n \otimes (\mathbb{K}^p)^*$, de hecho estas matrices son casos particulares de tensores de tipo $(1,1)$.