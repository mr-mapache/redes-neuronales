
Una hipótesis ampliamente aceptada para explicar el éxito del aprendizaje profundo es la existencia de una estructura de **variedad (manifold)** en los datos. Según esta suposición, los datos naturales de alta dimensión no se distribuyen de forma aleatoria, sino que tienden a concentrarse cerca de una variedad no lineal de baja dimensión.

El objetivo de un modelo es, entonces, aprender tanto la estructura geométrica subyacente en los datos como la distribución de "energía" que reside sobre ella, y obtener una representación paramétrica de la variedad que describe dichos datos.

Bajo este marco, definimos al **espacio ambiente** $\mathcal{X}$ como aquel espacio de alta dimensión en el que originalmente viven los datos. Geométricamente, es el espacio en el que la variedad está embebida.

### Autoencoders

Un **autoencoder** es un modelo que aprende un mapa de codificación $\psi: \mathcal{X}  \rightarrow U$ y un mapa de decodificación $\phi: U \rightarrow \mathcal{X}$ en donde $\mathcal{F}$ es el **espacio latente**.  Si los datos se encuentran sobre una variedad diferencial $M$, esta variedad puede aprender a reconstruirse usando que.

$$
M' = (\psi \circ \phi)(M)
$$

En donde $M'$ es la variedad reconstruida. Supongamos que $M$ es el soporte para una distribución de energía $E: \mathcal{X} \rightarrow \mathbb{R}$. 

Sabemos que cada punto de $M$ existe una carta $(O, \psi)$ local tal que $O$ es un subconjunto de $M$ y $\psi: O \rightarrow U$ es el *mapa de codificación* que mapea el subconjunto $O$ en un espacio euclídeo. El mapa de decodificación es el inverso $\phi \sim \psi^{-1}: U \rightarrow O$ que mapea vectores a puntos de la variedad. 











Sabemos que una variedad puede cubrirse mediante una familia de subconjuntos abiertos y que cada punto de un subconjunto $O$ puede describirse localmente como si fuera un punto del espacio euclidiano, es decir, existe un mapa $\psi: O \rightarrow U$, donde $U$ es un subconjunto abierto de $\mathbb{R}^n$.










Supongamos ahora que tenemos un modelo que toma estos datos 


Dicho esto, definimos al **espacio latente** es un espacio de menor dimensión, al que el modelo mapea los datos originales, buscando capturar su estructura esencial. 




En general, existe una distribución de energía $E$ en el

Bajo este marco podemos por ejemplo definir 





- El **espacio ambiente** (o _ambient space_) es el espacio de alta dimensión en el que los datos están originalmente representados. Por ejemplo, una imagen de 64x64 píxeles con 3 canales de color puede ser vista como un punto en $\mathbb{R}^{12288}$. Este espacio contiene toda la información posible, incluyendo redundancias, correlaciones y variaciones irrelevantes. **Geométricamente, es el espacio en el que la variedad está embebida**, es decir, donde “vive” su representación.
    
- La **variedad** es una subestructura de menor dimensión que se encuentra "curvada" dentro del espacio ambiente. Representa la **estructura intrínseca** de los datos: las combinaciones de variables que realmente ocurren en la naturaleza. Aunque el espacio ambiente puede ser de dimensión muy alta, los datos reales suelen concentrarse cerca de una región mucho más pequeña y regular: esa es la variedad.
    
- El **espacio latente** (o _latent space_) es un espacio de baja dimensión que **modela la variedad desde el punto de vista intrínseco**. No se trata de una subregión del espacio ambiente, sino de un espacio nuevo —como un sistema de coordenadas abstracto— desde donde se puede parametrizar la variedad. Desde la perspectiva de la definición anterior, el espacio latente se corresponde con el dominio de las cartas $\psi_{\alpha}^{-1}$: regiones abiertas de $\mathbb{C}^n$ desde las cuales se mapea a porciones locales de la variedad en el espacio ambiente.
    
- Finalmente, un **embedding** es el conjunto de funciones $\psi_{\alpha}^{-1}$ que, al actuar sobre subconjuntos del espacio latente, permiten "construir" localmente la variedad dentro del espacio ambiente. En el aprendizaje profundo, estas funciones son aprendidas por modelos como _autoencoders_, _VAEs_ o redes generativas, que intentan aprender una aplicación suave que preserve la geometría local al pasar del espacio latente al ambiente.


El **espacio latente** es una representación matemática en la que objetos complejos (e.g. palabras, imágenes, sonidos o incluso conceptos abstractos) que se codifican como **vectores** dentro de un **espacio de dimensión fija**.  Las relaciones entre objetos pueden analizarse y manipularse a través de operaciones como:

- La **suma vectorial**, que puede capturar composiciones de significado. Por ejemplo, en lenguaje, el famoso caso:  
    `vec("rey") - vec("hombre") + vec("mujer") ≈ vec("reina")`,  
    sugiere que ciertas transformaciones semánticas se representan como **desplazamientos lineales** consistentes dentro del espacio latente.
    
- El **producto escalar** o **coseno del ángulo entre vectores** sirve como medida de **similitud**. Vectores que apuntan en direcciones similares (con ángulo pequeño entre ellos) representan objetos con alta cercanía semántica.
    
- Las **direcciones específicas en el espacio** pueden codificar atributos abstractos o dimensiones latentes como género, estilo, tono, categoría, etc., dependiendo del dominio (lenguaje, visión, audio).

Así, el espacio latente no es un mero conjunto de puntos, sino un entorno **geométricamente estructurado** donde es posible realizar **operaciones lineales** que corresponden a manipulaciones conceptuales.





Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). _Efficient estimation of word representations in vector space_. arXiv. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)

Lei, N., Luo, Z., Yau, S.-T., & Gu, D. X. (2018). _Geometric understanding of deep learning_. arXiv preprint arXiv:1805.10451. [https://arxiv.org/abs/1805.10451](https://arxiv.org/abs/1805.10451)

Su, J., Lu, Y., Pan, S. J., Wen, J., Liu, Y., & Sun, M. (2021). _RoFormer: Enhanced Transformer with Rotary Position Embedding_. arXiv. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)