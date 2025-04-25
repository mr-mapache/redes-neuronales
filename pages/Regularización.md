
Dada una variable aleatoria $X: \Omega \rightarrow \mathbb{R}$ con densidad $p_X$ consideramos el cambio de variables $X = g(Y)$ en donde $Y: \Omega \rightarrow \mathbb{R}$ es otra variable aleatoria.  Se puede hallar la densidad $p_Y$ como:

$$
p_Y(\vec{y}) = p_X(\vec{x}) |\det J| 
$$

En donde $J$ es la matriz jacobiana cuyos elementos son las derivadas parciales $\partial g_i / \partial y^j$. 


Un problema común al entrenar un modelo es el cambio de la distribución de las covariables del conjunto de entrenamiento y las covariables con las que el modelo es puesto a prueba. Este problema es conocido como *covariate shift* y es un problema que tiene que ver con los datos. 

En redes neuronales profundas, este problema se da no solo a la diferencia entre los conjuntos de entrenamiento y prueba, sino también que también se da entre las capas de la red. Este problema es conocido como "internal covariate shift". Las covariables en las capas intermedias de la red neuronal cambian su distribución a medida que se actualizan los pesos de la red. Esto hace que el entrenamiento sea mas lento y difícil.



![[20250414121927.png]]

Los círculos corresponden a las neuronas o covariables. Las flechas corresponden a los pesos de la red. Si cambiamos los pesos de una capa, la distribución de las covariables en la siguiente capa cambia, por lo que en cada iteración del entrenamiento, la siguiente capa se encuentra con una distribución diferente de covariables a las que aprendió en la iteración anterior.

  

Esto causa dos problemas:

- La red se entrena mas lentamente.

- La red es inestable durante el entrenamiento.


---
**Bishop, C. M., & Bishop, H.** (2024). _Deep learning: Foundations and concepts_. Springer.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). _Dropout: A simple way to prevent neural networks from overfitting_. Journal of Machine Learning Research, 15(56), 1929–1958. http://jmlr.org/papers/v15/srivastava14a.html