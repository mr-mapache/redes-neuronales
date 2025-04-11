
Para modelar un conjunto de posibles estados o eventos, asignamos a cada uno una cantidad escalar $E_i$ que denominamos **energía**. Esta energía no debe entenderse en el sentido físico, sino como una medida abstracta de activación dentro del sistema y esta abierta a interpretación. 

Supongamos que un modelo que asigna probabilidades $q_1, q_2, ..., q_n$ a estos estados, sujeto a las siguientes condiciones:

$$
\sum_{i=1}^n q_i = 1 \qquad \sum_{i=1}^n q_i E_i = U
$$

La segunda introduce una cantidad escalar $U$, que representa la **energía promedio del sistema** según la distribución. Esta restricción nos permite incorporar una noción de balance o expectativa energética en el modelo.

Veamos ahora como se relacionan las probabilidades y las energías. Aplicando el método de los multiplicadores de Lagrange sobre la entropía de del modelo:

$$
S = -\sum_{i=1}^n q_i \log q_i + \alpha\Big(\sum_{i=1}^nq_i - 1 \Big) + \beta\Big(\sum_{i=1}^nq_i E_i - U\Big) 
$$

Derivando con respecto al termino $q_i$ obtenemos:

$$
\frac{\partial S}{\partial q_i} = -1 - \log q_i +  \alpha + \beta E_i = 0
$$

$$
\Rightarrow q_i = \exp(\alpha - 1 + \beta E_i) = \exp(\alpha - 1)\exp(\beta E_i)
$$

Sumando ambos miembros de la expresión teniendo en cuenta la restricción sobre las probabilidades:

$$
\sum_{i=1}^n q_i = 1 = \exp(\alpha- 1) \sum_{i=1}^n\exp(\beta E_i)
$$

Definimos de esta forma la función de partición del sistema como:

$$
Z(\alpha) \equiv \exp(1 - \alpha) = \sum_{i=1}^n\exp(\beta E_i)
$$

Reemplazando en la expresión obtenida para $q_i$ anteriormente, podemos expresar las probabilidades predichas por el modelo en términos de las energías. 

$$
q_i = \frac{\exp(\beta E_i)}{\sum_{i=1}^n\exp(\beta E_i)} \equiv \text{Softmax}(\beta E_i)  \qquad i=1,...,n
$$

Es importante destacar que no se ha especificado una forma explícita para las energías $E_i$​. Las probabilidades resultantes no dependen directamente de cómo se calculen estas energías. La responsabilidad de determinar sus valores recae en el modelo, y su forma específica dependerá de la arquitectura empleada.

El parámetro $\beta$ tampoco esta determinado, pero es posible re-parametrizarlo en términos de un parámetro cuyo significado se conoce, la **temperatura**. La temperatura es un concepto termodinámico conocido, y como hiper-parámetro su utilidad es clara. Si escribimos $\beta$ en términos de la temperatura $T$ como: 

$$
\beta = \frac{1}{T} , \quad T>0
$$

Variando los valores de $T$ podemos alterar la distribución de probabilidades predichas por el modelo. 
- Si $T$ disminuye entonces el modelo se enfría y favorece a los estados con mayor certeza. 
- Si $T$ aumenta entonces el modelo se calienta y la distribución predicha se vuelve uniforme, aumentando su incertidumbre. 

**Hinton, G., Vinyals, O., & Dean, J. (2015).** Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_. [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)