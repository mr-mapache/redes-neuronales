
Un **modelo** es, en esencia, una simplificación de un dominio. Proporciona una interpretación de la realidad al abstraer los aspectos relevantes para resolver un problema. En el contexto del aprendizaje automático, un modelo se representa matemáticamente y se emplea para aprender patrones y hacer predicciones a partir de datos.

Un modelo debe aprender de datos, los cuales se organizan en un **dataset** (o conjunto de datos). Un dataset es una muestra tomada de una **población**. Este conjunto representa observaciones del mundo real, estructuradas de forma que puedan ser procesadas por un modelo.

Definimos al **espacio resultados** $\Omega$ como al conjunto de todos los posibles resultados de un modelo. Dado un modelo nos interesa la ocurrencia de determinados **eventos**, es decir subconjuntos $A$ de $\Omega$ que agrupan uno o más resultados posibles del experimento, y definimos al espacio de eventos $\mathcal{F}$ como al conjunto de todos los posibles eventos sobre el espacio $\Omega$. Una distribución de probabilidad es una función matemática que asigna a cada evento $A$ un numero $P(A)\in [0,1]$ tal que $P(\Omega)=1$.  
