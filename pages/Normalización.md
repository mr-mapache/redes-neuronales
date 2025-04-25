
En estadística, una covariable es una variable que posiblemente predice el resultado bajo estudio. Pueden o no ser de importancia en el mismo. En las redes neuronales, las covariables son los covectores de entrada que se utilizan para predecir la variable de salida.

Un problema común en machine learning es el cambio de la distribución de las covariables en el conjunto de entrenamiento y las covariables con las que el modelo es puesto a prueba. Este problema es conocido como "covariate shift".

En las redes neuronales profundas, este problema se da no solo a la diferencia entre los conjuntos de entrenamiento y prueba, sino también que también se da entre las capas de la red. Este problema es conocido como "internal covariate shift". Las covariables en las capas intermedias de la red neuronal cambian su distribución a medida que se actualizan los pesos de la red. Esto hace que el entrenamiento sea mas lento y dificil.

Los circulos corresponden a las neuronas o covariables. Las flechas corresponden a los pesos de la red. Si cambiamos los pesos de una capa, la distribución de las covariables en la siguiente capa cambia, por lo que en cada iteración del entrenamiento, la siguiente capa se encuentra con una distribución diferente de covariables a las que aprendió en la iteración anterior.

Esto causa dos problemas:

- La red se entrena mas lentamente.

- La red es inestable durante el entrenamiento.