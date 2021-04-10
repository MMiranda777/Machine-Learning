# Machine-Learning

En este repositorio comparto todos los trabajos que he desarrollado en el área de Machine Learning. A continuación se presenta un breve resumen de cada uno de ellos.

<img src="Deep Neural Network (DNN)/Media/dnn.jpg" width="50%" style="display: block; margin: auto;" /><img src="Deep Neural Network (DNN)/Media/dnn.jpg" width="50%" style="display: block; margin: auto;" />

## [Deep Neural Network (DNN)](https://github.com/MMiranda777/Machine-Learning/tree/main/Deep%20Neural%20Network%20(DNN))

Este proyecto tiene como objetivo entrenar una **DNN** y una **regresión logística no regularizada**, para clasificar un subconjunto de observaciones de la base **MNIST** con un error de prueba lo más bajo posible.

La base de datos **MNIST** es una base que contiene dígitos escritos a mano, cada dígito está contenido en una imagen de _28x28_ pixeles donde cada pixel representa un número en la escala de grises. La base original contiene _60,000_ imagenes de entrenamiento (_train_) y _10,000_ imágenes de prueba (*test*) sin embargo, para este trabajo solo se usó la parte de _train_ y se partió en tres subconjuntos `MNISTtrain.csv`, `MNISTtest.csv` y `MNISTvalidate.csv` con _40,000_, _9,000_ y _11,000_ registros respectivamente, donde el archivo `MNISTvalidate.csv` solo contiene a las variables predictoras (`x`'s) y no a la variable clase (`y`); esto para posteriormente poder evaluar el poder predictivo de la red neuronal. `(Estas bases se encuentran en el archivo `[`Bases.zip`](https://github.com/MMiranda777/Machine-Learning/blob/main/Deep%20Neural%20Network%20(DNN)/Bases.zip)`)`

> _**NOTA**_ : Todas las especificaciones relacionadas al código, a los criterios bajo los que se modificaron los parámetros y el cómo se decidió cuál era el mejor modelo vienen explicadas a detalle en el documento [`Resultados_MNIST_DNN.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/Deep%20Neural%20Network%20(DNN)/Resultados_MNIST_DNN.pdf)

- Programación:

Para la parte de la programación, se ocupó la paqueteria _`h2o`_ en _`RStudio`_ para generar mallas aleaterias de DNN que tuvieran entre _10_ y _20_ modelos cada una y así poder comparar los modelos y probar ajustando parámetros, como el número de epocas o nodos, en la siguiente malla para tratar de bajar el error.

>_`h2o`_ es una plataforma abierta que ofrece implementaciones paralelas de diversos algoritmos de aprendizaje supervisado y no supervisado para ML como: Generalized Linear Models (GLM), Gradient Boosting Machines (including XGBoost), Random Forests, Deep Neural Networks (Deep Learning), Stacked Ensembles, Naive Bayes, Generalized Additive Models (GAM), Cox Proportional Hazards, K-Means, PCA, etc.

A continuación se presenta un ejemplo de una de las mallas que se hicieron:

<img src="Deep Neural Network (DNN)/Media/im2.png" width="50%" style="display: block; margin: auto;" />

- Resultados:

Hasta aquí solo se habián evaluando modelos con las bases de  `MNISTtrain.csv` y `MNISTtest.csv`, pero se necesitaba decidir el modelo final para probar la base  `MNISTvalidate.csv`. A continuación se presenta una imagen que compara los mejores _8_ modelos obtenidos de un total de _15_ mallas aleatorias (_240_ modelos):

<img src="Deep Neural Network (DNN)/Media/im1.png" width="50%" style="display: block; margin: auto;" />

Finalmente, nuestro criterio para escoger el modelo que haría las predicciones fue tomar los que tuvieran los logloss más bajos y que la diferencia entre el logloss de train contra el de test fuese mínima, bajo estos criterios, estas son las especificaciones del modelo final escogido para probar la base `MNISTvalidate.csv` :


<img src="Deep Neural Network (DNN)/Media/im3.png" width="50%" style="display: block; margin: auto;" />

- Logros:


## Regresión logística no regularizada 



alt + 96
