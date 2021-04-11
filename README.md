<img src="PCA & Cluster Analysis/Media/clusteri.jpg" width="50%" style="display: block; margin: auto;" /><img src="PCA & Cluster Analysis/Media/clusteri2.jpg" width="50%" style="display: block; margin: auto;" />

## [PCA & Cluster Analysis](https://github.com/MMiranda777/Machine-Learning/tree/main/PCA%20%26%20Cluster%20Analysis)
### *Unsupervised Learning*

El objetivo de esta entrega es poner en práctica algoritmos de **aprendizaje no supervisado** en diversas bases de datos e identificar sus características y diferencias.

  1. `(Ej 8 y 9)` La base **USArrest** está incluida en las bases de datos de _`RStudio`_, esta base contiene estadísticas de los arrestos por cada 100,000 habitantes por los cargos de asalto, homicidio y violación en cada uno de los _50_ estados de EE.UU en 1973. También incluye el porcentaje de la población que vive en áreas urbanas.
  2. `(Ej 10)` Muestra normal aleatoria de _60_ observaciones clasificadas en _3_ grupos (_20_ obs. c/u).
  3. `(Ej 11)` La base [**Ch10Ex11.csv**](https://github.com/MMiranda777/Machine-Learning/blob/main/PCA%20%26%20Cluster%20Analysis/Ch10Ex11.csv) consiste en una muestra de _40_ pañuelos con muestras de _1,000_ genes. Los primeros _20_ pañuelos son de personas saludables mientras que los últimos _20_ son de personas enfermas.

> _**NOTA**_ : Los ejercicios desarrollados se refieren a los ejercicios _8-11_ del capítulo _10 Unsupervized Learning, de James, G., Witten, D., Hastie, T., Tibshirani, R. (2013). An Introduction to Statistical Learning. With applications in R, Springer, ISL_. Todas las especificaciones relacionadas al código y los resultados vienen explicados a detalle en el documento [`PCA&ClusterAnalysis.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/PCA%20%26%20Cluster%20Analysis/PCA%26ClusterAnalysis.pdf)

- Programación:

Para este trabajo se ocuparon las siguientes funciones de la paquetería _`stats`_:

|   Función  |                                                                              Descripción                                                                             |
|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `prcomp()` |   Realiza **PCA** sobre los datos y es posible extraer la _proportion  of variance explained_ (PVE) de cada componente así como sus eigenvalores y eigenvectores.    |
| `hclust()` | Realiza un análisis de **Hierarchical Clustering** sobre los datos  y es posible escoger el método de aglomeración  `(Complete linkage, Euclidian distance, etc...)` |
| `cutree()` |                          Corta árboles de decisión (un _hclust_ por ejemplo) especificando el número de grupos (k) o la altura deseada (h).                          |
| `kmeans()` |                                                  Realiza clusters por el método de **K-means** especificando la _k_                                                  |

El código completo puede consultarse en el archivo [`PCA&ClusterAnalysis_code.Rmd`](https://github.com/MMiranda777/Machine-Learning/blob/main/PCA%20%26%20Cluster%20Analysis/PCA%26ClusterAnalysis_code.Rmd) .

- Resultados: 















# Machine-Learning

En este repositorio comparto todos los trabajos que he desarrollado en el área de Machine Learning. A continuación se presenta un breve resumen de cada uno de ellos.

<img src="Deep Neural Network (DNN)/Media/dnn.jpg" width="50%" style="display: block; margin: auto;" /><img src="Deep Neural Network (DNN)/Media/dnn.jpg" width="50%" style="display: block; margin: auto;" />

## [Deep Neural Network (DNN)](https://github.com/MMiranda777/Machine-Learning/tree/main/Deep%20Neural%20Network%20(DNN))

Este proyecto tiene como objetivo entrenar una **DNN** y una **regresión logística no regularizada**, para clasificar un subconjunto de observaciones de la base **MNIST** con un error de prueba lo más bajo posible.

La base de datos **MNIST** es una base que contiene dígitos escritos a mano, cada dígito está contenido en una imagen de _28x28_ pixeles donde cada pixel representa un número en la escala de grises. La base original contiene _60,000_ imagenes de entrenamiento (_train_) y _10,000_ imágenes de prueba (*test*) sin embargo, para este trabajo solo se usó la parte de _train_ y se partió en tres subconjuntos `MNISTtrain.csv`, `MNISTtest.csv` y `MNISTvalidate.csv` con _40,000_, _9,000_ y _11,000_ registros respectivamente, donde el archivo `MNISTvalidate.csv` solo contiene a las variables predictoras (`x`'s) y no a la variable clase (`y`); esto para posteriormente poder evaluar el poder predictivo de la red neuronal. `(Estas bases se encuentran en el archivo`[`Bases.zip`](https://github.com/MMiranda777/Machine-Learning/blob/main/Deep%20Neural%20Network%20(DNN)/Bases.zip)`)`

> _**NOTA**_ : Todas las especificaciones relacionadas al código, a los criterios bajo los que se modificaron los parámetros y el cómo se decidió cuál era el mejor modelo vienen explicadas a detalle en el documento [`Resultados_MNIST_DNN.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/Deep%20Neural%20Network%20(DNN)/Resultados_MNIST_DNN.pdf)

- Programación:

Para la parte de la programación, se ocupó la paqueteria _`h2o`_ en _`RStudio`_ para generar mallas aleaterias de DNN que tuvieran entre _10_ y _20_ modelos cada una y así poder comparar los modelos y probar ajustando parámetros, como el número de épocas, nodos o la _n_ para _cross validation_, en la siguiente malla para tratar de bajar el error.

>_`h2o`_ es una plataforma abierta que ofrece implementaciones paralelas de diversos algoritmos de aprendizaje supervisado y no supervisado para ML como: Generalized Linear Models (GLM), Gradient Boosting Machines (including XGBoost), Random Forests, Deep Neural Networks (Deep Learning), Stacked Ensembles, Naive Bayes, Generalized Additive Models (GAM), Cox Proportional Hazards, K-Means, PCA, etc.

A continuación se presenta un ejemplo de una de las mallas que se hicieron:

<img src="Deep Neural Network (DNN)/Media/im2.png" width="50%" style="display: block; margin: auto;" />

El código completo puede consultarse en el archivo [`H2O_DNN_MNIST.R`](https://github.com/MMiranda777/Machine-Learning/blob/main/Deep%20Neural%20Network%20(DNN)/H2O_DNN_MNIST.R)

- Resultados:

Hasta aquí solo se habián evaluando modelos con las bases de  `MNISTtrain.csv` y `MNISTtest.csv`, pero se necesitaba decidir el modelo final para probar la base  `MNISTvalidate.csv`. A continuación se presenta una imagen que compara los mejores _8_ modelos obtenidos de un total de _15_ mallas aleatorias (_240_ modelos):

<img src="Deep Neural Network (DNN)/Media/im1.png" width="50%" style="display: block; margin: auto;" />

Finalmente, nuestro criterio para escoger el modelo que haría las predicciones fue tomar los que tuvieran los logloss más bajos y que la diferencia entre el logloss de train contra el de test fuese mínima, bajo estos criterios, estas son las especificaciones del modelo final escogido para probar la base `MNISTvalidate.csv`:

<img src="Deep Neural Network (DNN)/Media/im3.png" width="50%" style="display: block; margin: auto;" />

Al probar esta red neuronal con datos _'nuevos'_ se obtuvieron los siguientes errores:

**Error DNN %**
| train | test |  c-v | valid |
|:-----:|:----:|:----:|-------|
| 1.58  | 2.79 | 3.13 | 2.72  |

Lo que se traduce a un error de clasificación del **2.72%**

- Logros:

En la entrega de este proyecto participaron un total de _6_ equipos donde obtubimos el **primer lugar** en la búsqueda del mejor modelo de DNN, seguido por un modelo con un error de clasificación del 2.97%

## Regresión logística no regularizada 

> Coming soon...

--------------------------------------------------------------------------------------




























> Gracias por leer c: 
