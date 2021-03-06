 <a href="https://www.linkedin.com/in/melissamirandap/">
 <img src="https://img.shields.io/badge/Linked-in-blue">

# [LASSO, Ridge, Elastic Net y Random Forest](https://github.com/MMiranda777/Machine-Learning/tree/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest)
_**Model Selection and Regularization**_

<img src="Media/rf1.png" width="50%" style="display: block; margin: auto;" /><img src="Media/rf2.png" width="50%" style="display: block; margin: auto;" />

El objetivo de esta entrega es ajustar modelos regularizados como **LASSO, Ridge, Elastic Net** y un modelo de **Random Forest**, posteriormente calcular sus errores de _train_ y _test_ por medio de _cross-validation_ y finalmente analizar y comparar los mejores modelos.

La base de datos **Boston** está incluida en la paqueteria `MASS` en `RStudio` y contiene datos acerca de los valores de las viviendas en los suburbios de Boston (*506* rows x *14* columns), se tomó la columna **`crim`** como variable respuesta; está columna represneta la tasa de crimen per capita por ciudad.

La base de datos **Riboflavin** está incluida en la paqueteria `hdi` en `RStudio` y contiene datos de producción de riboflavina por Bacillus subtilis que contiene *n = 71* observaciones de *p = 4088* predictores (expresiones genéticas) y una respuesta unidimensional (producción de riboflavina). 

> _**NOTA**_ : Todas las especificaciones relacionadas al código y a los criterios bajo los que se escogieron los mejores modelos vienen explcados a detalle en el documento [`ModelosRegularizados & RF.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/ModelosRegularizados%20%26%20RF.pdf)

## - Programación:

Para este trabajo se ocupó la siguientes funciones de las paqueterías  _`glmnet`_ y `randomForest` en `RStudio`:

|   Función  |                                                                              Descripción                                                                             |
|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `cv.glmnet()` |   Ajusta un modelo lineal generalizado mediante la penalización de máxima verosimilitud, la vía de regularización se calcula de acuerdo a la penalización del del modelo (α)  en una malla de valores de regularización para _lambda_. Y `cv` se refiere a que hace el _k-folds cross-validation_ para el modelo.   |
| `randomForest()` |   Implementa el algoritmo de Random Forest de Breiman (basado en el código original de Fortran de Breiman y Cutler) para clasificación y regresión. También se puede utilizar en modo no supervisado para evaluar proximidades entre puntos de datos.    |
> *Nota*: Para los modelos de LASSO y Ridge el valor de α es 1 y 0 respectivamente, mientras que para Elastic Net 0≤α≤1.

El código completo puede consultarse en el archivo [`MR & RF.R`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/MR%20%26%20RF.R)

## - Resultados:
### LASSO, Ridge y Elastic Net:
**Riboflabin**

Recordemos que para un modelo elegido con Ridge este mantendrá la cantidad de variables originales y sólo se buscará encontrar el valor de lambda que minimice el error cuadrático medio (**ECM/MSE**). Para LASSO los modelos sí irán viendo una reducción en variables y se buscará la lambda que minimice el ECM como en el caso de Ridge. Para Elastic Net se tendrá una idea análoga a LASSO.

> Recordemos que _training error rate_ se refiere al error de entrenamiento (**tasa de error aparente**) y que _test error rate_ se refiere al error de predicción validado (**tasa de error no aparente**) (Repeated training/test o utilizando cross-validation con más de una repetición).

<img src="Media/im1.png" width="85%" style="display: block; margin: auto;" />

> **β**: Se refiere al número de variables que toma el modelo.

El criterio para escoger todos los modelos fue tomar `lambda.min`. Para Ridge se utilizó porque era la que minimizaba el ECM, para LASSO y Elastic Net se utilizó porque minimizaba el ECM y porque tomaba en cuenta más variables (**β**) que si se utilizaba `lambda.1se`. Buscamos que los últimos dos modelos tomaran en cuenta más variables ya que la base original cuenta con miles de ellas y a nivel de decenas no cambia mucho el costo computacional.

### Random Forest:
**Boston**

Para `randomForest` se usaron *13* modelos donde cada uno difiere por el valor de `mtry`. De esos *13* modelos se obtuvo la siguiente gráfica: 

<img src="Media/im2.png" width="70%" style="display: block; margin: auto;" />

> `mtry`: Se refiere al número de variables muestreadas aleatoriamente como candidatas en cada división. Los valores predeterminados son diferentes para el método de clasificación (*sqrt(p)* donde *p* es el número de variables en *x*) y regresión (*p/3*).

Se comprararon los _MSE aparentes_ de cada uno de los modelos con los _MSE OOB_ (observados) . Se escogió el modelo que corresponde al valor de *`mtry`=7* con las siguientes especificaciones: 

|     Modelo    | Tr error (AP) | Tr/Test error (NoAP) | `mtry` | `ntree` |
|:-------------:|:-------------:|:--------------------:|:------:|:-------:|
| RandomForest7 |     0.0391    |        0.2355        |    7   |   500   |

> `ntree`: Número de árboles que crecen.







:blue_book: [Regresar a Machine-Learning portafolio](https://github.com/MMiranda777/Machine-Learning)
