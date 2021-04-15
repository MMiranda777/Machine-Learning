# [LASSO, Ridge, Elastic Net y Random Forest](https://github.com/MMiranda777/Machine-Learning/tree/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest)
_**Model Selection and Regularization**_

<img src="Media/rf1.png" width="50%" style="display: block; margin: auto;" /><img src="Media/rf2.png" width="50%" style="display: block; margin: auto;" />

El objetivo de esta entrega es ajustar modelos regularizados como **LASSO, Ridge, Elastic Net** y un modelo de **Randomforest**, posteriormente calcular sus errores de _train_ y _test_ por medio de _cross-validation_ y finalmente analizar y comparar los mejores modelos.

La base de datos **Boston** está incluida en la paqueteria `MASS` en `RStudio` y contiene datos acerca de los valores de las viviendas en los suburbios de Boston (*506* rows x *14* columns), se tomó la columna **`crim`** como variable respuesta; está columna represneta la tasa de crimen per capita por ciudad.

La base de datos **Riboflavin** está incluida en la paqueteria `hdi` en `RStudio` y contiene datos de producción de riboflavina por Bacillus subtilis que contiene *n = 71* observaciones de *p = 4088* predictores (expresiones genéticas) y una respuesta unidimensional (producción de riboflavina). 

> _**NOTA**_ : Todas las especificaciones relacionadas al código y a los criterios bajo los que se escogieron los mejores modelos vienen explcados a detalle en el documento [`ModelosRegularizados & RF.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/ModelosRegularizados%20%26%20RF.pdf)

## - Programación:

Para este trabajo se ocupó la siguientes funciones de las paqueterías  _`glmnet`_ y `randomForest` en `RStudio`:

|   Función  |                                                                              Descripción                                                                             |
|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `cv.glmnet()` |   Ajusta un modelo lineal generalizado mediante la penalización de máxima verosimilitud, la vía de regularización se calcula de acuerdo a la penalización del del modelo (α)  en una malla de valores de regularización para _lambda_. Y `cv` se refiere a que hace el _k-folds cross-validation_ para el modelo.   |
| `randomForest()` |   Implementa el algoritmo de Random Forest de Breiman (basado en el código original de Fortran de Breiman y Cutler) para clasificación y regresión. También se puede utilizar en modo no supervisado para evaluar proximidades entre puntos de datos.    |
> *Nota*: Para los modelos de LASSO y Ridge el valor de α es 1 y 0 respectivamente, mientras que para Elasticnet 0≤α≤1.

El código completo puede consultarse en el archivo [`MR & RF.R`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/MR%20%26%20RF.R)

## - Resultados:

#### LASSO, Ridge, Elastic Net
Recordemos que para un modelo elegido con ridge este mantendrá la cantidad de variables originales y sólo se buscará encontrar el valor de lambda que minimice el error cuadrático medio (**ECM/MSE**). Para lasso los modelos s´ı ir´an viendo una
reducci´on en variables y se buscar´a la lambda que minimice el ECM como en el caso de ridge. Para elastic net se
tendr´a una idea an´aloga a lasso.

El metodo usado fue trainig/test para la obtención de las tasas no aparentes con 500 repeticiones.
Nuestros errores resultantes fueron.
## Tr error (Ap) Tr/test 500 (NoAp)














:blue_book: [Regresar a Machine-Learning portafolio](https://github.com/MMiranda777/Machine-Learning)
