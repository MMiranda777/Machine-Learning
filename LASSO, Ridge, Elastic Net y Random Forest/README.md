# [LASSO, Ridge, Elastic Net y Random Forest](https://github.com/MMiranda777/Machine-Learning/tree/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest)
_**Model Selection and Regularization**_

<img src="Media/rf1.png" width="50%" style="display: block; margin: auto;" /><img src="Media/rf2.png" width="50%" style="display: block; margin: auto;" />

El objetivo de esta entrega es ajustar modelos regularizados como **LASSO, Ridge, Elastic Net** y un modelo de **Randomforest**, posteriormente calcular sus errores de _train_ y _test_ por medio de _cross-validation_ y finalmente analizar y comparar los mejores modelos.

La base de datos **Boston** está incluida en la paqueteria `MASS` en `RStudio` y contiene datos acerca de los valores de las viviendas en los suburbios de Boston (*506* rows x *14* columns), se tomó la columna **`crim`** como variable respuesta; está columna represneta la tasa de crimen per capita por ciudad.

> _**NOTA**_ : Todas las especificaciones relacionadas al código y a los criterios bajo los que se escogieron los mejores modelos vienen explcados a detalle en el documento [`ModelosRegularizados & RF.pdf`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/ModelosRegularizados%20%26%20RF.pdf)

## - Programación:

Para este trabajo se ocupó la siguiente función del la paquetería  _`glmnet`_ en `RStudio`:

|   Función  |                                                                              Descripción                                                                             |
|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `prcomp()` |   Realiza **PCA** sobre los datos y es posible extraer la _proportion  of variance explained_ (PVE) de cada componente así como sus eigenvalores y eigenvectores.    |

El código completo puede consultarse en el archivo [`MR & RF.R`](https://github.com/MMiranda777/Machine-Learning/blob/main/LASSO%2C%20Ridge%2C%20Elastic%20Net%20y%20Random%20Forest/MR%20%26%20RF.R)
## - Resultados:

















:blue_book: [Regresar a Machine-Learning portafolio](https://github.com/MMiranda777/Machine-Learning)
