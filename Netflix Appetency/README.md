 <a href="https://www.linkedin.com/in/melissamirandap/">
 <img src="https://img.shields.io/badge/Linked-in-blue">

# [Netflix Appetency](https://github.com/MMiranda777/Machine-Learning/tree/main/Netflix%20Appetency)

<img src="Media/Netflix_logo.png" width="50%" style="display: block; margin: auto;" /><img src="Media/Netflix_logo.png" width="50%" style="display: block; margin: auto;" />

 Este proyecto tiene como objetivo predecir predecir si un usuario se suscribirá al final del mes de prueba de acuerdo con sus interacciones en la plataforma y, en caso de que no se suscriba, encontrar las posibles razones y tomar acciones para completar la mayor cantidad de suscripciones al final del mes.
 
Para lograrlo, se entrenaran modelos de aprendizaje supervizado, para tratar de predecir si se suscribirá o no, y no supervizado, para obtener información extra de los datos.
 
 ## Sobre los datos
 
 La base está conformada por _100,000_ datos de _509_ variables. Se encuentra segmentada en _**train**_ y _**test**_ con 70,000_ y _30,000_ datos respectivamente.
 
Los datos fueron extraídos de **Kaggle** y fueron proporcionados por Netflix, provocando que, por confidencialidad, no se publicara con exactitud la descripción de
las variables “feature” (que conforman 507 columnas de las 509 de la base). Las dos columnas restantes son ‘id’ y ‘target’.
 
id: muestra el número de identificación del usuario;
target: muestra un 1 si el usuario se suscribió a Netflix y 0 si no.
 
Para los fines de este proyecto se trabajó únicamente con los datos de la base train para poder evaluar la capacidad de los modelos, ya que la base test no contaba
con valores para la variable respuesta (’target’) y no era posible evaluar el poder predictivo de los modelos.
