######################################
############### DNN ##################
######################################

library(h2o)
#h2o.removeAll()
#h2o.getVersion()
h2o.init(nthreads = -1)  #internet connection required.

### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])


### Grid ####

(hidden_opt = lapply(1:100, function(x)10+sample(200,sample(4), replace=TRUE)))
(l1_opt = seq(1e-5,1e-3,1e-5))
(hyper_params <- list(hidden = hidden_opt, l1 = l1_opt))
search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 20, 
                       seed=915)  



system.time(
  model_gridz <- h2o.grid("deeplearning",
                          grid_id = "mygrid",
                          hyper_params = hyper_params,
                          search_criteria = search_criteria,
                          x = x,
                          y = y,
                          distribution = "multinomial",
                          training_frame = train,
                          validation_frame = test,
                          score_interval = 2,
                          epochs = 100,
                          stopping_rounds = 3,
                          stopping_tolerance = 0.05,
                          stopping_metric = "misclassification"))


#search_criteria:El default es 'Cartesian' y cubre todo el espacio de combinaciones de parámetros
#score_interval(5):Tiempo que le das para generar cada predicción...Scoring is also called prediction, and is the process of generating values based on a trained machine learning model, given some new input data.
#epochs(10): son como 'capas de interación', si pones pocas no ajusta, y muchas sobreajustan
#stopping_rounds(5): depende de stopping_metric, y se detiene si despues de k iteraciones el promedio movil de stopping metric no mejora para k (?, algo así)
#stopping_tolerance(0): Toleracia relativa (detener si la mejora relativa no es al menos k)
#stopping_metric(AUTO): métrica que usa para detención temprana
#AUTO:logloss for classification, deviance for regression and anonomaly_score for Isolation Forest).


#user  system elapsed 
#4.75    1.52 3575.60  (1hr)


summary(model_gridz, show_stack_traces = TRUE)

#H2O Grid Details
#
# Grid ID: mygrid 
#Used hyper parameters: 
#  -  hidden 
#-  l1 
#Number of models: 20 
#Number of failed models: 0 

#Hyper-Parameter Search Summary: ordered by increasing logloss
##################hidden     l1       model_ids             logloss
#1       [141, 193, 185] 1.3E-4  mygrid_model_3 0.08985901284393313
#2   [139, 194, 82, 207] 4.0E-4  mygrid_model_1 0.09319661374010647
#3        [135, 170, 99] 1.9E-4 mygrid_model_17 0.09997283684198391
#4  [207, 179, 127, 115] 2.3E-4 mygrid_model_18  0.1009363933272627
#5        [135, 170, 99] 7.0E-5 mygrid_model_10 0.10772314315492386
#6          [75, 24, 58] 7.1E-4  mygrid_model_9  0.1095860464732132
#7    [42, 24, 159, 194] 4.9E-4  mygrid_model_7 0.11754347173701399
#8            [124, 174] 8.4E-4 mygrid_model_14 0.11818072942964687
#9    [189, 94, 99, 139] 1.5E-4  mygrid_model_5 0.11889266653241276
#10  [172, 120, 102, 65] 8.0E-4  mygrid_model_4 0.11930311971402195
#11      [203, 105, 118] 6.4E-4 mygrid_model_12 0.12103671274081035
#12 [207, 179, 127, 115] 6.3E-4  mygrid_model_8 0.12150241758039672
#13            [135, 95] 8.2E-4  mygrid_model_2 0.12172103854730192
#14 [207, 179, 127, 115] 9.8E-4 mygrid_model_11 0.12199768995834938
#15        [119, 96, 28] 1.5E-4 mygrid_model_20 0.12510123076197588
#16       [33, 129, 206]  0.001  mygrid_model_6 0.12794407971093696
#17                 [87] 6.2E-4 mygrid_model_13 0.14226215098826886
#18                [175] 5.0E-4 mygrid_model_19  0.1506714275672102
#19  [15, 160, 192, 135] 8.2E-4 mygrid_model_15 0.15844318094826731
#20                 [13] 2.9E-4 mygrid_model_16  0.3494162762317335
#H2O Grid Summary
#

model_gridz

tabb=matrix(NA, nrow=20, ncol=12)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1","InDropoutRat",
                 "loglss_trn","loglss_tst","loglss_cv",
                 "Err_trn","Err_tst","Err_cv")
tabb 

models=list()
mperftrn=list()
mperftst=list()

models <- lapply(model_gridz@model_ids, function(id){ h2o.getModel(id)})
model_gridz@model_ids

for(i in 1:20){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}

#dev.off()
#par(mfrow = c(3,4), mai = c(.5,.5,.5,.5))
# Plot the scoring history over time for the first 10 models
#for(i in 1:10){
# plot(models[[i]],metric = "classification_error",cex=.7)
#}

for(i in 1:20){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,7]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,8]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,10]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,11]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb

#### #model             RunTimeMins layers            epochs l1        InDropoutRat loglss_trn loglss_tst loglss_cv Err_trn  Err_tst 
#[1,] "mygrid_model_3"  "2.27"      "141-193-185"     "100"  "0.00013" "0"          "0.028"    "0.0899"   NA        "0.009"  "0.0246"
#[2,] "mygrid_model_1"  "3.48"      "139-194-82-207"  "100"  "4e-04"   "0"          "0.0479"   "0.0932"   NA        "0.0143" "0.0276"
#[3,] "mygrid_model_17" "3.59"      "135-170-99"      "100"  "0.00019" "0"          "0.0505"   "0.1"      NA        "0.0158" "0.0278"
#[4,] "mygrid_model_18" "5.5"       "207-179-127-115" "100"  "0.00023" "0"          "0.0455"   "0.1009"   NA        "0.0147" "0.0296"
#[5,] "mygrid_model_10" "3.21"      "135-170-99"      "100"  "7e-05"   "0"          "0.0245"   "0.1077"   NA        "0.0072" "0.0264"
#[6,] "mygrid_model_9"  "2.16"      "75-24-58"        "100"  "0.00071" "0"          "0.0718"   "0.1096"   NA        "0.021"  "0.0309"
#[7,] "mygrid_model_7"  "3.11"      "42-24-159-194"   "100"  "0.00049" "0"          "0.0627"   "0.1175"   NA        "0.0192" "0.033" 
#[8,] "mygrid_model_14" "3.13"      "124-174"         "100"  "0.00084" "0"          "0.0888"   "0.1182"   NA        "0.025"  "0.0344"
#[9,] "mygrid_model_5"  "2.82"      "189-94-99-139"   "100"  "0.00015" "0"          "0.048"    "0.1189"   NA        "0.0157" "0.0292"
#[10,] "mygrid_model_4"  "3.02"      "172-120-102-65"  "100"  "8e-04"   "0"          "0.085"    "0.1193"   NA        "0.026"  "0.0333"
#[11,] "mygrid_model_12" "4.02"      "203-105-118"     "100"  "0.00064" "0"          "0.0869"   "0.121"    NA        "0.0256" "0.0353"
#[12,] "mygrid_model_8"  "4.41"      "207-179-127-115" "100"  "0.00063" "0"          "0.0866"   "0.1215"   NA        "0.0268" "0.0368"
#[13,] "mygrid_model_2"  "1.92"      "135-95"          "100"  "0.00082" "0"          "0.0956"   "0.1217"   NA        "0.0267" "0.0331"
#[14,] "mygrid_model_11" "6.68"      "207-179-127-115" "100"  "0.00098" "0"          "0.0827"   "0.122"    NA        "0.0244" "0.0339"
#[15,] "mygrid_model_20" "2.1"       "119-96-28"       "100"  "0.00015" "0"          "0.0507"   "0.1251"   NA        "0.0158" "0.0334"
#[16,] "mygrid_model_6"  "1.35"      "33-129-206"      "100"  "0.001"   "0"          "0.0948"   "0.1279"   NA        "0.0283" "0.0366"
#[17,] "mygrid_model_13" "1.43"      "87"              "100"  "0.00062" "0"          "0.094"    "0.1423"   NA        "0.0269" "0.0403"
#[18,] "mygrid_model_19" "2.34"      "175"             "100"  "5e-04"   "0"          "0.1045"   "0.1507"   NA        "0.0304" "0.039" 
#[19,] "mygrid_model_15" "1.87"      "15-160-192-135"  "100"  "0.00082" "0"          "0.117"    "0.1584"   NA        "0.0358" "0.0463"
#[20,] "mygrid_model_16" "1.09"      "13"              "100"  "0.00029" "0"          "0.203"    "0.3494"   NA        "0.0544" "0.0783"


tabb2 = as.data.frame(tabb)
tabb2
tabb2[,c(2,4:8,10:11)] = apply(tabb2[,c(2,4:8,10:11)],2,as.numeric)
str(tabb2)
dev.off()
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(1:20, cbind(tabb2[,7], tabb2[,8],tabb2[,10], tabb2[,11]),pch=19,cex=.7, 
        col=c("orange", "red","lightblue","blue"),
        type="b",ylab="Errors/logloss", xlab="models")
legend("topleft", legend=c("logloss_train", "logloss_test",
                           "Error_train", "Error_test"), pch=19, cex=.5,
       col=c("orange", "red","lightblue","blue"))


write.csv(tabb2,"~/FAC/8vo/Sem.Est/Proyecto/intento2.csv", row.names = TRUE)

tabb2

################ intento 4 ###############
library(h2o)
#h2o.removeAll()
#h2o.getVersion()
h2o.init(nthreads = -1)  #internet connection required.

### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
#m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
#validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])


### Grid ####

#### Cambios ####

#lo cambiamos a mínimo 120 nodos por capa y máximo 220
#l1 por un cero menos
#agregamos una k y modificamos para que agarre de 3-4 capas


k<-c(3,4)

(hidden_opt = lapply(1:100, function(x)120+sample(100,sample(k,1), replace=TRUE)))
(l1_opt = seq(1e-4,1e-3,1e-5))
(hyper_params <- list(hidden = hidden_opt, l1 = l1_opt))
search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 20, 
                       seed=915)  

#### Cambios 2####

#Regresamos a 2 score interval
# regresamos a 100 epochs
#stopping_metric regresamos


system.time(
  model_gridz <- h2o.grid("deeplearning",
                          grid_id = "mygrid",
                          hyper_params = hyper_params,
                          search_criteria = search_criteria,
                          x = x,
                          y = y,
                          distribution = "multinomial",
                          training_frame = train,
                          validation_frame = test,
                          score_interval = 2,
                          epochs = 100,
                          stopping_rounds = 3,
                          stopping_tolerance = 0.05,
                          stopping_metric = "misclassification"))

summary(model_gridz, show_stack_traces = TRUE)


#user  system elapsed 
#2.36    0.41 3608.17 (1hr)

#Grid ID: mygrid 
#Used hyper parameters: 
#  -  hidden 
#-  l1 
#Number of models: 20 
#Number of failed models: 0 

#Hyper-Parameter Search Summary: ordered by increasing logloss
#hidden     l1       model_ids             logloss
#1  [182, 126, 153, 216] 2.0E-4  mygrid_model_2 0.08863397194889648
#2       [201, 174, 148] 3.0E-4  mygrid_model_4 0.09041704271944948
#3       [162, 142, 168] 2.5E-4 mygrid_model_10 0.09170088254335254
#4  [170, 144, 187, 121] 2.9E-4  mygrid_model_9 0.09781739546563373
#5       [186, 154, 204] 6.3E-4  mygrid_model_7 0.10201445326212513
#6  [170, 122, 184, 144] 5.0E-4 mygrid_model_13 0.10372538717417118
#7       [194, 127, 174] 8.7E-4  mygrid_model_8 0.11083968310703035
#8       [162, 142, 168] 6.7E-4 mygrid_model_17 0.11115972210667659
#9       [155, 203, 140] 8.3E-4  mygrid_model_5 0.11118852446262688
#10 [155, 130, 146, 154] 1.4E-4 mygrid_model_14 0.11189307305985699
#11 [206, 208, 203, 141] 4.2E-4  mygrid_model_1 0.11193178495508659
#12      [216, 165, 184] 1.2E-4  mygrid_model_3 0.11222126650599054
#13      [173, 180, 186] 3.1E-4  mygrid_model_6 0.11237864864010566
#14 [205, 219, 208, 156] 9.5E-4 mygrid_model_19 0.11419852372923366
#15 [160, 165, 145, 148] 7.1E-4 mygrid_model_16 0.11439203717562046
#16      [121, 196, 121] 8.8E-4 mygrid_model_20 0.11450533930557044
#17 [202, 152, 188, 215] 6.2E-4 mygrid_model_12 0.11636262237669404
#18 [165, 125, 218, 177] 8.1E-4 mygrid_model_15 0.11704387701539073
#19      [194, 127, 174] 7.5E-4 mygrid_model_11 0.11911212760983396
#20      [194, 127, 174] 3.2E-4 mygrid_model_18 0.12001535623156744


model_gridz


tabb=matrix(NA, nrow=20, ncol=9)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1",
                 "loglss_trn","loglss_tst",
                 "Err_trn","Err_tst")
tabb 

models=list()
mperftrn=list()
mperftst=list()




models <- lapply(model_gridz@model_ids, function(id){ h2o.getModel(id)})
model_gridz@model_ids

for(i in 1:20){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}


for(i in 1:20){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  # tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,6]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,7]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,8]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,9]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb


##    model             RunTimeMins layers            epochs l1        loglss_trn loglss_tst Err_trn  Err_tst 
#[1,] "mygrid_model_2"  "4.96"      "182-126-153-216" "100"  "2e-04"   "0.0297"   "0.0886"   "0.0098" "0.0246"
#[2,] "mygrid_model_4"  "3.28"      "201-174-148"     "100"  "3e-04"   "0.0503"   "0.0904"   "0.0153" "0.0259"
#[3,] "mygrid_model_10" "4.79"      "162-142-168"     "100"  "0.00025" "0.0392"   "0.0917"   "0.012"  "0.025" 
#[4,] "mygrid_model_9"  "3.91"      "170-144-187-121" "100"  "0.00029" "0.042"    "0.0978"   "0.0123" "0.0279"
#[5,] "mygrid_model_7"  "3.29"      "186-154-204"     "100"  "0.00063" "0.0666"   "0.102"    "0.0202" "0.0302"
#[6,] "mygrid_model_13" "3"         "170-122-184-144" "100"  "5e-04"   "0.0604"   "0.1037"   "0.018"  "0.0293"
#[7,] "mygrid_model_8"  "3.59"      "194-127-174"     "100"  "0.00087" "0.0774"   "0.1108"   "0.0225" "0.0308"
#[8,] "mygrid_model_17" "2.32"      "162-142-168"     "100"  "0.00067" "0.0804"   "0.1112"   "0.0234" "0.0311"
#[9,] "mygrid_model_5"  "2.56"      "155-203-140"     "100"  "0.00083" "0.0798"   "0.1112"   "0.0239" "0.032" 
#[10,] "mygrid_model_14" "1.51"      "155-130-146-154" "100"  "0.00014" "0.0478"   "0.1119"   "0.0156" "0.0304"
#[11,] "mygrid_model_1"  "3.06"      "206-208-203-141" "100"  "0.00042" "0.0707"   "0.1119"   "0.0213" "0.0331"
#[12,] "mygrid_model_3"  "2.32"      "216-165-184"     "100"  "0.00012" "0.0422"   "0.1122"   "0.0136" "0.0296"
#[13,] "mygrid_model_6"  "1.5"       "173-180-186"     "100"  "0.00031" "0.0722"   "0.1124"   "0.0227" "0.0342"
#[14,] "mygrid_model_19" "3.4"       "205-219-208-156" "100"  "0.00095" "0.0828"   "0.1142"   "0.0238" "0.0328"
#[15,] "mygrid_model_16" "3.46"      "160-165-145-148" "100"  "0.00071" "0.0683"   "0.1144"   "0.02"   "0.0319"
#[16,] "mygrid_model_20" "2.77"      "121-196-121"     "100"  "0.00088" "0.0804"   "0.1145"   "0.0236" "0.0326"
#[17,] "mygrid_model_12" "3.22"      "202-152-188-215" "100"  "0.00062" "0.0819"   "0.1164"   "0.0243" "0.034" 
#[18,] "mygrid_model_15" "3.4"       "165-125-218-177" "100"  "0.00081" "0.0731"   "0.117"    "0.022"  "0.034" 
#[19,] "mygrid_model_11" "2.13"      "194-127-174"     "100"  "0.00075" "0.0912"   "0.1191"   "0.0275" "0.0341"
#[20,] "mygrid_model_18" "1.6"       "194-127-174"     "100"  "0.00032" "0.0774"   "0.12"     "0.0245" "0.0362"

tabb2 = as.data.frame(tabb)
tabb2
tabb2[,c(2,4:9)] = apply(tabb2[,c(2,4:9)],2,as.numeric)
str(tabb2)
dev.off()
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(1:20, cbind(tabb2[,6], tabb2[,7],tabb2[,8], tabb2[,9]),pch=19,cex=.7, 
        col=c("orange", "red","lightblue","blue"),
        type="b",ylab="Errors/logloss", xlab="models")
legend("topleft", legend=c("logloss_train", "logloss_test",
                           "Error_train", "Error_test"), pch=19, cex=.5,
       col=c("orange", "red","lightblue","blue"))


write.csv(tabb2,"~/FAC/8vo/Sem.Est/Proyecto/intento4.csv", row.names = TRUE)

tabb2



############# intento 6 ############################
library(h2o)
#h2o.removeAll()
#h2o.getVersion()
h2o.init(nthreads = -1)  #internet connection required.

### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
#m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
#validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])


### Grid ####

#### Cambios ####

#lo cambiamos a mínimo 120 nodos por capa y máximo 220
#lambdas más grandes
#cmbiamos a 3 capas
## cambiamos a 10 modelos



(hidden_opt = lapply(1:100, function(x)120+sample(100,3, replace=TRUE)))
(l1_opt = seq(1e-4,1e-1,1e-3))
(hyper_params <- list(hidden = hidden_opt, l1 = l1_opt))
search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 10, 
                       seed=915)  

#### Cambios 2####

#Regresamos a 2 score interval
# cambiamos a 30 epochs
#stopping_metric regresamos


system.time(
  model_gridz <- h2o.grid("deeplearning",
                          grid_id = "mygrid",
                          hyper_params = hyper_params,
                          search_criteria = search_criteria,
                          x = x,
                          y = y,
                          distribution = "multinomial",
                          training_frame = train,
                          validation_frame = test,
                          score_interval = 2,
                          epochs = 100,
                          stopping_rounds = 3,
                          stopping_tolerance = 0.05,
                          stopping_metric = "misclassification"))

summary(model_gridz, show_stack_traces = TRUE)


#user  system elapsed 
#1.34    0.22  659.74 (11min)


#Grid ID: mygrid 
#Used hyper parameters: 
#  -  hidden 
#-  l1 
#Number of models: 30 
#Number of failed models: 0 

#Hyper-Parameter Search Summary: ordered by increasing logloss
#hidden     l1       model_ids             logloss
#1 [211, 196, 166] 3.2E-4  mygrid_model_6 0.08737384055949264
#2 [213, 170, 209] 2.7E-4  mygrid_model_9 0.09254708711813735
#3 [145, 123, 157] 3.4E-4 mygrid_model_19 0.09297535295568152
#4 [190, 174, 190] 3.1E-4  mygrid_model_2  0.0935688173714483
#5 [135, 204, 174] 1.6E-4 mygrid_model_17 0.09458282231248467

#---
###### hidden     l1       model_ids            logloss
#25 [154, 191, 147] 0.0721 mygrid_model_30  2.368990788804118
#26 [124, 168, 212] 0.0781 mygrid_model_21 2.3876065381230567
#27 [134, 135, 153] 0.0511 mygrid_model_25  2.421816848247989
#28 [127, 174, 160] 0.0451 mygrid_model_24  2.491344294335199
#29 [147, 196, 182] 0.0421 mygrid_model_22  2.511229729185758
#30 [180, 125, 148] 0.0681 mygrid_model_29 2.5517406039470014
#H2O Grid Summary


tabb=matrix(NA, nrow=10, ncol=9)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1",
                 "loglss_trn","loglss_tst",
                 "Err_trn","Err_tst")
tabb 

models=list()
mperftrn=list()
mperftst=list()




models <- lapply(model_gridz@model_ids, function(id){ h2o.getModel(id)})
model_gridz@model_ids

for(i in 1:10){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}


for(i in 1:10){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  # tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,6]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,7]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,8]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,9]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb


######model             RunTimeMins layers        epochs l1        loglss_trn loglss_tst Err_trn  Err_tst 
#[1,] "mygrid_model_6"  "5.29"      "211-196-166" "100"  "0.00032" "0.0498"   "0.0874"   "0.0148" "0.0257"
#[2,] "mygrid_model_9"  "3.18"      "213-170-209" "100"  "0.00027" "0.0512"   "0.0925"   "0.0153" "0.0293"
#[3,] "mygrid_model_19" "2.28"      "145-123-157" "100"  "0.00034" "0.056"    "0.093"    "0.0163" "0.0284"
#[4,] "mygrid_model_2"  "3.7"       "190-174-190" "100"  "0.00031" "0.0526"   "0.0936"   "0.0167" "0.029" 
#[5,] "mygrid_model_17" "2.62"      "135-204-174" "100"  "0.00016" "0.0334"   "0.0946"   "0.0103" "0.0254"
#[6,] "mygrid_model_18" "2.22"      "169-151-172" "100"  "0.00016" "0.0369"   "0.0971"   "0.0115" "0.025" 
#[7,] "mygrid_model_1"  "2.51"      "127-180-199" "100"  "0.00036" "0.05"     "0.0974"   "0.0151" "0.0266"
#[8,] "mygrid_model_14" "2.13"      "128-125-212" "100"  "0.00023" "0.0475"   "0.0987"   "0.0153" "0.0293"
#[9,] "mygrid_model_15" "1.95"      "150-200-124" "100"  "0.00027" "0.0527"   "0.0994"   "0.017"  "0.0284"
#[10,] "mygrid_model_11" "2.43"      "169-151-172" "100"  "0.00021" "0.0472"   "0.1013"   "0.0157" "0.0291"





tabb2 = as.data.frame(tabb)
tabb2
tabb2[,c(2,4:9)] = apply(tabb2[,c(2,4:9)],2,as.numeric)
str(tabb2)
dev.off()
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(1:10, cbind(tabb2[,6], tabb2[,7],tabb2[,8], tabb2[,9]),pch=19,cex=.7, 
        col=c("orange", "red","lightblue","blue"),
        type="b",ylab="Errors/logloss", xlab="models")
legend("topleft", legend=c("logloss_train", "logloss_test",
                           "Error_train", "Error_test"), pch=19, cex=.5,
       col=c("orange", "red","lightblue","blue"))


write.csv(tabb2,"~/FAC/8vo/Sem.Est/Proyecto/intento6.csv", row.names = TRUE)

tabb2

############### intento 8 #####################

library(h2o)
#h2o.removeAll()
#h2o.getVersion()
h2o.init(nthreads = -1)  #internet connection required.

### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
#m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
#validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])



#### Grid ####

(hidden_opt = lapply(1:100, function(x)10+sample(300,sample(5), replace=TRUE)))
(l1_opt = seq(1e-5,1e-3,1e-5))
(epoch_opt=20)
(activation_opt = c("Rectifier", "Rectifier with dropout", "Tanh",
                    "Tanh with dropout"))
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt, epochs=epoch_opt,
                     activation= activation_opt)
search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 20,
                       seed=1)

system.time(
  model_gridz<- h2o.grid("deeplearning",
                         grid_id = "randomg1",
                         hyper_params = hyper_params,
                         search_criteria = search_criteria,
                         x = x,
                         y = y,
                         distribution = "multinomial",
                         training_frame = train,
                         #                 nfolds = 5,
                         seed = 1,
                         score_interval = 2,
                         stopping_rounds = 3,
                         stopping_tolerance = 0.05,
                         stopping_metric = "misclassification"))

#user  system elapsed 
#2.75    0.47 2443.27 (40.72 min)

summary(model_gridz, show_stack_traces = TRUE)


#Grid ID: randomg1 
#Used hyper parameters: 
#  -  activation 
#-  epochs 
#-  hidden 
#-  l1 
#Number of models: 20 
#Number of failed models: 0 

#Hyper-Parameter Search Summary: ordered by increasing logloss
##############activation epochs                    hidden     l1         model_ids              logloss
#1             Rectifier   20.0                 [253, 69] 4.3E-4 randomg1_model_18   0.0584707386131454
#2             Rectifier   20.0                     [309] 2.6E-4  randomg1_model_4 0.059641561849153715
#3             Rectifier   20.0  [128, 303, 190, 204, 46] 5.8E-4 randomg1_model_20  0.07396705170937252
#4             Rectifier   20.0       [206, 131, 252, 15] 7.2E-4 randomg1_model_13  0.08506259174596759
#5  RectifierWithDropout   20.0                 [253, 69] 7.6E-4  randomg1_model_6  0.10237293092634603
#6                  Tanh   20.0                     [132] 2.7E-4  randomg1_model_3  0.12325280855291623
#7  RectifierWithDropout   20.0            [92, 237, 252] 9.6E-4 randomg1_model_17  0.12665426834685134
#8                  Tanh   20.0            [54, 208, 238] 3.7E-4 randomg1_model_15  0.14390094075101564
#9                  Tanh   20.0            [202, 75, 232] 5.7E-4 randomg1_model_14  0.19067006050835594
#10                 Tanh   20.0   [163, 23, 80, 153, 195] 6.7E-4  randomg1_model_9  0.20710059275615914
#11 RectifierWithDropout   20.0   [200, 87, 170, 257, 28] 7.7E-4 randomg1_model_11  0.21923229553480608
#12      TanhWithDropout   20.0           [306, 192, 237] 7.6E-4 randomg1_model_19   0.2274856625781845
#13                 Tanh   20.0   [32, 193, 289, 19, 307] 8.1E-4  randomg1_model_8  0.23629966678240316
#14 RectifierWithDropout   20.0         [44, 176, 92, 26] 9.0E-5  randomg1_model_1  0.23965690017000743
#15                 Tanh   20.0      [235, 277, 241, 213]  0.001 randomg1_model_12  0.28178410559502104
#16                 Tanh   20.0 [127, 131, 283, 186, 103] 7.0E-4 randomg1_model_10   0.2918244295177738
#17      TanhWithDropout   20.0            [92, 237, 252] 9.6E-4  randomg1_model_2   0.3159715628004103
#18      TanhWithDropout   20.0            [117, 82, 194] 5.1E-4  randomg1_model_5  0.35151152745013375
#19                 Tanh   20.0                     [303] 6.3E-4 randomg1_model_16  0.37441226064484384
#20 RectifierWithDropout   20.0    [269, 301, 11, 79, 98] 4.4E-4  randomg1_model_7  0.39929709607358943
#H2O Grid Summary



tabb=matrix(NA, nrow=20, ncol=12)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1","InDropoutRat",
                 "loglss_trn","loglss_tst","loglss_cv",
                 "Err_trn","Err_tst","Err_cv")
tabb 

models=list()
mperftrn=list()
mperftst=list()




models <- lapply(model_gridz@model_ids, function(id){ h2o.getModel(id)})
model_gridz@model_ids

for(i in 1:20){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}


for(i in 1:20){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,7]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,8]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,10]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,11]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb


#####model               RunTimeMins layers                epochs l1        InDropoutRat loglss_trn loglss_tst loglss_cv Err_trn 
#[1,] "randomg1_model_18" "3.77"      "253-69"              "20"   "0.00043" "0"          "0.0547"   "0.0898"   NA        "0.0157"
#[2,] "randomg1_model_4"  "5.33"      "309"                 "20"   "0.00026" "0"          "0.0554"   "0.1079"   NA        "0.0156"
#[3,] "randomg1_model_20" "2.77"      "128-303-190-204-46"  "20"   "0.00058" "0"          "0.0691"   "0.1131"   NA        "0.0205"
#[4,] "randomg1_model_13" "2.24"      "206-131-252-15"      "20"   "0.00072" "0"          "0.0812"   "0.1175"   NA        "0.0239"
#[5,] "randomg1_model_6"  "2.21"      "253-69"              "20"   "0.00076" "0"          "0.0969"   "0.129"    NA        "0.0302"
#[6,] "randomg1_model_3"  "3.01"      "132"                 "20"   "0.00027" "0"          "0.1248"   "0.1606"   NA        "0.0366"
#[7,] "randomg1_model_17" "1.58"      "92-237-252"          "20"   "0.00096" "0"          "0.1186"   "0.1484"   NA        "0.0355"
#[8,] "randomg1_model_15" "1.65"      "54-208-238"          "20"   "0.00037" "0"          "0.1464"   "0.1898"   NA        "0.0418"
#[9,] "randomg1_model_14" "1.83"      "202-75-232"          "20"   "0.00057" "0"          "0.1898"   "0.2188"   NA        "0.0539"
#[10,] "randomg1_model_9"  "1.36"      "163-23-80-153-195"   "20"   "0.00067" "0"          "0.2043"   "0.23"     NA        "0.0585"
#[11,] "randomg1_model_11" "1.7"       "200-87-170-257-28"   "20"   "0.00077" "0"          "0.2065"   "0.2313"   NA        "0.054" 
#[12,] "randomg1_model_19" "1.94"      "306-192-237"         "20"   "0.00076" "0"          "0.2217"   "0.2671"   NA        "0.0645"
#[13,] "randomg1_model_8"  "1.55"      "32-193-289-19-307"   "20"   "0.00081" "0"          "0.2275"   "0.2582"   NA        "0.0649"
#[14,] "randomg1_model_1"  "0.57"      "44-176-92-26"        "20"   "9e-05"   "0"          "0.2335"   "0.2743"   NA        "0.0617"
#[15,] "randomg1_model_12" "2.67"      "235-277-241-213"     "20"   "0.001"   "0"          "0.2768"   "0.2974"   NA        "0.0777"
#[16,] "randomg1_model_10" "0.93"      "127-131-283-186-103" "20"   "7e-04"   "0"          "0.2874"   "0.307"    NA        "0.0836"
#[17,] "randomg1_model_2"  "0.97"      "92-237-252"          "20"   "0.00096" "0"          "0.3077"   "0.3336"   NA        "0.0902"
#[18,] "randomg1_model_5"  "0.6"       "117-82-194"          "20"   "0.00051" "0"          "0.3468"   "0.3692"   NA        "0.095" 
#[19,] "randomg1_model_16" "0.68"      "303"                 "20"   "0.00063" "0"          "0.3661"   "0.4096"   NA        "0.0885"
#[20,] "randomg1_model_7"  "3.29"      "269-301-11-79-98"    "20"   "0.00044" "0"          "0.3896"   "0.4301"   NA        "0.1094"
#Err_tst  Err_cv
#[1,] "0.0264" NA    
#[2,] "0.0309" NA    
#[3,] "0.032"  NA    
#[4,] "0.0337" NA    
#[5,] "0.0371" NA    
#[6,] "0.0477" NA    
#[7,] "0.0432" NA    
#[8,] "0.0554" NA    
#[9,] "0.0621" NA    
#[10,] "0.0652" NA    
#[11,] "0.0581" NA    
#[12,] "0.072"  NA    
#[13,] "0.0743" NA    
#[14,] "0.0691" NA    
#[15,] "0.0844" NA    
#[16,] "0.0909" NA    
#[17,] "0.0971" NA    
#[18,] "0.1007" NA    
#[19,] "0.0921" NA    
#[20,] "0.1196" NA    



############# intento 10 ######################
library(h2o)
#h2o.removeAll()
#h2o.getVersion()
h2o.init(nthreads = -1)  #internet connection required.

### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
#m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
#validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])



(hidden_opt = lapply(1:100, function(x)120+sample(100,sample(3), replace=TRUE)))
(l1_opt = seq(1e-5,1e-3,1e-5))
(hyper_params <- list(hidden = hidden_opt, l1 = l1_opt))
search_criteria = list(strategy = "RandomDiscrete", 
                       max_models = 20, 
                       seed=1)  



system.time(
  model_gridz <- h2o.grid("deeplearning",
                          grid_id = "mygrid",
                          hyper_params = hyper_params,
                          search_criteria = search_criteria,
                          x = x,
                          y = y,
                          distribution = "multinomial",
                          training_frame = train,
                          validation_frame = test,
                          score_interval = 2,
                          seed=1,
                          epochs = 20,
                          stopping_rounds = 3,
                          stopping_tolerance = 0.05,
                          stopping_metric = "misclassification"))

summary(model_gridz, show_stack_traces = TRUE)

#user  system elapsed 
#2.30    0.27 2852.75 (47.54)

#Grid ID: mygrid 
#Used hyper parameters: 
# -  hidden 
#-  l1 
#Number of models: 40 
#Number of failed models: 0 

#Hyper-Parameter Search Summary: ordered by increasing logloss
###############hidden     l1       model_ids             logloss
#1         [132, 133, 185] 3.4E-4 mygrid_model_26 0.09383989737386722
#2 [198, 83, 22, 192, 194] 3.2E-4 mygrid_model_19 0.09729790327766502
#3              [160, 184] 3.8E-4 mygrid_model_22 0.10189053843881818
#4      [63, 181, 71, 132] 5.9E-4 mygrid_model_17 0.10511708579756486
#5         [193, 171, 173] 6.1E-4 mygrid_model_21  0.1055496807820889

#---
##########  hidden     l1       model_ids             logloss
#35 [33, 282, 163, 207] 9.3E-4  mygrid_model_3 0.13548156420061402
#36                [71] 3.4E-4  mygrid_model_6 0.13774397233199764
#37               [182] 3.0E-4 mygrid_model_29 0.13839456326236493
#38               [255] 9.7E-4 mygrid_model_13 0.14230450093589145
#39               [255] 9.8E-4 mygrid_model_20 0.14915937014256722
#40               [189] 3.2E-4 mygrid_model_39 0.15140010474620885
#H2O Grid Summary

tabb=matrix(NA, nrow=20, ncol=9)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1",
                 "loglss_trn","loglss_tst",
                 "Err_trn","Err_tst")
tabb 

models=list()
mperftrn=list()
mperftst=list()




models <- lapply(model_gridz@model_ids, function(id){ h2o.getModel(id)})
model_gridz@model_ids

for(i in 1:20){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}


for(i in 1:20){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  # tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,6]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,7]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,8]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,9]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb

#####model             RunTimeMins layers                epochs l1        loglss_trn loglss_tst Err_trn  Err_tst 
#[1,] "mygrid_model_26" "2.51"      "132-133-185"         "20"   "0.00034" "0.055"    "0.0938"   "0.0163" "0.0272"
#[2,] "mygrid_model_19" "4.45"      "198-83-22-192-194"   "20"   "0.00032" "0.0498"   "0.0973"   "0.0154" "0.0273"
#[3,] "mygrid_model_22" "2.18"      "160-184"             "20"   "0.00038" "0.062"    "0.1019"   "0.0187" "0.03"  
#[4,] "mygrid_model_17" "1.73"      "63-181-71-132"       "20"   "0.00059" "0.0635"   "0.1051"   "0.0192" "0.0298"
#[5,] "mygrid_model_21" "3.41"      "193-171-173"         "20"   "0.00061" "0.0651"   "0.1055"   "0.0196" "0.0296"
#[6,] "mygrid_model_16" "4.03"      "191-177-144-210-250" "20"   "0.00017" "0.0431"   "0.1063"   "0.0137" "0.029" 
#[7,] "mygrid_model_34" "2.23"      "192-150"             "20"   "0.00015" "0.0403"   "0.1069"   "0.0128" "0.0278"
#[8,] "mygrid_model_37" "3.8"       "193-171-173"         "20"   "0.00059" "0.0715"   "0.1083"   "0.0214" "0.0303"
#[9,] "mygrid_model_12" "2.35"      "129-188-63-273-31"   "20"   "0.00084" "0.0702"   "0.1101"   "0.0212" "0.0319"
#[10,] "mygrid_model_7"  "3.33"      "94-281-255-186"      "20"   "0.00056" "0.0614"   "0.1105"   "0.0187" "0.0299"
#[11,] "mygrid_model_2"  "2.14"      "33-282-163-207"      "20"   "0.00038" "0.0551"   "0.1125"   "0.0174" "0.0313"
#[12,] "mygrid_model_11" "2.43"      "184"                 "20"   "0.00052" "0.0792"   "0.1141"   "0.0223" "0.0324"
#[13,] "mygrid_model_4"  "2.18"      "123-310-59"          "20"   "0.00089" "0.0818"   "0.1154"   "0.0234" "0.033" 
#[14,] "mygrid_model_1"  "1.77"      "63-181-71-132"       "20"   "0.00061" "0.0686"   "0.1155"   "0.0216" "0.0322"
#[15,] "mygrid_model_14" "2.15"      "180-245-132-47"      "20"   "0.00015" "0.0447"   "0.1169"   "0.0143" "0.0297"
#[16,] "mygrid_model_8"  "1.92"      "112-36-198-83-280"   "20"   "0.00046" "0.069"    "0.1169"   "0.0204" "0.0332"
#[17,] "mygrid_model_31" "3.06"      "207"                 "20"   "0.00052" "0.0854"   "0.1186"   "0.0234" "0.0317"
#[18,] "mygrid_model_35" "2.16"      "208-172-163"         "20"   "0.00029" "0.0706"   "0.1187"   "0.0221" "0.0344"
#[19,] "mygrid_model_10" "3.94"      "232-60-228-84-42"    "20"   "0.00085" "0.0818"   "0.1189"   "0.0245" "0.032" 
#[20,] "mygrid_model_36" "2.47"      "179"                 "20"   "0.00017" "0.0382"   "0.1199"   "0.0113" "0.0303"



tabb2 = as.data.frame(tabb)
tabb2
tabb2[,c(2,4:9)] = apply(tabb2[,c(2,4:9)],2,as.numeric)
str(tabb2)
dev.off()
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(1:20, cbind(tabb2[,6], tabb2[,7],tabb2[,8], tabb2[,9]),pch=19,cex=.7, 
        col=c("orange", "red","lightblue","blue"),
        type="b",ylab="Errors/logloss", xlab="models")
legend("topleft", legend=c("logloss_train", "logloss_test",
                           "Error_train", "Error_test"), pch=19, cex=.5,
       col=c("orange", "red","lightblue","blue"))


write.csv(tabb2,"~/FAC/8vo/Sem.Est/Proyecto/intento10.csv", row.names = TRUE)

tabb2

###### mejores modelos ################
library(h2o)
h2o.init(nthreads = -1)  #internet connection required.
h2o.removeAll()
h2o.getVersion()


### Cargamos los datos####
m_test <- read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtest.csv") #C785 es la variable respuesta
m_train<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTtrain.csv") #[M[M]]
#m_validate<-read.csv("~/FAC/8vo/Sem.Est/Proyecto/MNISTvalidate.csv")

test<-as.h2o(m_test)
train<-as.h2o(m_train)
#validate<-as.h2o(m_validate) ##creo que hay que hacerle as factor

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])


##### intento 3 modelo 9 "misclassif" #####
set.seed(7)
system.time(
  i3_modelo1 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i3_modelo1",
                                 epochs = 50,
                                 hidden = c(201,186),
                                 l1 =0.00067,
                                 stopping_rounds = 3,
                                 score_interval = 3,
                                 stopping_metric="misclassification",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 #nfolds = 5, 
                                 seed = 1))

#user  system elapsed 
#0.39    0.03  205.09 (3.41min)

##### intento 3 modelo 9 "logloss" #####

set.seed(7)
system.time(
  i3_modelo2 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i3_modelo2",
                                 epochs = 50,
                                 hidden = c(201,186),
                                 l1 =0.00067,
                                 stopping_rounds = 3,
                                 score_interval = 3,
                                 stopping_metric="logloss",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 #nfolds = 5, 
                                 seed = 1))

#user  system elapsed 
#0.30    0.02  212.34 (3.53)



##### intento 6 modelo 3 "misclassif"####

set.seed(7)
system.time(
  i6_modelo1 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i6_modelo1",
                                 epochs = 100,
                                 hidden = c(145,123,157),
                                 l1 =0.00034,
                                 stopping_rounds = 3,
                                 score_interval = 2,
                                 stopping_metric="misclassification",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 #nfolds = 5, 
                                 seed = 1))

#summary(i6_modelo1, show_stack_traces = TRUE)

#user  system elapsed 
#0.34    0.05  256.45 (4.27min)

#user  system elapsed 
#0.24    0.08  101.89 


##### intento 6 modelo 3 "logloss"####

set.seed(7)
system.time(
  i6_modelo2 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i6_modelo2",
                                 epochs = 100,
                                 hidden = c(145,123,157),
                                 l1 =0.00034,
                                 stopping_rounds = 3,
                                 score_interval = 2,
                                 stopping_metric="logloss",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 #nfolds = 5, 
                                 seed = 1))

#user  system elapsed 
#0.31    0.00  276.08 (4.60min)

#user  system elapsed 
#0.24    0.01   85.33 


###### intento 6 modelo 4 "misclassif"#####

set.seed(7)
system.time(
  i6_modelo14 <- h2o.deeplearning(x = x,y=y,
                                  training_frame = train,
                                  model_id = "i6_modelo14",
                                  epochs = 100,
                                  hidden = c(190,174,190),
                                  l1 =0.00043,
                                  stopping_rounds = 3,
                                  score_interval = 2,
                                  stopping_metric="misclassification",
                                  stopping_tolerance=0.05,
                                  distribution = "multinomial",
                                  #nfolds = 5, 
                                  seed = 1))


#user  system elapsed 
#0.45    0.02  349.06 (5.81)

###### intento 6 modelo 4 "logloss"#####
set.seed(7)
system.time(
  i6_modelo142 <- h2o.deeplearning(x = x,y=y,
                                   training_frame = train,
                                   model_id = "i6_modelo142",
                                   epochs = 100,
                                   hidden = c(190,174,190),
                                   l1 =0.00043,
                                   stopping_rounds = 3,
                                   score_interval = 2,
                                   stopping_metric="logloss",
                                   stopping_tolerance=0.05,
                                   distribution = "multinomial",
                                   #nfolds = 5, 
                                   seed = 1))


#user  system elapsed 
#0.43    0.05  158.09
###### intento 8 modelo 1 "misclass"#####

set.seed(7)
system.time(
  i8_modelo1 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i8_modelo1",
                                 epochs = 20,
                                 hidden = c(253,69),
                                 l1 =0.00043,
                                 stopping_rounds = 3,
                                 score_interval = 2,
                                 stopping_metric="misclassification",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 #nfolds = 5, 
                                 seed = 1))

#user  system elapsed 
#0.31    0.05  257.21




###### intento 8 modelo 1 "logloss"#####

set.seed(7)
system.time(
  i8_modelo2 <- h2o.deeplearning(x = x,y=y,
                                 training_frame = train,
                                 model_id = "i8_modelo2",
                                 epochs = 20,
                                 hidden = c(253,69),
                                 l1 =0.00043,
                                 stopping_rounds = 3,
                                 score_interval = 2,
                                 stopping_metric="logloss",
                                 stopping_tolerance=0.05,
                                 distribution = "multinomial",
                                 nfolds = 5, 
                                 seed = 1))

#user  system elapsed 
#0.41    0.01  198.85 (3.31)





#### Tabla ####
tabb=matrix(NA, nrow=8, ncol=9)
colnames(tabb)=c("model","RunTimeMins","layers","epochs","l1",
                 "loglss_trn","loglss_tst",
                 "Err_trn","Err_tst")
tabb 




models=list()
mperftrn=list()
mperftst=list()

model_list<-c("i3_modelo1","i3_modelo2","i6_modelo1","i6_modelo2","i6_modelo14",
              "i6_modelo142","i8_modelo1","i8_modelo2")

models <- lapply(model_list, function(id){ h2o.getModel(id)})



for(i in 1:8){
  mperftrn[[i]] <- h2o.performance(models[[i]], newdata = train)
  mperftst[[i]] <- h2o.performance(models[[i]], newdata = test)
}


for(i in 1:8){
  tabb[i,1]=models[[i]]@model_id
  tabb[i,2]=round(models[[i]]@model$run_time/60000, digits=2)
  tabb[i,3]=paste(models[[i]]@allparameters$hidden, collapse="-")
  tabb[i,4]=round(models[[i]]@allparameters$epochs, digits=2)
  tabb[i,5]=models[[i]]@allparameters$l1
  # tabb[i,6]=models[[i]]@allparameters$input_dropout_ratio 
  tabb[i,6]=round(h2o.logloss(mperftrn[[i]]), digits=4)
  tabb[i,7]=round(h2o.logloss(mperftst[[i]]), digits=4)
  #tab[i+1,9]=models[[i]]@model$cross_validation_metrics_summary$mean[4]
  tabb[i,8]=round(mperftrn[[i]]@metrics$cm$table$Error[11], digits=4)
  tabb[i,9]=round(mperftst[[i]]@metrics$cm$table$Error[11], digits=4)
  #tab[i+1,12]=models[[i]]@model$cross_validation_metrics_summary$mean[2]
}
tabb

######model          RunTimeMins layers        epochs l1        loglss_trn loglss_tst Err_trn  Err_tst 
#[1,] "i3_modelo1"   "3.41"      "201-186"     "50"   "0.00067" "0.0789"   "0.12"     "0.0231" "0.0344"
#[2,] "i3_modelo2"   "3.53"      "201-186"     "50"   "0.00067" "0.077"    "0.1157"   "0.0218" "0.0338"
#[3,] "i6_modelo1"   "1.68"      "145-123-157" "100"  "0.00034" "0.0579"   "0.0991"   "0.0173" "0.0288"
#[4,] "i6_modelo2"   "1.41"      "145-123-157" "100"  "0.00034" "0.0692"   "0.1123"   "0.0223" "0.0339"
#[5,] "i6_modelo14"  "5.8"       "190-174-190" "100"  "0.00043" "0.0527"   "0.1078"   "0.015"  "0.0299"
#[6,] "i6_modelo142" "2.62"      "190-174-190" "100"  "0.00043" "0.0612"   "0.1021"   "0.0189" "0.0297"
#[7,] "i8_modelo1"   "4.27"      "253-69"      "20"   "0.00043" "0.0534"   "0.0961"   "0.0153" "0.0262"
#[8,] "i8_modelo2"   "3.29"      "253-69"      "20"   "0.00043" "0.0587"   "0.0937"   "0.0186" "0.0266"


for (i in 1:8) {
  x<-round(h2o.logloss(mperftst[[i]]), digits=4)-round(h2o.logloss(mperftrn[[i]]), digits=4)
  print(x)
}


#+   print(x)
#+ }
#[1] 0.0411
#[1] 0.0387
#[1] 0.0412
#[1] 0.0431
#[1] 0.0551
#[1] 0.0409
#[1] 0.0427
#[1] 0.035


#### el mejor es el 8 con logloss




tabb2 = as.data.frame(tabb)
tabb2
tabb2[,c(2,4:9)] = apply(tabb2[,c(2,4:9)],2,as.numeric)
str(tabb2)
dev.off()
par(mfrow = c(1,1), mai = c(1,1,1,1))
matplot(1:8, cbind(tabb2[,6], tabb2[,7],tabb2[,8], tabb2[,9]),pch=19,cex=.7, 
        col=c("orange", "red","lightblue","blue"),
        type="b",ylab="Errors/logloss", xlab="models")
legend("topright", legend=c("logloss_train", "logloss_test",
                            "Error_train", "Error_test"), pch=10, cex=.4,
       col=c("orange", "red","lightblue","blue"))


write.csv(tabb2,"~/FAC/8vo/Sem.Est/Proyecto/deep.csv", row.names = TRUE)

tabb2


###############################################################################################################
##########   REGRESION LOGÍSTICA   ############################################################################
###############################################################################################################


#modelos = list()
pftrain = list()
pftest = list()

infotable <- matrix(ncol = 8,nrow = 5)
colnames(infotable) = c('Train','Test','CV','Logloss_Tr', 'R Deviance_Tr','Logloss_Ts', 'R Deviance_Ts','l')
rownames(infotable) = c('RegLog_maxit_50', 'RegLog_maxit_100',
                        'RegLog_maxit_1e3', 'RegLog_maxit_1e4',
                        'RegLog_maxit_1e5')
infotable

system.time(modelos <- lapply(c(50, 100, 1000, 1e4, 1e5),
                              function(iters){
                                h2o.glm(x=x,y=y,
                                        training_frame = train,
                                        nfolds = 5,
                                        family = "multinomial",
                                        lambda = 0,
                                        seed = 1,
                                        max_iterations = iters)}))
# max_it default
# user  system elapsed 
# 1.093   0.097  33.097 
# 
# max_it = 1e6
#  user  system elapsed 
# 9.531   1.279 562.016 
# 
# con lapply (7 modelos)
# user   system  elapsed 
# 65.531    8.995 3471.894 

################# nfolds=5############
#user  system elapsed 
#30.47    4.45 4127.61 (68.79min)

# intento 2 
#user  system elapsed
#25.30    4.89 3740.56 (62.34min)



summary(modelos[[1]])
summary(modelos[[2]])
summary(modelos[[3]])
summary(modelos[[4]])
summary(modelos[[5]])
summary(modelos[[6]])
summary(modelos[[7]])


for (i in c(1:length(modelos))) {
  (pftrain[[i]] <- h2o.performance(modelos[[i]],train))
  (pftest[[i]] <- h2o.performance(modelos[[i]],test))
  h2o.std_coef_plot(modelos[[i]])
}

# (pred1 <- h2o.predict(modelos[[1]],valid)$predict)

info <- function(mod,ptr,pte){
  return(c(
    ptr@metrics$cm$table$Error[11],
    pte@metrics$cm$table$Error[11],
    mod@model$cross_validation_metrics@metrics$cm$table$Error[11],
    ptr@metrics$logloss,
    ptr@metrics$residual_deviance,
    pte@metrics$logloss,
    pte@metrics$residual_deviance,
    mod@model$lambda_best
  ))
}



for (i in c(1:length(modelos))){
  infotable[i,] = info(modelos[[i]],pftrain[[i]],pftest[[i]])
}

infotable

################# Train       Test       CV Logloss_Tr R Deviance_Tr Logloss_Ts R Deviance_Ts l
#RegLog_maxit_50  0.054775 0.08566667 0.090400  0.1971165      15769.32  0.3697912      6741.775 0
#RegLog_maxit_100 0.050450 0.09277778 0.098375  0.1797972      14383.78  0.4349849      8035.303 0
#RegLog_maxit_1e3 0.046650 0.10100000 0.109225  0.1688844      13510.76  0.5680563     12188.012 0
#RegLog_maxit_1e4 0.046550 0.10077778 0.110125  0.1686774      13494.19  0.5796767     13034.858 0
#RegLog_maxit_1e5 0.046550 0.10077778 0.110125  0.1686774      13494.19  0.5796767     13034.858 0




plot(infotable[,"Logloss_Tr"],xlim=c(1,5), ylim=c(0.1,0.6), col="red", type="b",
     xlab="modelo",ylab="logloss")
points(infotable[,"Logloss_Ts"], col="blue",type="b")

legend("topleft", legend=c("logloss_test", "logloss_train"), pch=10, cex=.7,
       col=c("blue", "red"))


# ---- Otros modelos

system.time(modelos[[6]] <- h2o.glm(x=x,y=y,
                                    training_frame = train,
                                    nfolds = 5,
                                    family = "multinomial",
                                    seed = 1,
                                    max_iterations = 50,
                                    lambda_search = T))
# user  system elapsed 
# 0.710   0.067  32.996 

summary(modelos[[6]])

# ---- Modelos Grids
alpha_opts <- seq(0.1, 0.95, 0.05)
lambda_opts <- c(0.1, 0.01, 0.001, 0.0001, 0.00001)

hyper_params <- list(alpha = alpha_opts, lambda = lambda_opts)

search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 20,
                       seed=1)

system.time(glm_mods_grid <- h2o.grid("glm",
                                      grid_id = "random_grid",
                                      hyper_params = hyper_params,
                                      search_criteria = search_criteria,
                                      x = x,
                                      y = y,
                                      distribution = "multinomial",
                                      family = "multinomial",
                                      training_frame = train,
                                      seed = 1,
                                      stopping_rounds = 3,
                                      stopping_tolerance = 1e-3,
                                      stopping_metric = "logloss"))

# ---- Otro Modelo
system.time(modelos[[1]] <- h2o.glm(x=x,y=y,
                                    training_frame = train,
                                    nfolds = 0, 
                                    family = "multinomial",
                                    lambda = 0,
                                    seed = 1,
                                    lambda_search = T))

############# validate ######

pred<-h2o.predict(modelos[[1]], newdata=validate)
pred


predi<-as.data.frame(pred)

write.csv(predi,"~/FAC/8vo/Sem.Est/Proyecto/Equipo4_GnzlzFeria_RegLog_pred.csv",col.names = T)

View(predi)





















