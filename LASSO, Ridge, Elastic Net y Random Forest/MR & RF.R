
#######Ejercicio 1 ######
library(MASS)
library(leaps)
library(glmnet)
library(randomForest)


load("~/FAC/8vo/Sem.Est/Tarea 5/Tarea5y6_4Gzlez Feria.RData")

#load("T5_IMG.RData")
rm(list=ls())
par(mfrow=c(1,1))
data(Boston)
Boston$rad=as.numeric(Boston$rad)
# Boston <- as.data.frame(scale(Boston))    # escalar?
attach(Boston)

modelos <- list(6)
mods_aux <- list(6)

errores <- matrix(NA, ncol=7,nrow=8)
rownames(errores) <- c("1. lm", "2. stepAIC", "3. stepBIC", "4. regsubsets",
                       "5. lasso", "6. ridge", "7. elastic_0.5",
                       "8. RandomForest")
colnames(errores) <- c("Tr error (Ap)", "Tr/test 500 (NoAp)", "#Parms(df)",
                       "lambda.min", "lambda 1se", "mtry", "ntree")
errores

coeficientes <- matrix(NA, ncol=14, nrow=7) # randomForest es NO parametrico
rownames(coeficientes) <- c("1. lm", "2. stepAIC", "3. stepBIC", "4. regsubsets",
                       "5. lasso", "6. ridge", "7. elastic_0.5")
colnames(coeficientes) <- c("(Intercept)", colnames(Boston[-1]))
coeficientes

###############1#########################

bc <- boxcox(lm(crim~., data=Boston), plotit = T)
(bc$x[which(bc$y == max(bc$y))])    # lambda = 0.0202 elegimos transf log

# mat_cor <- signif(cor(Boston[,-4]), digits=3)

###############2 y 3##############

modelos[[1]] <- lm(log(crim)~., data=Boston) # modelo aditivo
drop1(modelos[[1]], test="F")
(summary(modelos[[1]]))

modelos[[2]] <- step(lm(log(crim)~., data = Boston), k = 2, trace = F) # mod AIC
drop1(modelos[[2]], test="F")
(summary(modelos[[2]]))

modelos[[3]] <- step(lm(log(crim)~., data = Boston), k =log(dim(Boston)[1]),
                   trace = F) # mod BIC
drop1(modelos[[3]], test="F")
(summary(modelos[[3]]))

errores[1:3, 1] <- sapply(c(1:3), function(mod){mean(modelos[[mod]]$residuals^2)})
errores[1:3, 3] <- sapply(c(1:3), function(mod){length(modelos[[mod]]$coefficients)})
# errores[1:3, 3] <- sapply(c(1:3), function(mod){summary(modelos[[mod]])$adj.r.squared})

errores

mod_lm_regs <- regsubsets(log(crim)~., data = Boston, nvmax = 13)

par(mfrow=c(1,3))
plot(summary(mod_lm_regs)$rss, pch=19, col="blue",
     xlab="Variables k usadas en el modelo",
     ylab="Residual sum of squares")
abline(h = min(summary(mod_lm_regs)$rss), col="red", lty=2)

plot(summary(mod_lm_regs)$bic, pch=19, col="blue",
     xlab="Variables k usadas en el modelo",
     ylab="BIC")
abline(h = min(summary(mod_lm_regs)$bic), col="red", lty=2)

plot(summary(mod_lm_regs)$adjr2, pch=19, col="blue",
     xlab="Variables k usadas en el modelo",
     ylab="Adj R2")
abline(h = max(summary(mod_lm_regs)$adjr2), col="red", lty=2)

summary(mod_lm_regs)$which[8,] # zn, indus, nox, age, rad, ptratio, black, lstat

modelos[[4]] <- lm(log(crim)~ zn + indus + nox + age + rad + ptratio + black + lstat, data = Boston)
errores[4, 1] <- sapply(c(4), function(mod){mean(modelos[[mod]]$residuals^2)})
errores[4, 3] <- sapply(c(4), function(mod){length(modelos[[mod]]$coefficients)})
# errores[4, 3] <- sapply(c(4), function(mod){summary(modelos[[mod]])$adj.r.squared})

par(mfrow=c(1,1))

x <- model.matrix(log(crim)~. , data = Boston) 
y <- log(crim) 

####### 4################# 
set.seed(81805)
alpha <- c(1, 0, 0.5) # lasso, ridge, elastic 0.5
mods <- lapply(alpha, function(num){glmnet(x, y, alpha = num)})
modelos <- append(modelos, mods)

m_cv_aux <- lapply(alpha, function(num){cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=num)})
m_cv_aux[[5]] <- m_cv_aux[[1]]
m_cv_aux[[6]] <- m_cv_aux[[2]]
m_cv_aux[[7]] <- m_cv_aux[[3]]

par(mfrow=c(1,3), mar=c(4,4,6,2))
plot(m_cv_aux[[5]], main="Lasso", cex.main=1.3) # 13 lambda, 8 lambda 1se
plot(m_cv_aux[[6]], main="Ridge", cex.main=1.3) # 13 lambda y lambda 1se
plot(m_cv_aux[[7]], main="Elastic: .5,.5", cex.main=1.3) # 13 lambda, 9 lambda 1se

# Seleccionamos lambda.min

m_cv_aux[[5]]; m_cv_aux[[6]]; m_cv_aux[[7]]

par(mfrow=c(1,1))

errores[5:7, 1] <- sapply(c(5:7), function(mod){assess.glmnet(modelos[[mod]],s=m_cv_aux[[mod]]$lambda.min,newx = x, newy = y)$mse})
errores[5:7, 3] <- sapply(c(5:7), function(mod){m_cv_aux[[mod]]$nzero[which(m_cv_aux[[mod]]$lambda==m_cv_aux[[mod]]$lambda.min)]+1})
errores[5:7, 4] <- sapply(c(5:7), function(mod){m_cv_aux[[mod]]$lambda.min})
errores[5:7, 5] <- sapply(c(5:7), function(mod){m_cv_aux[[mod]]$lambda.1se})
errores

N <- 500
ntrain <- round(length(y)*0.75)
aux <- matrix(NA,nrow=N, ncol=8) 
set.seed(111213)

t1=proc.time()
for(iter in 1:N){
    train <- sample(1:length(y), ntrain, replace=FALSE)
    mods_aux[[1]] <- lm(log(crim)~.,data=Boston[train,])
    aux[iter, 1] <- mean((predict(mods_aux[[1]], newdata=Boston[-train,]) -log(crim)[-train])^2)
    mods_aux[[2]] <- lm(log(crim)~zn + indus + nox + age + rad + ptratio + black + lstat + medv, data = Boston[train,])
    aux[iter, 2] <- mean((predict(mods_aux[[2]],newdata=Boston[-train,]) -log(crim)[-train])^2)
    mods_aux[[3]] <- lm(log(crim)~zn + nox + age + rad + black + lstat, data = Boston[train,])
    aux[iter, 3] <- mean((predict(mods_aux[[3]],newdata=Boston[-train,]) -log(crim)[-train])^2)
    mods_aux[[4]] <- lm(log(crim)~zn + indus + nox + age + rad + ptratio + black + lstat, data = Boston[train,])
    aux[iter, 4] <- mean((predict(mods_aux[[4]],newdata=Boston[-train,]) -log(crim)[-train])^2)
    mods_aux[[5]] <- glmnet(x[train,],y[train], alpha=1)
    m_cv_aux[[5]] <- cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=10,alpha=1)
    aux[iter, 5] <- assess.glmnet(mods_aux[[5]],s=m_cv_aux[[5]]$lambda.min ,newx = x[-train,], newy = y[-train])$mse
    mods_aux[[6]] <- glmnet(x[train,],y[train], alpha=0)
    m_cv_aux[[6]] <- cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=10,alpha=0)
    aux[iter, 6] <- assess.glmnet(mods_aux[[6]],s=m_cv_aux[[6]]$lambda.min ,newx = x[-train,], newy = y[-train])$mse
    mods_aux[[7]] <- glmnet(x[train,],y[train], alpha=0.5)
    m_cv_aux[[7]] <- cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=10,alpha=0.5)
    aux[iter, 7] <- assess.glmnet(mods_aux[[7]],s=m_cv_aux[[7]]$lambda.min ,newx = x[-train,], newy = y[-train])$mse
}
(proc.time()-t1)
# user  system elapsed 
# 178.815   0.113 179.244 

errores[1:7, 2] <- sapply(c(1:7), function(mod){mean(aux[,mod])})

mod_aux_ABIC <- list(2)
aux_ABIC <- matrix(NA, nrow = N, ncol = 2)

t1=proc.time()
for (iter in 1:N){
    train <- sample(1:length(y), ntrain, replace=FALSE)
    mod_aux_ABIC[[1]] <- step(lm(log(crim)~., data = Boston[train,]), k = 2, trace = F)
    mod_aux_ABIC[[2]] <- step(lm(log(crim)~., data = Boston[train,]), k =log(dim(Boston)[1]), trace = F)
    aux_ABIC[iter, 1] <- mean((predict(mod_aux_ABIC[[1]],newdata=Boston[-train,]) -log(crim)[-train])^2)
    aux_ABIC[iter, 2] <- mean((predict(mod_aux_ABIC[[2]],newdata=Boston[-train,]) -log(crim)[-train])^2)
}
(proc.time()-t1)
# user  system elapsed 
# 68.86    0.04   69.00 

(aux_ABIC1 <- c(mean(aux_ABIC[,1]), mean(aux_ABIC[,2])))



mod_aux_ABIC2 <- list(2)
aux_ABIC2 <- matrix(NA, nrow = N, ncol = 2)

train <- sample(1:length(y), ntrain, replace=FALSE)
mod_aux_ABIC2[[1]] <- step(lm(log(crim)~., data = Boston[train,]), k = 2, trace = F)
mod_aux_ABIC2[[2]] <- step(lm(log(crim)~., data = Boston[train,]), k =log(dim(Boston)[1]), trace = F)

t1=proc.time()
for (iter in 1:N){
    train <- sample(1:length(y), ntrain, replace=FALSE)
    aux_ABIC2[iter, 1] <- mean((predict(mod_aux_ABIC2[[1]],newdata=Boston[-train,]) -log(crim)[-train])^2)
    aux_ABIC2[iter, 2] <- mean((predict(mod_aux_ABIC2[[2]],newdata=Boston[-train,]) -log(crim)[-train])^2)
}
(proc.time()-t1)
# user  system elapsed 
# 1.398   0.003   1.402 

(aux_ABIC2 <- c(mean(aux_ABIC2[,1]), mean(aux_ABIC2[,2])))

#########7######################

t1=proc.time()
list_rf <- lapply(c(1:13), function(mtry){randomForest(log(crim)~., data = Boston, mtry=mtry, ntree=500, importance=T)})
(proc.time()-t1)

mseap <- sapply(c(1:13), function(mod){mean((predict(list_rf[[mod]], Boston)- log(crim))^2)})
mseoob <- sapply(c(1:13), function(mod){list_rf[[mod]]$mse[500]})
plot(mseap, col="blue", type = "b", ylab = "MSE", xlab = "mtry")
par(new= TRUE)
plot(mseoob, col="red", type="b", axes=FALSE, xlab = "", ylab = "")
legend("topright", legend=c("mse_app", "mse_oob"), pch=19,
       cex=.6, col = c("blue", "red"))

# Dada la grafica seleccionamos mtry =7, ntree = 500

modelos[[8]] <- list_rf[[7]]
errores[8,1] <- mseap[7]

# varImpPlot(modelos[[8]])

auxrf <- c()
t1=proc.time()
for(iter in 1:N){
    train <- sample(1:length(y), ntrain, replace=FALSE)
    mods_aux[[8]]<-randomForest(log(crim)~.,data=Boston[train,], ntree=500, mtry=7, importance=TRUE)
    auxrf[iter]<-mean((predict(mods_aux[[8]], newdata=Boston[-train,])-log(crim)[-train])^2)
}
(proc.time()-t1)


errores[8, 2] <- mean(auxrf)
errores[8, 6:7] <- c(7, 500)

coeficientes[1, names(modelos[[1]]$coefficients)] <- modelos[[1]]$coefficients
coeficientes[2, names(modelos[[2]]$coefficients)] <- modelos[[2]]$coefficients
coeficientes[3, names(modelos[[3]]$coefficients)] <- modelos[[3]]$coefficients
coeficientes[4, names(modelos[[4]]$coefficients)] <- modelos[[4]]$coefficients

auxcoef <- predict.glmnet(modelos[[5]], s=m_cv_aux[[5]]$lambda.min, type = "coefficients")
coeficientes[5, 1] <- auxcoef[1]
coeficientes[5, "zn"] <- auxcoef[3] # el 2 tiene un cero (intercept repetido) 
coeficientes[5, "indus"] <- auxcoef[4]
coeficientes[5, "chas"] <- auxcoef[5]
coeficientes[5, "nox"] <- auxcoef[6]
coeficientes[5, "rm"] <- auxcoef[7]
coeficientes[5, "age"] <- auxcoef[8]
coeficientes[5, "dis"] <- auxcoef[9]
coeficientes[5, "rad"] <- auxcoef[10]
coeficientes[5, "ptratio"] <- auxcoef[12] # 11 no existente en el modelo
coeficientes[5, "black"] <- auxcoef[13]
coeficientes[5, "lstat"] <- auxcoef[14]
coeficientes[5, "medv"] <- auxcoef[15]

auxcoef <- predict.glmnet(modelos[[6]], s=m_cv_aux[[6]]$lambda.min, type = "coefficients")
coeficientes[6, 1] <- auxcoef[1]
coeficientes[6, "zn"] <- auxcoef[3] # el 2 tiene un cero (intercept repetido) 
coeficientes[6, "indus"] <- auxcoef[4]
coeficientes[6, "chas"] <- auxcoef[5]
coeficientes[6, "nox"] <- auxcoef[6]
coeficientes[6, "rm"] <- auxcoef[7]
coeficientes[6, "age"] <- auxcoef[8]
coeficientes[6, "dis"] <- auxcoef[9]
coeficientes[6, "rad"] <- auxcoef[10]
coeficientes[6, "tax"] <- auxcoef[11]
coeficientes[6, "ptratio"] <- auxcoef[12]
coeficientes[6, "black"] <- auxcoef[13]
coeficientes[6, "lstat"] <- auxcoef[14]
coeficientes[6, "medv"] <- auxcoef[15]

auxcoef <- predict.glmnet(modelos[[7]], s=m_cv_aux[[7]]$lambda.min, type = "coefficients")
coeficientes[7, 1] <- auxcoef[1]
coeficientes[7, "zn"] <- auxcoef[3] # el 2 tiene un cero (intercept repetido) 
coeficientes[7, "indus"] <- auxcoef[4]
coeficientes[7, "chas"] <- auxcoef[5]
coeficientes[7, "nox"] <- auxcoef[6]
coeficientes[7, "rm"] <- auxcoef[7]
coeficientes[7, "age"] <- auxcoef[8]
coeficientes[7, "dis"] <- auxcoef[9]
coeficientes[7, "rad"] <- auxcoef[10]
coeficientes[7, "ptratio"] <- auxcoef[12]
coeficientes[7, "black"] <- auxcoef[13]
coeficientes[7, "lstat"] <- auxcoef[14]
coeficientes[7, "medv"] <- auxcoef[15]

(round(coeficientes, 4))
(round(errores, 4))



 prueb <- Boston[-train, "crim"]
 pred <- predict(modelos[[3]], newdata = Boston[-train,])
 plot(pred, log(prueb))
 abline(0,1)
 
 
 #####Ejercicio 2####
 
 ####Preparamos los datos####
 library(Matrix)
 library(glmnet)
 library(hdi)
 data("riboflavin")
 datos=riboflavin
 #View(datos)
 
 x = model.matrix(datos$y~datos$x, data=datos)
 y = datos$y
 
 Nmodels=3
 M=list(Nmodels)
 Mcv=list(Nmodels)
 Mcoef=list(Nmodels)
 Maux=list(Nmodels)
 Mcvaux=list(Nmodels)
 
 ##decidimos k=5 por la cantidad de datos, ya que hicimos pruebas con k=10 y el error aumentaba
 ##y esto tiene sentido ya que los train de quedaban con 7 datos, y con k=5 tienen el doble y sesga
 ## menos el error
 
 err=matrix(NA, ncol=8,nrow=Nmodels)
 rownames(err)=c( "Ridge", "Lasso", "Elastic net")
 colnames(err)=c("TasaErr(Ap)", "  Tr/test500(NoAp)", "sd","Cross.v(k=5, B=1)",
                 "#Betas(df)", "lambda.min" ,"lambda.1se", "dfmax=")
 
 err ##tabla vacía que iremos llenando
 
 
 
 ###Ridge####
 ridge=1
 
 set.seed(1); (M[[ridge]] = glmnet(x,y, alpha=0)) 
 M[[1]]
 
 
 ##1)
 
 #modelo original
 set.seed(12); Mcv[[ridge]]=cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=0) #entrenamos con todos los datos
 Mcv[[1]] 
 
 #     Lambda Measure      SE Nonzero
 # min  5.934  0.2535 0.04379    4088
 # 1se 27.544  0.2952 0.04030    4088
 
 
 ###Entonces para seleccionar el modelo tomamos _lambda.min_ y dejamos libre dfmax
 
 plot(Mcv[[1]])
 
 
 
 #Tasa de error aparente
 err[ridge,1]=assess.glmnet(M[[ridge]],s=Mcv[[ridge]]$lambda.min,newx = x, newy = y)$mse
 #tasa de error cv
 err[ridge,4]=Mcv[[ridge]]$cvm[which(Mcv[[ridge]]$lambda==Mcv[[ridge]]$lambda.min)]
 #num de variables
 err[ridge,5]=Mcv[[ridge]]$nzero[which(Mcv[[ridge]]$lambda==Mcv[[ridge]]$lambda.min)]+1 ##mas 1 por el intercepto
 #lambda min
 err[ridge,6]=Mcv[[ridge]]$lambda.min
 #lambda 1se
 err[ridge,7]=Mcv[[ridge]]$lambda.1se
 #num de var ajustadas
 err[ridge,8]=4088
 
 cbind(100*err[,1:4],err[,5:8])#### HASTA AQUI, sólo tenemos la tasa de error ap, la de cv con B=1 y las lambdas
 ##ahora hay que sacar las tasas con ReptTra/test
 
 N=500 #num de rep
 aux=matrix(NA,nrow=N, ncol=10)
 
 #Ridge Train/Test
 t1=proc.time()
 set.seed(123)
 for(Nrep in 1:N){
     train = sample(seq(71),57,replace=FALSE)
     Maux[[ridge]] = glmnet(x[train,],y[train], alpha=0)
     Mcvaux[[ridge]]=cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=5,alpha=0)
     aux[Nrep,ridge]=assess.glmnet(Maux[[ridge]],s=Mcvaux[[ridge]]$lambda.min ,newx = x[-train,], newy = y[-train])$mse
 }
 err[ridge,2]=mean(aux[,ridge])
 err[ridge,3]=sd(aux[,ridge])
 (proc.time()-t1)
 
 ######Usando train 65%
 #user  system elapsed 
 #492.84   78.75  586.38  #10 min
 
 ############# TasaErr(Ap)   Tr/test100(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min lambda.1se dfmax=
 #Ridge           2.76317           30.96243 12.64775          27.70554       4089   5.934163   34.75651   4088
 #Lasso                NA                 NA       NA                NA         NA         NA         NA     NA
 #Elastic net          NA                 NA       NA                NA         NA         NA         NA     NA
 
 err
 cbind(100*err[,1:4],err[,5:8])
 
 ####Usando train 80%
 #             TasaErr(Ap)   Tr/test500(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min lambda.1se dfmax=
 # Ridge           2.76317           27.46009 14.25293          25.34601       4089   5.934163   27.54394   4088
 # Lasso                NA                 NA       NA                NA         NA         NA         NA     NA
 # Elastic net          NA                 NA       NA                NA         NA         NA         NA     NA
 
 #   user  system elapsed 
 # 552.38   95.99  650.47   10 casi 11 min
 
 
 
 
 ####################Lasso####
 lasso=2
 set.seed(123); (M[[lasso]] = glmnet(x,y, alpha=1))
 M[[2]] 
 
 #modelo original
 set.seed(321); Mcv[[lasso]]=cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=1) #entrenamos con todos los datos
 Mcv[[2]] 
 
 #      Lambda Measure      SE Nonzero
 # min 0.03641  0.2405 0.04591      41
 # 1se 0.06983  0.2853 0.07003      31
 
 ###Entonces para seleccionar el modelo tomamos _lambda.min_ y _dfmax=40_
 
 plot(Mcv[[2]])
 
 
 ##modelo ajustado
 #modelo dfmax = 41
 set.seed(1); Mcv[[lasso]]=cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=1, dfmax=41)
 Mcv[[2]]
 
 #      Lambda Measure      SE Nonzero
 # min 0.04813  0.2419 0.08237      35
 # 1se 0.11649  0.3184 0.11009      24
 
 
 plot(Mcv[[2]])
 
 
 #Tasa de error aparente
 err[lasso,1]=assess.glmnet(M[[lasso]],s=Mcv[[lasso]]$lambda.min,newx = x, newy = y, dfmax=41)$mse
 
 err[lasso,4]=Mcv[[lasso]]$cvm[which(Mcv[[lasso]]$lambda==Mcv[[lasso]]$lambda.min)]
 
 err[lasso,5]=Mcv[[lasso]]$nzero[which(Mcv[[lasso]]$lambda==Mcv[[lasso]]$lambda.min)]+1
 
 err[lasso,6]=Mcv[[lasso]]$lambda.min
 
 err[lasso,7]=Mcv[[lasso]]$lambda.1se
 
 err[lasso,8]=41
 
 cbind(100*err[,1:4],err[,5:8]) ###sólo sacamos las tasas aparentes
 
 
 N=500
 aux=matrix(NA,nrow=N, ncol=10)
 
 #Lasso Train/Test
 t1=proc.time()
 set.seed(23)
 
 for(Nrep in 1:N){
     train = sample(seq(71),57,replace=FALSE)
     Maux[[lasso]] = glmnet(x[train,],y[train], alpha=1)
     Mcvaux[[lasso]]=cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=5,alpha=1)
     aux[Nrep,lasso]=assess.glmnet(Maux[[lasso]],s=Mcvaux[[lasso]]$lambda.min,dfmax=41 ,newx = x[-train,], newy = y[-train])$mse
 }
 err[lasso,2]=mean(aux[,lasso])
 err[lasso,3]=sd(aux[,lasso])
 (proc.time()-t1)
 
 
 ###### Usando train 65%
 #user  system elapsed 
 #101.70    5.56  138.75   2min
 
 #############TasaErr(Ap)   Tr/test100(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min  lambda.1se dfmax=
 #Ridge          2.763170           30.96243 12.64775          27.70554       4089 5.93416252 34.75651335   4088
 #Lasso          3.861503           31.54407 11.70466          20.85702         43 0.03814523  0.06665987     40
 #Elastic net          NA                 NA       NA                NA         NA         NA          NA     NA
 
 
 err
 cbind(100*err[,1:4],err[,5:8])
 
 
 ###### Usando train 80%
 #             TasaErr(Ap)   Tr/test500(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min lambda.1se dfmax=
 # Ridge           2.76317           27.46009 14.25293          25.34601       4089 5.93416252   27.54394   4088
 # Lasso           4.90291           26.64044 14.84498          24.18637         36 0.04813382    0.11649     41
 # Elastic net          NA                 NA       NA                NA         NA         NA         NA     NA
 
 
 #   user  system elapsed 
 # 122.33   12.32  135.80   2min
 
 
 
 
 
 ###Elastic Net####
 enet=3
 set.seed(32); (M[[enet]] = glmnet(x,y, alpha=.5))
 M[[3]] 
 
 #modelo original
 set.seed(325); Mcv[[enet]]=cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=.5) #entrenamos con todos los datos
 Mcv[[3]] 
 
 #      Lambda Measure      SE Nonzero
 # min 0.07282  0.2047 0.04404      55
 # 1se 0.16823  0.2477 0.06043      41
 
 
 ###Entonces para seleccionar el modelo tomamos _lambda.min_ y _dfmax=55_
 
 plot(Mcv[[3]])
 
 
 ##modelo ajustado
 #modelo dfmax = 55
 set.seed(45); Mcv[[enet]]=cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=.5, dfmax=55)
 Mcv[[3]]
 
 #     Lambda Measure     SE Nonzero
 # min 0.0919  0.3172 0.1386      56
 # 1se 0.4071  0.4522 0.1932      20
 
 
 plot(Mcv[[3]])
 
 
 #Tasa de error aparente
 err[enet,1]=assess.glmnet(M[[enet]],s=Mcv[[enet]]$lambda.min,newx = x, newy = y, dfmax=55)$mse
 
 err[enet,4]=Mcv[[enet]]$cvm[which(Mcv[[enet]]$lambda==Mcv[[enet]]$lambda.min)]
 
 err[enet,5]=Mcv[[enet]]$nzero[which(Mcv[[enet]]$lambda==Mcv[[enet]]$lambda.min)]+1
 
 err[enet,6]=Mcv[[enet]]$lambda.min
 
 err[enet,7]=Mcv[[enet]]$lambda.1se
 
 err[enet,8]=55
 
 cbind(100*err[,1:4],err[,5:8]) ###sólo sacamos las tasas aparentes
 
 
 N=500
 aux=matrix(NA,nrow=N, ncol=10)
 
 #enet Train/Test
 t1=proc.time()
 set.seed(333)
 
 for(Nrep in 1:N){
     train = sample(seq(71),57,replace=FALSE)
     Maux[[enet]] = glmnet(x[train,],y[train], alpha=.5)
     Mcvaux[[enet]]=cv.glmnet(x[train,], y[train],type.measure="mse",nfolds=5,alpha=.5)
     aux[Nrep,enet]=assess.glmnet(Maux[[enet]],s=Mcvaux[[enet]]$lambda.min,dfmax=55 ,newx = x[-train,], newy = y[-train])$mse
 }
 err[enet,2]=mean(aux[,enet])
 err[enet,3]=sd(aux[,enet])
 (proc.time()-t1)
 
 
 ##### Usando train 65%
 #user  system elapsed 
 #102.38    6.22  127.36  #2min
 
 ############# TasaErr(Ap)   Tr/test500(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min  lambda.1se dfmax=
 #Ridge          2.763170           30.96243 12.64775          27.70554       4089 5.93416252 34.7565}1335   4088
 #Lasso          3.861503           31.54407 11.70466          20.85702         43 0.03814523  0.06665987     40
 #Elastic net    3.552024           29.92271 12.11720          27.20873         61 0.06635355  0.30798591     74
 
 
 
 err
 cbind(100*err[,1:4],err[,5:8])
 
 
 ##### Usando train 80%
 #             TasaErr(Ap)   Tr/test500(NoAp)       sd Cross.v(k=5, B=1) #Betas(df) lambda.min lambda.1se dfmax=
 # Ridge          2.763170           27.46009 14.25293          25.34601       4089 5.93416252 27.5439424   4088
 # Lasso          4.902910           26.64044 14.84498          24.18637         36 0.04813382  0.1164900     41
 # Elastic net    5.031183           24.69469 13.83126          31.71953         57 0.09189213  0.4071392     55
 
 
 #   user  system elapsed 
 # 119.65   11.34  155.87 
 
 
 
