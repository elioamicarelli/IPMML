## Elio Amicarelli
## IPMML Learning Activities

## Load libs

library(dplyr)
library(caret)
library (tree)

#############
# Functions #
#############

gendata <- function(n, alpha = 0, noise, ploto1=1, ploto2=1)
{
  # Signal + noise
  x <- seq(-3,3,by=0.001)
  x <- sample(x=x, size=n, replace=T)
  noise = rnorm(length(x), 0, noise)
  y <- alpha + I(x**7) - 14*I(x**5) + 49*I(x**3) - 36*x + noise
  x<-x-min(x)
  x<-x/max(x)
  
  # Signal
  xt <- seq(-3,3,by=0.001)
  yt <- alpha + I(xt**7) - 14*I(xt**5) + 49*I(xt**3) - 36*xt
  xt<-xt-min(xt)
  xt<-xt/max(xt)
  
  if(ploto1==1) {
    xt<-xt-min(xt)
    xt<-xt/max(xt)
    plot(x,y, col="violet", pch=1)
  }
  if(ploto2==1) {
    lines(xt, yt, col="black", lwd=2, type="l")
  } 
  
  data <- as.data.frame(cbind(x,y))
  return(data)
}

###############
# Simulations #  
###############
#####
# 1 #
#####
##### A couple of simulations to show how bias and variance are related with model complexity
#
#
## Generate some data
set.seed(9034)
data <- gendata(200,200,50, 1,1)
#
## Different levels of complexity with splines
#
spline1<-smooth.spline(x=data$x,y=data$y, df=2, cv=F)
spline2<-smooth.spline(x=data$x,y=data$y, df=10, cv=F)
spline3<-smooth.spline(x=data$x,y=data$y, df=40, cv=F)
lines(spline1,col="orange")
lines(spline2, col="blue")
lines(spline3, col="darkgreen")
#
yhat.spline1<-predict(spline1, data)$y[1]
trainerror.spline1<-mean((yhat.spline1-data$y)^2)
yhat.spline2<-predict(spline2, data)$y[1]
trainerror.spline2<-mean((yhat.spline2-data$y)^2)
yhat.spline3<-predict(spline3, data)$y[1]
trainerror.spline3<-mean((yhat.spline3-data$y)^2)
#
plot(c(trainerror.spline1,
       trainerror.spline2,
       trainerror.spline3),
     ylab="MSE",type = "l",pch=4,col="darkgrey",
     xaxt="n",xlab="flexibility")
axis(1,labels = c(2,10,40),at=1:3)
points(c(trainerror.spline1,
         trainerror.spline2,
         trainerror.spline3), col=c("orange","blue","darkgreen"), pch = 15)

## Now we can examine bias and variance for different levels of complexity
#
set.seed(9034)
data <- gendata(500,200,50, 1,1)
flexibility<- 180 #2 180
for(i in 1:30){
  print(i)
  Sys.sleep(1)
  indexes<-sample(seq(1:nrow(data)),0.5*nrow(data))
  sampledata<-data[indexes,]
  model<-smooth.spline(x=sampledata$x,y=sampledata$y, 
                       df=flexibility, cv=F)
  lines(model,col="darkgrey")
}


#####
# 2 #
#####
##### The predictive modelling pipeline
#
set.seed(9034)
#
## Generate artificial data
data <- gendata(200,200,30, 1,0)
#
## Perform a train/test split
indexes<-sample(seq(1:nrow(data)),0.8*nrow(data))
data.train<-data[indexes,]
data.test<-data[-indexes,]
#
cat("Length of original data:",nrow(data),"\n", 
    "Length of training set:",nrow(data.train),"\n",
    "Length of test set:",nrow(data.test))
#
### TRAINING PHASE ###
#
## Create two folds for cross-validation from the training data
set.seed(4309)
indexes<-sample(seq(1:nrow(data.train)),0.5*nrow(data.train))
data.train.fold1<-data.train[indexes,]
data.train.fold2<-data.train[-indexes,]
#
## Can you explain what follows?
trainerror<-vector()
validationerror<-vector()
flexibility<-c(seq(from=2,to=50,by=1))
#
for(i in flexibility){
  #
  # Block 1
  cat("Iteration:",1,"\n model's complexity:",i,"\n")
  trainset<-data.train.fold1
  validationset<-data.train.fold2
  model <- smooth.spline(x=trainset$x,y=trainset$y, df=i, cv=F)
  yhat.training.iteration1<-predict(model, trainset)$y[1]
  yhat.validation.iteration1<-predict(model, validationset)$y[1]
  prederror.training.iteration1<-mean((trainset$y - yhat.training.iteration1)^2)
  prederror.validation.iteration1<-mean((validationset$y - yhat.validation.iteration1)^2)
  #
  # Block 2
  cat("Iteration:",2,"\n model's complexity:",i,"\n")
  trainset<-data.train.fold2
  validationset<-data.train.fold1
  model <- smooth.spline(x=trainset$x,y=trainset$y, df=i, cv=F)
  yhat.training.iteration2<-predict(model, trainset)$y[1]
  yhat.validation.iteration2<-predict(model, validationset)$y[1]
  prederror.training.iteration2<-mean((trainset$y - yhat.training.iteration2)^2)
  prederror.validation.iteration2<-mean((validationset$y - yhat.validation.iteration2)^2)
  #
  # Block 3
  prederror.training<-mean(prederror.training.iteration1,prederror.training.iteration2)
  prederror.validation<-mean(prederror.validation.iteration1,prederror.validation.iteration2)
  #
  trainerror<-append(trainerror,prederror.training)
  validationerror<-append(validationerror,prederror.validation)
}

## Plot train error
plot(trainerror,type="l", xlab = "complexity", ylab = "MSE", ylim = c(0,max(validationerror)), col="blue")
# Best model complexity according with the training error
flexibility[which.min(trainerror)]
#
## Plot train and validation error
plot(trainerror,type="l", xlab = "complexity", ylab = "MSE", ylim = c(0,max(validationerror)), col="blue")
lines(validationerror,type="l", col="violet")
# Best model complexity according with the validation error
flexibility[which.min(validationerror)]
#
## We have a winner! Let's train the final model then
finalmodel <- smooth.spline(x=data.train$x,y=data.train$y, df=flexibility[which.min(validationerror)], cv=F)
#
plot(data.train,col="violet")
lines(finalmodel)
yhat.train<-predict(finalmodel,data.train)$y[1]
plot(data.train$y,unlist(yhat.train),xlab="observed",ylab="predicted", col="purple")
#
### TESTING PHASE ### 
#
# Now we use the final model to make predictions on unseen data
yhat.test<-predict(finalmodel,data.test)$y[1]
#
# we can look at the agreement between our predictions and the actural values
plot(data.test$y,unlist(yhat.test),ylab="predicted",xlab="observed", col="purple")
#
# obtain generalizaiton error
mean((yhat.test-data.test$y)^2)
#
## Check
testerrorcheck<-vector()
for(i in flexibility){
  model <- smooth.spline(x=data.train$x,y=data.train$y, df=i, cv=F)
  yhat.training<-predict(model, data.train)$y[1]
  prederror.training<-mean((yhat.training-data.train$y)^2)
  yhat.test<-predict(model, data.test)$y[1]
  prederror.test<-mean((yhat.test-data.test$y)^2)
  testerrorcheck<-append(testerrorcheck,prederror.test)
}
#
plot(testerrorcheck,ylab="generalization error (MSE)", xlab="flexibility",col="red",type="l") # What do you think? 
abline(v=which.min(testerrorcheck), lty=2,col="red")
abline(v=which.min(validationerror), lty=2,col="blue")

#####
# 3 #
#####
## Heart data
# Creators:
# 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
# 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
# 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
# 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
# Donor:
# David W. Aha (aha '@' ics.uci.edu) (714) 856-8779 
# info: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# 303 patients with chest pain. HD indicates the presence of heart deasease based on an angiographic test ("Yes","No")

load("Heart.rda")
tree.heart<-tree(AHD ~.-X, split = "dev", data= Heart)
summary(tree.heart)
plot(tree.heart)
text(tree.heart,col="red")
cv.tree.heart<-cv.tree(tree.heart,FUN=prune.misclass)
pruned.tree<-prune.misclass(tree.heart,best=8)
plot(pruned.tree,col="lightgrey")
text(pruned.tree,col="purple")

#####
# 4 #
#####
## Predicting election
# Load the data* 
# * data from Kennedy et al. 2017 "Improving election prediction internationally" Science)
# * I have imputed and rescaled some variables
#
load("electiondata.rda")
colSums(is.na(nelda4))
#
# Train/test split
train.nelda <- subset(nelda4, year < 2007)
test.nelda <- subset(nelda4, year >= 2007)
#
## Classificaion Tree
tree.elections<-tree(response ~.-incosWin, split = "dev", data= train.nelda) # gini?
summary(tree.elections)
plot(tree.elections)
text(tree.elections)
set.seed(1243)
cv.tree.elections<-cv.tree(tree.elections,FUN=prune.misclass)
cv.tree.elections$size[which(cv.tree.elections$dev == min(cv.tree.elections$dev))]
plot(cv.tree.elections$size,cv.tree.elections$dev,type = "b")
pruned.tree<-prune.misclass(tree.elections,best=14)
plot(pruned.tree)
text(pruned.tree)
tree.yhat.training<-predict(pruned.tree,train.nelda, type="class")
confusionMatrix(train.nelda$response,tree.yhat.training,positive=levels(train.nelda$response)[2])
"
          Reference
Prediction  No Yes
       No  154  43
       Yes  37 259"
#
## Random Forests
rf.ctrl <-trainControl(method="cv",
                       number=10,
                       verboseIter=TRUE,
                       savePredictions = TRUE)

rf.myGrid<-expand.grid(mtry=seq(1,25,by=1))

rfspace = train(response ~.-incosWin,data= train.nelda, 
                method = "rf",
                trControl = rf.ctrl, 
                tuneGrid = rf.myGrid,
                ntree=500,
                metric="Accuracy")

plot(rfspace)
str(rfspace)
rfspace$results$mtry[which(rfspace$results$Accuracy == max(rfspace$results$Accuracy))]
rf.yhat.training<-predict(rfspace$finalModel,train.nelda)
confusionMatrix(train.nelda$response,rf.yhat.training,positive=levels(train.nelda$response)[2])
"          Reference
Prediction  No Yes
       No  197   0
       Yes   0 296"
rfspace$finalModel
sum(train.nelda$response=="Yes")# 296
"Confusion matrix: 
     No Yes class.error
No  142  55   0.2791878
Yes  44 252   0.1486486"
varImp(rfspace)
plot(varImp(rfspace))
rf.yhat.test<-predict(rfspace$finalModel,test.nelda)
confusionMatrix(test.nelda$response,rf.yhat.test,positive=levels(train.nelda$response)[2])
"         Reference
Prediction No Yes
       No  20  12
       Yes 12  84"