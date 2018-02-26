## Project.R

library(caret)
library(kernlab)
library(rpart)
library(corrplot)

# data source
source1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
source2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# set up target data folder
if (!file.exists("data")) {dir.create("data")}

# download files
download.file(source1, destfile = "data/trainingRaw.csv")
download.file(source2, destfile = "data/testingRaw.csv")
## dateDownloaded <- date()

# read source data into dataframes 
trainingRaw <- read.csv("data/trainingRaw.csv", na.strings= c("NA",""," "))
testingRaw <- read.csv("data/testingRaw.csv", na.strings= c("NA",""," "))

## check for empty columns
## note: sum() returns the sum of all values in the argument
nulls <- apply(trainingRaw, 2, function(x) {sum(is.na(x))})

## subset the raw data to remove empty columns
trainingData <- trainingRaw[,which(nulls == 0)]
testingData <- testingRaw[,which(nulls == 0)]

## using str() shows that the first 7 columns are not measurements
# str(trainingData)
## subset training data to measurement data only
trainingData <- trainingData[,8:length(trainingData)]
testingData <- testingData[8:length(testingData)]

## create training and validation tables using classe field
inTrain <- createDataPartition(y = trainingData$classe, p = 0.7, list = FALSE)
training <- trainingData[inTrain,]
validation <- trainingData[-inTrain,]

# plot a correlation matrix <<-- skip - no valuable data here
corMatrix <- cor(training[, -length(training)])
corrplot(corMatrix, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))

# fit a model to predict the classe using everything else as a predictor
## randomForest
## In Random Forest, we’ve collection of decision trees (so known as “Forest”). 
## To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. 
## The forest chooses the classification having the most votes (over all the trees in the forest).
library(randomForest)
fitRF <- randomForest(classe ~ ., data = training)
## - note this method ran in about 5-min
fitRF 
## test prediction
predRF <- predict(fitRF, validation)
confusionMatrix(validation$classe, predRF)

## ------ run the caret version of random forests
## this method is processing intensive therefore using x/y syntax
x<-training[,-53] ## training set without classe
y<-training[,53] ## factor list for classe
## set up parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
## configure trainControl object
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
## develop the training model
# modFitRF <- train(classe~ .,data=training,method="rf") # ran excess of 40-min then killed
# modFitRF <- train(classe~ .,data=training,method="rf",trControl=fitControl) # non x/y syntax
modFitRF <- train(x,y,data=training,method="rf",trControl=fitControl) # using x/y syntax
## -- note - this method ran in about 8 minutes
## de-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
## now review model and test prediction
modFitRF # only used in testing

## test prediction
predFitRF <- predict(modFitRF, validation)
confusionMatrix(validation$classe, predFitRF)

## ------------- Decision Tree <<-- NOT USED
## In this algorithm, we split the population into two or more homogeneous sets.
## might not be appropriate for this data with so many variants
fitDT<-rpart(classe~.,data=training,method="class")
summary(fitDT) # only used in testing
## test prediction using validation dataset
predDT<-predict(fitDT,validation)
confusionMatrix(validation$classe,predDT)

## -------------- Boosting
## Boosting is actually an ensemble of learning algorithms which combines the prediction 
## of several base estimators in order to improve robustness over a single estimator. 
## It combines multiple weak or average predictors to a build strong predictor.
# library(caret)
## set up parallel processing
# library(parallel)
# library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
## configure trainControl object
fitControl <- trainControl(method="repeatedcv",number=4,repeats=4,allowParallel=TRUE)
## Fit model
fitGBM <- train(classe ~ ., data=training, method="gbm", trControl=fitControl,verbose = FALSE)
## --- note - this ran in 8 minutes
## de-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
## now review model and test prediction
summary(fitGBM) # only used in testing
fitGBM # only used in testing

## test prediction using validation dataset
predGBM= predict(fitGBM,validation)
confusionMatrix(validation$classe,predGBM)

# predict the classes of the test set
predTest <- predict(model, testingData)
