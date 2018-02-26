---
title: "index.Rmd"
author: "Craig Cunningham"
date: "2/15/2018"
output: 
  html_document: 
    keep_md: yes
---



# Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this report, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  In the study six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). These classes are identified the "classe" variable in the data set.

Detailed information is available here: http://groupware.les.inf.puc-rio.br/har#ixzz58425kwpn  
Citations are noted at bottom of the report  

### Libraries
The following libraries were used throughout the code.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(kernlab)
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
library(knitr)
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


### Loading and preprocessing
Download two csv files containing source data from HAR Study 


```r
## data source
source1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
source2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

## set up target data folder
if (!file.exists("data")) {dir.create("data")}

## download files
download.file(source1, destfile = "data/trainingRaw.csv")
download.file(source2, destfile = "data/testingRaw.csv")
dateSourced <- date()
```

Load data into R


```r
## read source data into dataframes 
trainingRaw <- read.csv("data/trainingRaw.csv", na.strings= c("NA",""," "))
testingRaw <- read.csv("data/testingRaw.csv", na.strings= c("NA",""," "))
# str(trainingRaw)
```

Using str() shows several columns without data (NA) so process to subset for only columns with data


```r
## identify empty columns
## note: sum() returns the sum of all values in the argument
nulls <- apply(trainingRaw, 2, function(x) {sum(is.na(x))})

## subset the raw data to remove empty columns
trainingData <- trainingRaw[,which(nulls == 0)]
testingData <- testingRaw[,which(nulls == 0)]
```

... using the str() function shows that the first 7 columns are not measurements  
- therefore subset to data to include only measurements

```r
## subset training data to measurement data only
trainingData <- trainingData[,8:length(trainingData)]
testingData <- testingData[8:length(testingData)]
```


### Set up modeling data
Split the trainingData data into training and validation data sets to use for modeling and validating


```r
## create training and validation tables using classe field
inTrain <- createDataPartition(y = trainingData$classe, p = 0.7, list = FALSE)
training <- trainingData[inTrain,]
validation <- trainingData[-inTrain,]
```

### Find a model to best predict the classe using HAR measurements as a predictor
Random Forests and Boosting models are well suited for assessing predictors in a data set with a large number of independent variables. In this study 3 model techniques were tested:  
1. Using the R package "randomForest" function randomForest()  
2. Using the R "caret" package rf "Random Forest" method  
3. Using the R "caret" package gbm method for "Stochastic Gradient Boosting"  
  
#### Method 1 -- randomForest()
In Random Forest, we’ve collection of decision trees (so known as “Forest”).  To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest).


```r
# fit a model to predict the classe using everything else as a predictor
library(randomForest)
set.seed(344)
fitRF <- randomForest(classe ~ ., data = training)
## - note this method ran in about 5-min
## fitRF 
```


#### Method 2 -- "caret" package rf (Random Forest)


```r
## ------ run the caret version of random forests
## this method is processing intensive therefore using x/y syntax and parallel processing
x<-training[,-53] ## training set without classe
y<-training[,53] ## factor list for classe
## set up parallel processing
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
## configure trainControl object
set.seed(344)
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
## develop the training model
# modFitRF <- train(classe~ .,data=training,method="rf") # ran excess of 40-min then killed
# modFitRF <- train(classe~ .,data=training,method="rf",trControl=fitControl) # non x/y syntax
modFitRF <- train(x,y,data=training,method="rf",trControl=fitControl) # using x/y syntax
## -- note - this method ran in about 8 minutes
## de-register parallel processing cluster
# stopCluster(cluster) ## note will be de-registered after next process
# registerDoSEQ()
## review model and test prediction
# modFitRF # only used in testing
```

#### Method 3 -- "caret" package gbm (Boosting)   
Boosting is actually an ensemble of learning algorithms which combines the prediction of several base estimators in order to improve robustness over a single estimator.  
It combines multiple weak or average predictors to a build strong predictor.


```r
## -------------- Boosting with caret gbm
# library(caret)
## set up parallel processing # NOTE - already set up from previous exercise
# library(parallel)
# library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
## configure trainControl object
set.seed(344)
fitControl <- trainControl(method="repeatedcv",number=4,repeats=4,allowParallel=TRUE)
## Fit model
fitGBM <- train(classe ~ ., data=training, method="gbm", trControl=fitControl,verbose = FALSE)
## --- note - this ran in 8 minutes
## de-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
## now review model and test prediction
# fitGBM # only used in testing
```


### Cross-validation
The models are then used to classify the remaining data and a confusion matrix with actual classifications were used to determine the accuracy of the models and pick a best model.  

#### Test the randomForest model accuracy  

```r
## crossvalidate the model using the validation set
## randomForest()
predRF <- predict(fitRF, validation)
confusionMatrix(validation$classe, predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1136    3    0    0
##          C    0    6 1020    0    0
##          D    0    0    9  955    0
##          E    0    0    0    5 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9961          
##                  95% CI : (0.9941, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9951          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9884   0.9948   1.0000
## Specificity            1.0000   0.9994   0.9988   0.9982   0.9990
## Pos Pred Value         1.0000   0.9974   0.9942   0.9907   0.9954
## Neg Pred Value         1.0000   0.9987   0.9975   0.9990   1.0000
## Prevalence             0.2845   0.1941   0.1754   0.1631   0.1830
## Detection Rate         0.2845   0.1930   0.1733   0.1623   0.1830
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   0.9971   0.9936   0.9965   0.9995
```
----- accuracy .9961 -----------

#### Test the caret rf model accuracy  

```r
## caret rf method
predCaretRF <- predict(modFitRF, validation)
confusionMatrix(validation$classe, predCaretRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1130    4    0    0
##          C    0    6 1018    2    0
##          D    0    0   16  948    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9947   0.9807   0.9948   1.0000
## Specificity            1.0000   0.9981   0.9983   0.9968   0.9994
## Pos Pred Value         1.0000   0.9921   0.9922   0.9834   0.9972
## Neg Pred Value         0.9988   0.9987   0.9959   0.9990   1.0000
## Prevalence             0.2853   0.1930   0.1764   0.1619   0.1833
## Detection Rate         0.2845   0.1920   0.1730   0.1611   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9964   0.9895   0.9958   0.9997
```
----- accuracy .9939 -----------

#### Test the caret boosting model accuracy  

```r
## caret gbm boosting method
predGBM= predict(fitGBM,validation)
confusionMatrix(validation$classe,predGBM)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1647   17    7    1    2
##          B   31 1072   35    1    0
##          C    0   40  971   15    0
##          D    1    6   21  934    2
##          E    3   11    4   19 1045
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9633         
##                  95% CI : (0.9582, 0.968)
##     No Information Rate : 0.2858         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9536         
##  Mcnemar's Test P-Value : 2.22e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9792   0.9354   0.9355   0.9629   0.9962
## Specificity            0.9936   0.9859   0.9887   0.9939   0.9923
## Pos Pred Value         0.9839   0.9412   0.9464   0.9689   0.9658
## Neg Pred Value         0.9917   0.9844   0.9862   0.9927   0.9992
## Prevalence             0.2858   0.1947   0.1764   0.1648   0.1782
## Detection Rate         0.2799   0.1822   0.1650   0.1587   0.1776
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9864   0.9606   0.9621   0.9784   0.9943
```
----- accuracy .9633 -----------

Both the randomForest() and caret rf method models yielded respectyiberly 99.6% and 99.4% prediction accuracy vs. the boosting model's 96.3% accuracy, clearly either random forest model is a better candidate to predict new data.

### Predictions
The randomForest model (fitRF) was then applied to the original "pml-testing" source data set (processed as testingData above) to predict the classifications for the 20 results of this data.


```r
# predict the classes of the test set
predTest <- predict(fitRF, testingData)
predTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Conclusions
Given the measurements provided by the feedback devices outputs during exercise, it is possible to accurately predict whether a person is preforming an excercise properly by using a random forest prediction model. An extension of this would be to apply this model in realtime to give feedback to the user when they are performing an exercise properly.

### Citations
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. _Qualitative Activity Recognition of Weight Lifting Exercises_. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.  
More information can be found here: http://groupware.les.inf.puc-rio.br/har
