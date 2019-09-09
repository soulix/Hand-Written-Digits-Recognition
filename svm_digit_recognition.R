############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 polynomial Kernel
#  4.3 RBF Kernel
# 5.Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#A classic problem in the field of pattern recognition is that of handwritten
#digit recognition. Suppose that you have an image of a digit submitted by a user
#via a scanner, a tablet, or other digital devices. The goal is to develop a model 
#that can correctly identify the digit (between 0-9) written in an image. 

#####################################################################################

# 2. Data Understanding: 
# Number of Instances: 60,000
# Number of Attributes: 785

#####################################################################################
#3. Data Preparation: 

#Loading Neccessary libraries

rm(list=ls())
library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(readr)

#Loading Data
mnist_train <- read.csv("mnist_train.csv",header = FALSE)
mnist_test<-read.csv("mnist_test.csv",header = FALSE)


#Understanding Dimensions
dim(mnist_train)

#Structure of the dataset
str(mnist_train)

#printing first few rows
head(mnist_train)

#Exploring the data
summary(mnist_train)

#checking missing value
sapply(mnist_train, function(x) sum(is.na(x)))
sapply(mnist_test, function(x) sum(is.na(x)))


#Making our target class to factor
mnist_train$V1<-factor(mnist_train$V1)


#taking 10% sample of the mnist_train dataset
set.seed(1)
train.indices = sample(1:nrow(mnist_train), 0.1*nrow(mnist_train))
train = mnist_train[train.indices, ]

#scaling the train and mnist_test dataset
train[,-1]<-train[,-1]/255
mnist_test[,-1]<- mnist_test[,-1]/255


#####################################################################################

# 4. Model Building

#Constructing Model

#4.1 Using Linear Kernel
Model_linear <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "vanilladot")

Eval_linear<- predict(Model_linear, mnist_test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,mnist_test$V1) #Accuracy : 0.9164   


#4.2 Using Polydot Kernel

Model_poly <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "polydot",
                   kpar= list(degree=2))

Eval_poly<- predict(Model_poly, mnist_test)

#confusion matrix - Polydot Kernel
confusionMatrix(Eval_poly,mnist_test$V1) #Accuracy : 0.9582   


#4.3 Using rbf Kernel
Model_rbf <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")

Eval_rbf<- predict(Model_rbf, mnist_test)

confusionMatrix(Eval_rbf,mnist_test$V1) #Accuracy : 0.952  



############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

#Accuracy: 0.9609989
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 0.025 and C = 2.

plot(fit.svm)


###############################################################################

# Valdiating the model after cross validation on mnist_test data

evaluate_radial_test<- predict(fit.svm, mnist_test)

confusionMatrix(evaluate_radial_test,mnist_test$V1)

#Accuracy : 0.9654   












