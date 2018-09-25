# Coded by - Sagnik Banerjee
#
# PROBLEM STATEMENT  
# Business Understanding -
# A classic problem in the field of pattern recognition is that of handwritten digit recognition. 
# Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other 
# digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) 
# written in an image. 
#
# APPROACH 
# In any Data Science problem, it is imperative that the business problem is understood extensively & 
# according to the question that is being attempted to be answered using analysis, should the data be 
# collected from all relevant sources. The next steps would involve zeroing on the modelling techniques to 
# be used, trying different similar techniques & trying to monetize the cost of error of the y_pred. This 
# would help us determine how much confidence % we are attempting to achieve as part of the whole exercise.
# 
# Objective - 
# You are required to develop a model using Support Vector Machine which should correctly classify the handwritten 
# digits based on the pixel values given as features.
#
# MODEL & FRAMEWORK SELECTION 
# Since it has been already established in the objective that the model is to be built using SVM, we are not 
# going to try and attempt solve the classification problem using other models that could have been used. 
# It is known that in trying to solve a problem of this kind, a much higher percentage of accuracy can be achieved 
# if a Deep Neural Network is used - https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning
# In the README.md section it can be seen that Deep Neural Networks achieve a beter degree of accuracy on the same data set 
# we are trying to classify. 
# But for the sake of this exercise, we will limit the model selection from amongst these - 
# 1) SVM implementation using a linear kernel
# 2) SVM implementation using a radial kernel
# 3) SVM implementation using a polynomial kernel
# We also intend to try using different values of hyperparameters for each kernel type in addition to the 
# diferent kinds of kernel on trying to zero in on the best possible model for the classification. 
# CRISP Data Mining Framework will be been used in solving the problem at hand. After Loading the data, We will check for 
# data consistency, NA values etc after which EDA will be performed & then Model selection part will come into play.
#
# DATA UNDERSTANDING
# The MNIST database of handwritten digits, has a training set of 60,000 examples,
# and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been 
# size-normalized and centered in a fixed-size image.
# The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while 
# preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique 
# used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of 
# the pixels, and translating the image so as to position this point at the center of the 28x28 field.
# It is a good database for people who want to try learning techniques and pattern recognition methods on real-world 
# data while spending minimal efforts on preprocessing and formatting. - http://yann.lecun.com/exdb/mnist/
# So essentially, the raw handwritten data has been pre-processed to create a matrix of size 28 * 28 resulting in 784 
# features in trying to describe the handwritten digits (0-9). 
# It is well known in converting any form of photo to digitized form, there is always a scope of noise 
# creeping in at the first stage so it is very important to remove this. Since this has also been taken care for us, 
# we can dive right in to the exercise. 


# LOADING ALL REQUIRED LIBRARIES

library(ggplot2)
library(dplyr)
library(caret)
library(readr)
library(caTools)
library(gridExtra)
library(kernlab)

# Adding the next library at a later part of this exercise for allowing parallel processing for achieving a reasonable
# computational time in calculating the crossvalidation of SVM using a radial & Polynomial kernel
# Following are the steps I have used to achieve Parallel Processing in my machine - 
# 1.Load the library -  library(doParallel)
# 2.Then create a cluster with - cl <- makePSOCKcluster(<MAX NO OF CORES TO BE USED>)
# 3.Register the cluster with - registerDoParallel(cl)
# 4.Write code that needs parallel processing.In your cross validation train call, you can use allowparallel=TRUE
# 5.Stop cluster after you are done - stopCluster(cl)

library(doParallel)
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# Step 4,5 comes at the point in time where training the respective model is done

# DATA LOADING & MANIPULATION BEGINS HERE
# Working directory needs to be set to the appropriate directory containing the train & test data using
# setwd function

train_initial <- read.csv("mnist_train.csv", stringsAsFactors = FALSE, header = FALSE)
test_initial <- read.csv("mnist_test.csv", stringsAsFactors = FALSE, header = FALSE)

View(train_initial)
View(test_initial)

# Train_initial & test_initial datasets have 785 variables each.
# in addition to the 28*28 = 784 variables, there is one additional variable that tells us for which digit are the 
# feature vectors respectively
# We deliberately imported the datasets into R without the headers 
# However, emperically we can say that we need a name of the column for which we are doing the classification
# for ease of writing the model function

names(train_initial)[1] <- "label_pred"
names(test_initial)[1] <- "label_pred"

# CHECK FOR NA VALUES & DUPLICATE ROWS AND OTHER DATA PREPARATION CHECKS

sum(is.na.data.frame(train_initial))
sum(is.na.data.frame(test_initial))

sum(duplicated.data.frame(train_initial))
sum(duplicated.data.frame(test_initial))

# It can now be established that there is no NA value in any Row or there are no duplicate Rows in both the 
# test as well as the train datasets.

summary(factor(train_initial$label_pred))
summary(factor(test_initial$label_pred))  

# After checking for the predicting labels in both the datasets, we may say that no label has been 
# skipped & the labels in both the datsets do match

summary(train_initial[, 2:150])
summary(test_initial[, 2:150])

# testing for max, mean & median values, it is found that in train dataset, some of the max Pixel values run upto 
# only 100 or below it while a number of variables in test where max Pixel values run upto 255.
# hence, scaling is a must for the test dataset. This shall be done post the sampling stage where the data which will go into
# the actual model building will be selected

str(train_initial)
train_initial$label_pred <- factor(train_initial$label_pred)
str(test_initial)
test_initial$label_pred <- factor(test_initial$label_pred)

# structure of the datasets is checked to ensure that all the variables being considered are of the integer type
# it is found that both test & train type datasets have all the variables that are of integer types
# hence, we changed the data type of the 'label_pred' column to factor 

# SAMPLING & SCALING FINAL SET OF TEST & TRAIN DATA
# The actual datasets provided are huge. With limited computational power, it will be really difficult for us to 
# train & test my models on such a huge dataset. We know that it is suggested to take 15% of the dataset but even that seems to 
# be computationally taxing on my machine. Which is why I have went ahead with 5000 records instead of the suggested 9000
# records. I hope my marks are not cut because of this as I believe I have been able to execute all aspects of the model building process
# and the selection methodolgy as well properly.

set.seed(100)
sample_indices <- sample(1: nrow(train_initial), 5000)
train_final <- train_initial[sample_indices, ]
View(train_final)


max(train_final[ ,c(2:745)]) # 255
train_final[ , c(2:745)] <- train_final[ , c(2:745)]/255

# the max value found in the variabe columns is 255 and hence we decided to 
# scale the train dataset with that value


test_final[ , c(2:785)] <- test_initial[ , c(2:785)]/255

# the scaling of data is a very important step here as the SVM classifiers are constructed on the basis of the 
# Support Vectors & the classifiers are built on these support vectors 

# EDA 
# We did EDA to check if the distribution pattern of the NUMBERS in the column - label_pred
# are similar in train_initial, train_final & test_fianl data sets. A good sample will hold on to the 
# patterns actually exhibited by the population & as it turns out, the distribution indeed follows a 
# similar pattern not only for the sampled Train dataset, but also for the test data set.

plot1 <- ggplot(train_final,aes(x = factor(label_pred),fill=factor(label_pred)))+geom_bar()+ 
  xlab("HANDWRITTEN NUMBER") +
  ylab("COuNT OF SAMPLES")+ 
  ggtitle("DISTRIBUTION SHOWING COUNT OF SAMPLES OF EACH NUMBER IN TRAIN DATASET")

plot1

plot2 <- ggplot(train_initial,aes(x = factor(label_pred),fill=factor(label_pred)))+geom_bar()+ 
  xlab("HANDWRITTEN NUMBER") +
  ylab("COuNT OF SAMPLES")+ 
  ggtitle("DISTRIBUTION SHOWING COUNT OF SAMPLES OF EACH NUMBER IN TRAIN DATASET")

plot2

plot3 <- ggplot(test_final,aes(x = factor(label),fill=factor(label)))+geom_bar()+ 
  xlab("HANDWRITTEN NUMBER") +
  ylab("COuNT OF SAMPLES")+ 
  ggtitle("DISTRIBUTION SHOWING COUNT OF SAMPLES OF EACH NUMBER IN TEST DATASET")

plot3

# MODEL BUILDING BEGINS HERE
# We will try to find the best model for SVM by first trying out Linear SVM with it's default parameter values

model_linear1 <- ksvm(label_pred ~ ., data = train_final, scaled = FALSE,  C = 1)
print(model_linear1) 

eval_linear1 <- predict(model_linear1, newdata = test_final, type = "response")
confusionMatrix(eval_linear1, test_final$label_pred) 

# Summary of results of a Linear KSVM applied to the data set - 
# Accuracy - 94.7%
# Specifity => 99% in all classes. Lowest - 99.15
# Sensitivity => 91% in all classes. Lowest - 91.34
# Number of Support Vectors - 2544 
# cost C =1
# Training Error = .025

model_linear2 <- ksvm(label_pred ~ ., data = train_final, scaled = FALSE, kernel = "vanilladot",  C = 1)
print(model_linear2)

eval_linear2 <- predict(model_linear2, newdata = test_final, type = "response")
confusionMatrix(eval_linear2, test_final$label_pred) 

# Summary of results of a Linear KSVM applied to the data set alongwith vanilladot kernel- 
# Accuracy - 91%
# Specifity => 98% in all classes. Lowest - 98.44
# Sensitivity => 83% in all classes. Lowest - 83.37
# Number of Support Vectors - 1794 
# cost C =1
# Training Error = 2e-04 = .036

# on comaring the summary of both the models of linear type, it seems model_linear1 is trying to 
# overfit the classifier in this case. so, we decide to go ahead with model building using model_linear2
# next step would be decide the optimum C value. Decide to check with c = 10

model_linear3 <- ksvm(label_pred ~ ., data = train_final, scaled = FALSE, kernel = "vanilladot", C = 10)
print(model_linear3) 

eval_linear3 <- predict(model_linear3, newdata = test_final, type = "response")
confusionMatrix(eval_linear3, test_final$label_pred) 

# Summary of results of a Linear KSVM applied to the data set for c= 10 - 
# Accuracy - 91%
# Specifity => 98% in all classes. Lowest - 98.44
# Sensitivity => 86% in all classes. Lowest - 86.72
# Number of Support Vectors - 1796 
# cost C =1
# Training Error = 2e-04 = 0.0002

# here, are model has overfit it is quite clear. with 0 training error, this model is totally unacceptable
# specificity value has gone up as well. In order to finally decide upon a acceptable model, we will try 
# to figure out C value using cross validation.We will provide different values of C which are in multiples of 10
# and then print the results & plot them to come up with the suitable C value.
# Cross validation of data to happen in 5 folds.

grid_linear <- expand.grid(C= c(0.001, 0.1 ,1 ,10 ,100)) 

fit.linear <- train(label_pred ~ ., data = train_final, metric = "Accuracy", method = "svmLinear",
                    tuneGrid = grid_linear, preProcess = NULL,
                    trControl = trainControl(method = "cv", number = 5))


print(fit.linear) 
plot(fit.linear)

# Best accuracy is found to be for 1e-01 that translates to 0.1 
# of 91.5% post which the accuracy value degrades.It has already been established that for a high value of C
# the model overfits. The lower values of C gives a very simple value 

linear_final_model <- predict(fit.linear, newdata = test_final)
confusionMatrix(linear_final_model, test_final$label)

# Summary of results of a Linear KSVM applied to the data set for c= 0.1
# Accuracy - 92.3 % 
# Specifity => 98% in all classes. Lowest - 98.75
# Sensitivity => 86% in all classes. Lowest - 86.45
# cost C = 0.1

# we choose this as our model for comparison for the representation of linear ksvm

#  We will try to find the best model for SVM by first trying out SVM with a Radial kernel it's default parameter values

rbf_model1 <- ksvm(label_pred ~ ., data = train_final, scaled = FALSE, kernel = "rbfdot", C = 1, kpar = "automatic")
print(rbf_model1) 

eval_rbf1 <- predict(rbf_model1, newdata = test_final, type = "response")
confusionMatrix(eval_rbf1, test_final$label_pred) 

# Summary of results of KSVM applied for a radial to the data set for default parameters - 
# Accuracy - 94.6%
# Specifity => 99% in all classes. Lowest - 99.15
# Sensitivity => 91% in all classes. Lowest - 91.28
# Number of Support Vectors - 2538 
# cost C =1, sigma = 0.001
# Training Error = 0.025
 
# We will need to do a cross validation to find optimum sigma & c values. However, before that we should
# be able to gauge sigma & c boundary conditions to be used in the grid.rbf condition.
# In this step, after learning from my peers in discussion forums & whatsapp groups that cross validation is 
# taking a huge computational toll on the machines, we decided to implement Parallel processing with the 2 cores of 
# the machine. Also, we tried to keep a smaller grid size for achieving lower computational time.* 
# In trying to decide an upper boundary for sigma, we went ahead & tried wit ha pretty high sigma value of 1 

model_rbf2 <- ksvm(label_pred ~ ., data = train_final, scaled = FALSE, kernel = "rbfdot",
                   C = 1, kpar = list(sigma = 1))
print(model_rbf2) 

eval_rbf2 <- predict(model_rbf2, newdata = test_final, type = "response")
confusionMatrix(eval_rbf2, test_final$label_pred) 

# with a training error value = 0 & number of support vectors shooting up to 5000 (almost twice the value of the previous
# model & all of the training data) we can clearly tell that it is a classic case of overfitting
# Accuracy stoops down to 11%
# We thus need to bring it down a few notches to achieve a generalisable model
# So with reference to *, we will try to find a good fit to the model using the following C & sigma values - 
# sigma = 0.001, 0.01 , 0.1 & c = 0.1, 1, 5 for 2 folds cross validation

grid_rbf = expand.grid(C= c(0.1, 1, 5), sigma = c(0.001, 0.01 , 0.1))


fit.rbf1 <- train(label_pred ~ ., data = train_final, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                 trControl = trainControl(method = "cv", number = 2, allowParallel = TRUE), preProcess = NULL)


print(fit.rbf1) 
plot(fit.rbf1)

# Checking the Accuracy & looking at the plot, we are convinced that sigma works best for the value 0.01
# every value other than that has a very poor performance. 
# Upon deciding on this sigma value, we will now try to vary the value of c & check for which value do we 
# get a fairly generalisable & snug fit model. It is observed that values in between 1 & 5 gives us a 
# fairly accurate model. Thus we would want to try values of c in between this range in an effort to optimize c

grid_rbf = expand.grid(C= seq(1,5,by =1), sigma = 0.01)


fit.rbf2 <- train(label_pred ~ ., data = train_final, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                  trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE), preProcess = NULL)


print(fit.rbf2) 
plot(fit.rbf2)

# with the Accuracy values in front of us, we can say that for c = 3 we get the best generalised model. 
# Post that, the model seems to over fit & below that the model is too simple 
# With Accracy value as 93.5%, we select this model as our final model for radial type kernels
# having parameters as c = 3 & sigma = .01 found by by doing 2 fold cross validation
# now let's check for the specificity & sensitivity values through confusion matrix 

rbf_final_model <- predict(fit.rbf2, newdata = test_final)
confusionMatrix(rbf_final_model, test_final$label_pred)


# Summary of results of a Linear KSVM applied to the data set for c= 0.1
# Accuracy - 95.4 % 
# Specifity => 99% in all classes. Lowest - 99.29
# Sensitivity => 92% in all classes. Lowest - 92.47
# cost C = 0.1

# The accuracy found on the test data is easily comparable to that found in the train dataset.
# we choose this as our model for comparison for the representation of ksvm with a kernel of radial type


# Now that we are done testing linear kernel & radial kernel SVM on our data, we now finally look at Polynomial kernels
# This is the last type of kernel that we have tested in trying to find the best fit model for this classification problem

# We first try to use Polynomial kernel with degree 2, a default offset and scale values 

model_poly1 <- ksvm(label_pred ~ ., data = train_final, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = 1, offset = 1))
print(model_poly1)

eval_poly1 <- predict(model_poly1, newdata = test_final)
confusionMatrix(eval_poly1, test_final$label_pred)

# Summary of results of a Polynomial KSVM applied to the data set for c= 1
# Accuracy - 95%
# Specifity => 99% in all classes. Lowest - 99.19
# Sensitivity => 91% in all classes. Lowest - 91.97
# Number of Support Vectors - 1778 
# cost C =1
# Training Error = 0

# We see that this model & our model made with radial kernel is similar in terms of accuracy  
# We will try to see if that can be improved while still keeping the model general

# Polynomial kernel with varied scale

model_poly2 <- ksvm(label_pred ~ ., data = train_final, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = -2, offset = 1))
print(model_poly2)


eval_poly2 <- predict(model_poly2, newdata = test_final)
confusionMatrix(eval_poly2, test_final$label_pred)

# Accuracy falls slightly but a similar type of performance is observed.
# next, the offset value will be changed

model_poly3 <- ksvm(label_pred~ ., data = train_final, kernel = "polydot", scaled = FALSE, C = 1, 
                    kpar = list(degree = 2, scale = 1, offset = 10))
print(model_poly3)

eval_poly3 <- predict(model_poly3, newdata = test_final)
confusionMatrix(eval_poly3, test_final$label_pred)

# Again the accuracy is in the range of 95% and it seems offset doesn't have much significant impact 
# on the model. Lastly, we will gauge the impact that different values of C will have on the model


model_poly4 <- ksvm(label_pred ~ ., data = train_final, kernel = "polydot", scaled = FALSE, C = 3, 
                    kpar = list(degree = 2, scale = 1, offset = 1))
print(model_poly4)

eval_poly4 <- predict(model_poly4, newdata = test_final)
confusionMatrix(eval_poly4, test_final$label_pred)

# No observed significant changes in the model
# lastly, we will perform a grid search over varied range of Cost, Sigma & scale parameters & run it over 3 folds 
# of data cross validation 


grid_poly = expand.grid(C= c(0.01, 0.1, 1), degree = c(1, 2, 3), 
                        scale = c(-10, -1, 1, 10))

fit.poly <- train(label_pred ~ ., data = train_final, metric = "Accuracy", method = "svmPoly",tuneGrid = grid_poly,
                  trControl = trainControl(method = "cv", number = 2, allowParallel = TRUE), preProcess = NULL)

# printing results of cross validation
print(fit.poly) 
plot(fit.poly)

eval_poly <- predict(fit.poly, newdata = test_final)
confusionMatrix(eval_poly, test_final$label_pred)

# We see the best model is obtained for C = 0.01, degree = 2, scale = 1
# as the data has already been scaled, scale = 1 is optimum
# C has little to no effect on perfomance. We decide upon C = 0.01 as optimum model.
# Accuracy of 93.36%, sensitivities > 92%, specificities > 98%


## Implementing optmised polynomial model 
model_poly5 <- ksvm(label_pred ~ ., data = train_final, kernel = "polydot", scaled = FALSE, C = 0.01, 
                    kpar = list(degree = 2, scale = 1, offset = 0.5))
print(model_poly5)

eval_poly5 <- predict(model_poly5, newdata = test_final)
confusionMatrix(eval_poly5, test_final$label_pred)


# we see offset of 0.5 used as independent variables are in the range of 0 to 1
# best accuracy of polynomial kernels 95%

stopCluster(cl)

# Now we no longer need parallel processing

# DRAWING CONCLUSIONS & END NOTE
# Thus at the end of the model building exercide, we choose to go with the Radial Kernel Model. 
# Although can be said that the accuracy & other figures that are a yardstick of measurement of a 
# model are comparable in both the models,is marginally better for the radial kernel. Also weknow that 
# a Radia Kernel SVM is a simpler model & if we can achieve same accuracy for a problem with a simpler model,
# we should go with it always as this model is more genralisable. Testing on test data is fine but if both the
# models are subjected to unseen data of other hand written numbers, it can be said with a greater confidence that
# the radial model shall perform better. 

# FINAL MODEL - 
 final_model = fit.rbf2 
# Features of our final model
# Accuracy - 95.4 % 
# Specifity => 99% in all classes. Lowest - 99.29
# Sensitivity => 92% in all classes. Lowest - 92.47
# cost C = 0.1
 



