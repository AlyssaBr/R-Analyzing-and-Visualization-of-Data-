################################################################################
### Alyssa Brunen
################################################################################
### Individual Assignment 
################################################################################
### Visualizing and Analyzing Data in R  
################################################################################
### A2: Binary Classification with a Bank Churn Dataset 
################################################################################
### Hult International Business School San Francisco Spring 2024 
################################################################################

#Loading Bank Churn Dataset 
df.train <- read.csv("/Users/alyssabrunen/Desktop/Hult Docs/R /BankChurnDataset (2).csv",na.strings=c(""))

#Loading Libraries
library(Amelia)
library(ggplot2)
library(dplyr)
library(caTools)
library(caret)


################################################################################
################################################################################
## Part 1: Handeling NA Values
################################################################################
################################################################################

#Checking for NA values
df.train[is.na(df.train$Age),]
df.train[is.na(df.train$EstimatedSalary),]

# Fixing Estimated Salary column missing values
a2 <- median(df.train$EstimatedSalary, na.rm=T)
df.train[4,13]<- a2
df.train[14,13]<- a2
df.train[29930,13]<- a2

#Age Imputation, handling missing values in age 
median(df.train$Age, na.rm=T)  
  impute_age <- function(Age,Exited){
    out <- Age
    for (i in 1:length(Age)){
      
      if (is.na(Age[i])){
        
        if (Exited[i] == 0){
          out[i] <- 36
          
        }else if (Exited[i] == 1){
          out[i] <- 44
          
          
        }else{
          out[i] <- 37
        }
      }else{
        out[i]<-Age[i]
      }
    }
    return(out)
  }  


#applying the imputing formula
fixed.ages <- impute_age(df.train$Age,df.train$Exited)
df.train$Age <- fixed.ages #overwriting the initial data with the trained data

# Checking if missing values are handled well
missmap(df.train, main="BankChurn Missing Data", 
        col=c("yellow", "black"), legend=FALSE)


################################################################################
################################################################################
## Part 2: Accuracy, Sensitivity and Specificity (Bank Churn)
################################################################################
################################################################################


# removing unwanted columns
df.train <- df.train %>% select (-id, -CustomerId, -Surname)

# checking remaining columns 
head(df.train,3)
str(df.train)

# training the model using logistic regression
log.model <- glm(formula=Exited ~ . , family = binomial(link='logit'),data = df.train)
summary(log.model)

#Making a test set out of training training set 
set.seed(101)

#splitting the dataset into training and testing with a split ratio of 70%
split = sample.split(df.train$Exited, SplitRatio = 0.70)

#Subsetting the two datasets for testing and training 
final.train = subset(df.train, split == TRUE)  #training dataset (70%) that I will use for the final F function 
final.test = subset(df.train, split == FALSE)  #testing dataset (rest of the 30% in the dataset)

#F-function related to the final training model 
final.log.model <- glm(formula=Exited ~ ., family = binomial(link='logit'),data = final.train) #glm= generalized linear model
summary(final.log.model)

#Predicting values 
fitted.probabilities <- predict(final.log.model,newdata=final.test,type='response')
fitted.results <- ifelse(fitted.probabilities > 0.5,1,0)

#Calculating Accuracy for the Bank Churn Model
misClasificError <- mean(fitted.results != final.test$Exited)
print(paste('Accuracy',1-misClasificError))  #printing the accuracy as: 1- mis-classification error rate

# Creating Confusion Matrix and table the values
table(final.test$Exited, fitted.results)

#Factoring the Exited column
final.test$Exited <- factor(final.test$Exited)
fitted.results <- factor(fitted.results)

# Calculating Sensitivity and Specificity 
sensitivity(final.test$Exited, fitted.results)
specificity(final.test$Exited, fitted.results)


################################################################################
################################################################################
## Part 3: New Customer Dataset Prediction 
################################################################################
################################################################################


#Loading New Customer Dataset 
NewCust <- read.csv("/Users/alyssabrunen/Desktop/Hult Docs/R /NewCustomerDataset.csv")

#Checking for NA Values
NewCust[is.na(NewCust)]

# Excluding columns from dataset 
NewCust<- select(NewCust, -CustomerId, -id)

# Prediction Model for the New Customer Dataset
fitted.probs <- predict(final.log.model, newdata=NewCust, type='response')
fitted.res <- ifelse(fitted.probs > 0.5,1,0)

# Accuracy for New Customer Data set 
misClasificErrornewActive <- mean(fitted.res != NewCust$IsActiveMember)
print(paste('Accuracy',1-misClasificErrornewActive))

# Accuracy for New Customer Data set 
misClasificErrornewBalance <- mean(fitted.res != NewCust$Balance)
print(paste('Accuracy',1-misClasificErrornewBalance))

################################################################################
################################################################################
## Analysis and Visualization 
################################################################################
################################################################################
pl <- ggplot(df.train,aes(Exited,Age)) + geom_boxplot(aes(group=Exited,fill=factor(Exited),alpha=0.4))
pl + scale_y_continuous(breaks = seq(min(0), max(80), by = 2))

#* The visualization above shows that it is more common for customers of higher age 
#* groups to exit the company and cancel their membership and would churn. 

#* Accuracy is at 42.92% for the New Customer Data set for the IsActiveMember column 
#* which means that even though one has an active card, it does not mean that the 
#* customer is actively using it. 
#* Accuracy is at 51% for the new Customer Data set for the Balance column 
#* which means that the higher the balance, it shows that the customer is 
#* using their card, which means that they acquire a higher balance on their card. 

#* On the other hand, the accuracy for the bank churn model is at 83%, 
#* which means that the model is highly accurate for the bank churn model. 
#* The sensitivity and specificity scores are also relatively high (85% and 70% respectively).
#* A high sensitivity indicates that the test is effective at capturing most of 
#* the actual positive cases, minimizing false negatives, 
#* while a specificity suggests that the test is proficient at correctly excluding 
#* most of the actual negative cases, minimizing false positives. 

#* Due to the difference in accuracy between the Bank Churn Database and the New Customer
#* data base, it is apparent that the prediction model was made specifically for one of the data sets,
#* therefore yielding a lower accuracy for the New Customer data set. 
#* In order to get a higher accuracy for the new customer data set, a new prediction model
#* would need to be created. 