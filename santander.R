##########     PROJECT:1 Santander Customer Transaction Prediction  Project     ##############

#Remove the elements
rm(list = ls())



#Set working directory
setwd("c:/rstudio")

#Check working directory
getwd()
 
#loading libaries foe operations
library("ggplot2")
library("scales")
library("psych")
library("gplots")
library("corrgram")
library("DataCombine")
library("randomForest")
library("splitstackshape")
library("caret")
library("recipes")
library("e1071")


# loading datasets
train = read.csv("train.csv",header=T)


###################  Exploratory data analysis ###########################

#Getting the number of variables and obervation in the datasets
dim(train)

# Structure of data
str(train)


#Summary of datasets
summary(train)


#changing datatype of target variable to factor datatype.
train$target= as.factor(train$target)
class(train$target)

#Percenatge counts of target classes
table(train$target)/length(train$target)*100

#We have a unbalanced data,where 90% of the data is the data of number of customers those did not make a transaction and 10% of the data is those who  made a transaction.


#take subset by removing ID code
train = subset(train,select = -c(ID_code))




#########################  DATA PREPROCESSING     ###########################################

######################## Missing Values Analysis #####################################

#checking for missing values
sum(is.na(train))

# No missing values present so, we can move ahead.(no need to run whole process)

#visualization

ggplot(train, aes_string(x = train$target)) +
  geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
  xlab("target") + ylab('Count') + scale_y_continuous(breaks=pretty_breaks(n=10)) +
  ggtitle("santander transaction") +  theme(text=element_text(size=15))

############################## OUTLIER ANALYSIS ##########################################

#boxplot

boxplot(train$var_21,
        main = "Boxplot for var_21",
        xlab = "",
        ylab = "var_21",
        col = "orange",
        border = "brown",
        horizontal = FALSE,
        notch = FALSE
)


#selecting only numeric
numeric_index = sapply(train,is.numeric)

#subset of numeric data
numeric_data = train[,numeric_index]

#saving the column names of numeric data
cnames = colnames(numeric_data) 

#remove outliers

for(i in cnames){
  print(i)
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  #print(length(val))
  train = train[which(!train[,i] %in% val),]
}

############### Feature Selection ###############

#selecting only numeric
numeric_index = sapply(train,is.numeric)

#subset of numeric data
numeric_data = train[,numeric_index]

#correlation plot

corrgram(train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main="Correlation plot")


cor_mat = cor(numeric_data)
cor_mat = round(cor_mat, 2)

#here, we can see that no dependencies between two independent variable.so all variables need to be considered.


############## Feature Scaling ###############


#normality check
hist(train$var_21)

#data found to be normally distributed

#to check range before standardisation
train_num = subset(train, select=-target)   #subset of numeric data
range(train_num)

#saving the column names of numeric data
cnames = colnames(train_num) 

#standardisation

for (i in cnames){
  print(i)
  train[,i] = (train[,i] - mean(train[,i])) / sd(train[,i])
}

#to check range after standardisation

train_num = subset(train, select=-target)   #subset of numeric data
range(train_num)


######### Modelling ################

#Clean the environment
rmExcept("train")

df = train

#divide into train & test

train_index = sample(1:nrow(df), 0.8 * nrow(df))
train_df = df[train_index,]
test = df[-train_index,]

#### Logistic regression ####

logit_model = glm(target ~ ., data = train_df, family = "binomial")

summary(logit_model)

#prediction with probabilities
logit_predictions = predict(logit_model, newdata = test, type = "response")

#prediction into  0 & 1
logit_predictions = ifelse(logit_predictions > 0.5, 1, 0)

#confusion matrix
confmatrix = table(test$target, logit_predictions)

confmatrix

#Accuracy = (TN+TP)/(TN+FP+TP+FN)

#FNR = FN/(FN+TP) 

#Accuracy = 91.78
#FNR = 73.56


#### Random Forest #####


#stratified sampling with 10% data


train_strat = stratified(df, c('target'), 0.2)
test_strat = stratified(df, c('target'), 0.2)



#modelling
RF_model = randomForest(target ~ ., train_strat, importance = TRUE, ntree = 100)

#predictions
RF_predictions = predict(RF_model, test_strat[,-1])

#confusion matrix
ConfMatrix_RF = table(test_strat$target, RF_predictions)


#Accuracy = 92.23
#FNR = 79.50

##### naive bayes #####

NB_model = naiveBayes(target ~ ., data = train_df)

#predictions
NB_predictions = predict(NB_model, test[,2:201], type = 'class')

#confusion matrix
Confmatrix_NB = table(observed = test[,1], predicted = NB_predictions)



#Accuracy = 92.46
#FNR = 64.79

#Here we can see that naive bayes performs well among all models.so we will freeze naive bayes.


################### prediction on test.csv data ###################

#load large test data
santander = read.csv("test.csv", header = T)

#structure of data
str(santander)

#take subset by removing ID CODE
ID_code = subset(santander, select=ID_code)
santander = subset(santander,select = -c(ID_code))


############### Missing Value Analysis ###############

#checking for missing values
sum(is.na(santander))

#we dont have any missing value in data so no needed to do missing value process.


############## Feature Scaling ###############


#normality check
hist(santander$var_23)

#data is normally distributed

#to check range before standardisation
range(santander)

#saving the column names of numeric data
cnames = colnames(santander) 


#standardisation
for (i in cnames){
  print(i)
  santander[,i] = (santander[,i] - mean(santander[,i])) / sd(santander[,i])
}


#to check range after standardisation
range(santander)

##### Prediction #######

#predictions
NB_predictions_test = predict(NB_model, santander[,1:200], type = 'class')

#save predictions as dataframe
NB_predictions_test = as.data.frame(NB_predictions_test)

#columnbind target results with ID_code
ID_code = cbind(ID_code,NB_predictions_test)

#renaming column
names(ID_code)[2] = "Target_value"

#saving output in csv format
write.csv(ID_code, "Final Target value - R.csv", row.names = F)


