#Read in data
setwd("~/Dropbox/Kaggle/Titanic")
df.train <- read.csv("~/Dropbox/Kaggle/Titanic/train.csv")
df.test <- read.csv("~/Dropbox/Kaggle/Titanic/test.csv")
View(df.train)

#Check for missing data
sapply(df.train, function(x) sum(is.na(x)))
"""
-> A lot of missing data on age, proceed with caution
"""

#Understand Dataset
summary(df.train)
"""
-38% Survived...so if you had guessed everyone didn't survive you would already have 
-62% accuracy for the current data set. We defintitely want to do better than 62%
"""

#Plot
barplot(table(df.train$Survived), names.arg = c("Dead", "Survived"), main = "Survival")
barplot(table(df.train$Pclass), names.arg = c("1st", "2nd", "3rd"), main = "Class")
barplot(table(df.train$Sex), names.arg = c("female", "male"), main = "Sex")
barplot(table(df.train$SibSp), main = "Number of Siblings/Parents Aboard")
barplot(table(df.train$Parch), main = "Number of Parent/Children Aboard")
barplot(table(df.train$Embarked), names.arg = c("missing","Cherbourg", "Queenstown", "Southampton"), main = "Location Embarked")
hist(table(df.train$Fare), main = "Fare", xlab = "Price of Fare")
hist(table(df.train$Age), main = "Age")

"""
-Almost double dead compared to survived
-Majority are in 3rd class
-Almost twice as many males than females
-Most of fare was on the lower side
-Most passengers are younger
"""

#Compare Class and Survival
mosaicplot(df.train$Pclass ~ df.train$Survived, 
           main="Survival by Class", 
           color=TRUE, xlab="Pclass", ylab="Survived")

class_survive.table = table(df.train$Survived, df.train$Pclass)
round(prop.table(class_survive.table)*100)/100 #so we get 2 decimal places
round(prop.table(class_survive.table,1)*100)/100 #so we get 2 decimal places
round(prop.table(class_survive.table,2)*100)/100 #so we get 2 decimal places


"""
-Majority of second and third class passengers died 
-Majority of first class survived
-> Likely the higher the class, more likely to get on a lifeboat
"""

#Compare Gender and Survival
mosaicplot(df.train$Sex ~ df.train$Survived, 
           main="Survival by Sex", 
           color=TRUE, xlab="Sex", ylab="Survived")

sex_survive.table = table(df.train$Survived, df.train$Sex)
round(prop.table(sex_survive.table)*100)/100 #so we get 2 decimal places
round(prop.table(sex_survive.table,1)*100)/100 #so we get 2 decimal places
round(prop.table(sex_survive.table,2)*100)/100 #so we get 2 decimal places

"""
-More than 3/4th of men died and more than 3/4th of women survived
-> Likely female prioritized to life boats than male
"""

#Compare Age and Survival
boxplot(df.train$Age ~ df.train$Survived, main = "Survival by Age", ylab = "Age")

"""
No clear difference in age between survived and dead
"""
#Compare Embark and Survival

mosaicplot(df.train$Embarked ~ df.train$Survived, 
           main="Survival by Embarked", 
           color=TRUE, xlab="Departure", ylab="Survived")


embark_survive.table = table(df.train$Survived, df.train$Embarked)
round(prop.table(embark_survive.table)*100)/100 #so we get 2 decimal places
round(prop.table(embark_survive.table,1)*100)/100 #so we get 2 decimal places
round(prop.table(embark_survive.table,2)*100)/100 #so we get 2 decimal places

"""
-Majority who left from Queenstown and Southampton died. And majority came from Southampton!

"""

#Compare Fare and Survival
boxplot(df.train$Fare ~ df.train$Survived, main = "Survival by Age", ylab = "Fare")
"""
-> No immeidate clear difference in dead and survived fare
"""

#Imputation: Age
boxplot(df.train$Age ~ df.train$Pclass)
boxplot(df.train$Age ~ df.train$Embarked)
boxplot(df.train$Age ~ df.train$SibSp)
boxplot(df.train$Age ~ df.train$Parch)
boxplot(df.train$Age ~ df.train$Sex)

#Model to predict age
age.model = lm(Age ~ Pclass + Embarked + SibSp + Parch + Sex, data = df.train)

#Predict age that are missing
age.predict = predict.lm(age.model, df.train[which(is.na(df.train$Age) == TRUE),])

#Set missing age to predicted age
df.train$Age[which(is.na(df.train$Age) == TRUE)] = age.predict

#Models
model_0 = glm(Survived ~ Sex, data = df.train, family=binomial("logit"))
model_1 = glm(Survived ~ Sex + Pclass, data = df.train, family=binomial("logit"))
model_2 = glm(Survived ~ Sex + Pclass + Age, data = df.train, family=binomial("logit"))
model_3 = glm(Survived ~ Sex + Pclass + Age + Embarked, data = df.train, family=binomial("logit"))
model_4 = glm(Survived ~ Sex + Pclass + Age + Embarked + Fare, data = df.train, family=binomial("logit"))




#Predict on df.training Data

#Check for missing in training data
sapply(df.test, function(x) sum(is.na(x)))
"""
Need to predict values for Age and Fare.
Since only 1 data point for fare, we will take average
"""

#Predict age that are missing
age.predict = predict.lm(age.model, df.test[which(is.na(df.test$Age) == TRUE),])
df.test$Age[which(is.na(df.test$Age) == TRUE)] = age.predict

#Set mean to missing fare value
df.test$Fare[which(is.na(df.test$Fare) == TRUE)] = mean(df.test$Fare, na.rm = TRUE)



#Predict on Test Data
pred = predict.lm(model_3, df.test)
pred[which(pred >= 0.5)] = 1
pred[which(pred < 0.5)] = 0
predictions = cbind(df.test$PassengerId, pred)
colnames(predictions) = c("Passengerid", "Survived")
write.csv(predictions, 'predictions_2.csv', row.names = FALSE)
