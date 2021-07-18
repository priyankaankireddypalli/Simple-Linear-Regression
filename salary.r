# 4
library(readr)
# Importing the dataset
sd.at <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\Salary_Data.csv', header = T)
View(sd.at)
# Performing EDA
summary(sd.at)
library(Hmisc)
describe(sd.at)
library("lattice") 
# Graphical exploration
# Dotplot
dotplot(sd.at$YearsExperience, main = "Dot Plot of Waist Circumferences")
dotplot(sd.at$Salary, main = "Dot Plot of Adipose Tissue Areas")
# Boxplot
boxplot(sd.at$YearsExperience, col = "dodgerblue4")
boxplot(sd.at$Salary, col = "red", horizontal = T)
# Histogram
hist(sd.at$YearsExperience)
hist(sd.at$Salary)
# Normal QQ plot
qqnorm(sd.at$YearsExperience)
qqline(sd.at$YearsExperience)
qqnorm(sd.at$Salary)
qqline(sd.at$Salary)
hist(sd.at$Salary, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(sd.at$Salary))             # add a density estimate with defaults
lines(density(sd.at$Salary, adjust = 2), lty = "dotted")   # add another "smoother" density
hist(sd.at$YearsExperience, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(sd.at$YearsExperience))             # add a density estimate with defaults
lines(density(sd.at$YearsExperience, adjust = 2), lty = "dotted")   # add another "smoother" density
# Bi-variate analysis
# Scatter plot
plot(sd.at$Salary,sd.at$YearsExperience, main = "Scatter Plot", col = "Dodgerblue4", col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike", ylab = "Churn_out_rate", pch = 20)  

# Correlation Coefficient
cor(sd.at$Salary, sd.at$YearsExperience)
# Co-variance
cov(sd.at$Salary, sd.at$YearsExperience)
# Linear Regression model
reg <- lm(sd.at$Salary~ sd.at$YearsExperience, data = sd.at) 
summary(reg)
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)
View(pred)
# ggplot for adding Regression line for data
library(ggplot2)
ggplot(data = sd.at, aes(sd.at$YearsExperience, sd.at$YearsExperience) ) + geom_point() + stat_smooth(method = lm, formula = y ~ x)
# Evaluation the model for fitness 
cor(pred$fit,  sd.at$YearsExperience)
reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse
# Transformation Techniques
# input = log(x); output = y
plot(log(sd.at$Salary), sd.at$YearsExperience)
cor(log(sd.at$Salary), sd.at$YearsExperience)
reg_log <- lm(sd.at$YearsExperience ~ log(sd.at$Salary), data = sd.at)
summary(reg_log)
confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")
pred <- as.data.frame(pred)
cor(pred$fit, sd.at$YearsExperience)
rmse <- sqrt(mean(reg_log$residuals^2))
rmse
# Regression line for data
ggplot(data = sd.at, aes(log(sd.at$Salary), sd.at$YearsExperience) ) + geom_point() + stat_smooth(method = lm, formula = y ~ log(x))
# Log transformation applied on 'y'
# input = x; output = log(y)
plot(sd.at$Salary, log(sd.at$YearsExperience))
cor(sd.at$Salary, log(sd.at$YearsExperience))
reg_log1 <- lm(log(sd.at$YearsExperience) ~ sd.at$Salary, data = sd.at)
summary(reg_log1)
predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))
# Exponential function
pred <- exp(predlog)   
pred <- as.data.frame(pred)
cor(pred$fit, sd.at$YearsExperience)
res_log1 = sd.at$YearsExperience - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse
# Regression line for data
ggplot(data = sd.at, aes( sd.at$Salary, log(sd.at$YearsExperience )) ) + geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)
# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)
reg2 <- lm(log(sd.at$YearsExperience ) ~ sd.at$Salary + I(sd.at$Salary*sd.at$Salary), data = sd.at)
summary(reg2)
predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit,sd.at$YearsExperience)
res2 = sd.at$YearsExperience - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
# Regression line for data
ggplot(data = sd.at, aes(sd.at$Salary, log(sd.at$YearsExperience)) ) + geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))
# Data Partition
train <- sd.at[1:15, ]
test <- sd.at[16:30, ]
log(train$YearsExperience)
log(test$YearsExperience)
plot(train$Salary, log(train$YearsExperience))
plot(test$Salary, log(test$YearsExperience))
model <- lm(log(sd.at$YearsExperience) ~ sd.at$Salary + I(sd.at$Salary * sd.at$Salary), data = train)
summary(model)
confint(model,level=0.95)
log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$YearsExperience- predict_original$fit # calculate error/residual
test_error
test_rmse <- sqrt(mean(test_error^2))
test_rmse
log_res_train <- predict(model, interval = "confidence", newdata = train)
predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$YearsExperience - predict_original_train$fit # calculate error/residual
train_error
train_rmse <- sqrt(mean(train_error^2))
train_rmse

