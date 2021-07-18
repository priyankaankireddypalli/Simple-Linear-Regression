# 2
library(readr)
# Importing the dataset
deli <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\delivery_time.csv', header = T)
View(deli)
# Performing EDA
summary(deli)
library(Hmisc)
describe(deli)
library(lattice)
# Graphical exploration
# Dotplot
dotplot(deli$Delivery.Time, main = "Dot Plot of Delivery time")
dotplot(deli$Sorting.Time, main = "Dot Plot of Sorting time")
# Boxplot
boxplot(deli$Delivery.Time, col = "dodgerblue4")
boxplot(deli$Sorting.Time, col = "red", horizontal = T)
# Histogram
hist(deli$Delivery.Time)
hist(deli$Sorting.Time)
# Normal QQ plot
qqnorm(deli$Delivery.Time)
qqline(deli$Sorting.Time)
hist(deli$Sorting.Time, prob = TRUE)      # prob=TRUE for probabilities not counts
lines(density(deli$Sorting.Time))             # add a density estimate with defaults
lines(density(deli$Sorting.Time, adjust = 2), lty = "dotted")   # add another "smoother" density
hist(deli$Delivery.Time, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(deli$Delivery.Time))             # add a density estimate with defaults
lines(density(deli$Delivery.Time, adjust = 2), lty = "dotted")   # add another "smoother" density
# Bi-variate analysis
# Scatter plot
plot(deli$Delivery.Time,deli$Sorting.Time, main = "Scatter Plot", col = "Dodgerblue4",col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Delivery.Time", ylab = "Sorting.Time", pch = 20)  
# Correlation Coefficient
cor(deli$Delivery.Time, deli$Sorting.Time)
# Co-variance
cov(deli$Delivery.Time, deli$Sorting.Time)
# Linear Regression model
reg <- lm(deli$Sorting.Time ~ deli$Delivery.Time, data = deli) 
summary(reg)
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)
View(pred)
# GG plot for adding Regression line for data
library(ggplot2)
ggplot(data = deli, aes(deli$Delivery.Time, deli$Sorting.Time) ) + geom_point() + stat_smooth(method = lm, formula = y ~ x)
# Evaluation the model for fitness 
cor(pred$fit, deli$Sorting.Time)
reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse
# Transformation Techniques
# input = log(x); output = y
plot(log(deli$Delivery.Time), deli$Sorting.Time)
cor(log(deli$Delivery.Time), deli$Sorting.Time)
reg_log <- lm(deli$Sorting.Time ~ log(deli$Delivery.Time), data = deli)
summary(reg_log)
confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")
pred <- as.data.frame(pred)
cor(pred$fit, deli$Sorting.Time)
rmse <- sqrt(mean(reg_log$residuals^2))
rmse
# Regression line for data
ggplot(data = deli, aes(log(deli$Delivery.Time), deli$Sorting.Time ) ) + geom_point() + stat_smooth(method = lm, formula = y ~ log(x))
# Log transformation applied on 'y'
# input = x; output = log(y)
plot(deli$Delivery.Time, log(deli$Sorting.Time))
cor(deli$Delivery.Time, log(deli$Sorting.Time))
reg_log1 <- lm(log(deli$Sorting.Time) ~ deli$Delivery.Time, data = deli)
summary(reg_log1)
predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))
# Exponential function
pred <- exp(predlog) 
pred <- as.data.frame(pred)
cor(pred$fit, deli$Sorting.Time)
res_log1 =  deli$Sorting.Time - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse
# Regression line for data
ggplot(data = deli, aes( deli$Delivery.Time, log(deli$Sorting.Time)) ) + geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)
# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)
reg2 <- lm(log(deli$Sorting.Time) ~ deli$Delivery.Time + I(deli$Delivery.Time*deli$Delivery.Time), data = deli)
summary(reg2)
predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit,deli$Sorting.Time)
res2 = deli$Sorting.Time - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
# Regression line for data
ggplot(data = deli, aes( deli$Delivery.Time, log(deli$Sorting.Time)) ) + geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))
# Data Partition
train <- deli[1:10, ]
test <- deli[11:21, ]
log(train$Sorting.Time)
log(test$Sorting.Time)
plot(train$Delivery.Time, log(train$Sorting.Time))
plot(test$Delivery.Time, log(test$Sorting.Time))
model <- lm(log(deli$Sorting.Time) ~ deli$Delivery.Time + I(deli$Delivery.Time * deli$Delivery.Time), data = train)
summary(model)
confint(model,level=0.95)
log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Sorting.Time- predict_original$fit # calculate error/residual
test_error
test_rmse <- sqrt(mean(test_error^2))
test_rmse
log_res_train <- predict(model, interval = "confidence", newdata = train)
predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Sorting.Time - predict_original_train$fit # calculate error/residual
train_error
train_rmse <- sqrt(mean(train_error^2))
train_rmse

