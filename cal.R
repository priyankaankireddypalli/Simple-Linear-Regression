# 1
library(readr)
# Imporing the dataset
cal <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\calories_consumed.csv', header = T)
View(cal)
# Performing EDA
summary(cal)
install.packages("Hmisc")
library(Hmisc)
describe(cal)
library("lattice") 
# Graphical exploration
# Dotplot
dotplot(cal$Weight.gained..grams., main = "Dot Plot of Waist Circumferences")
dotplot(cal$Calories.Consumed, main = "Dot Plot of Adipose Tissue Areas")
# Boxplot
boxplot(cal$Weight.gained..grams, col = "dodgerblue4")
boxplot(cal$Calories.Consumed, col = "red", horizontal = T)
# Histogram
hist(cal$Weight.gained..grams)
hist(cal$Calories.Consumed)
# Normal QQ plot
qqnorm(cal$Weight.gained..grams)
qqline(cal$Weight.gained..grams)
qqnorm(cal$Calories.Consumed)
qqline(cal$Calories.Consumed)
hist(cal$Weight.gained..grams, prob = TRUE)          # prob = TRUE for probabilities not counts
lines(density(cal$Weight.gained..grams))             # add a density estimate with defaults
lines(density(cal$Weight.gained..grams, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(cal$Calories.Consumed, prob = TRUE)          # prob = TRUE for probabilities not counts
lines(density(cal$Calories.Consumed))             # add a density estimate with defaults
lines(density(cal$Calories.Consumed, adjust = 2), lty = "dotted")   # add another "smoother" density
# Bi-variate analysis
# Scatter plot
plot(cal$Weight.gained..grams,cal$Calories.Consumed, main = "Scatter Plot", col = "Dodgerblue4", col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Weight.gained..grams", ylab = "Calories.Consumed", pch = 20)

# Correlation Coefficient
cor(cal$Weight.gained..grams, cal$Calories.Consumed)
# Co-variance
cov(cal$Weight.gained..grams,cal$Calories.Consumed)

# Linear Regression model
reg <- lm(cal$Calories.Consumed ~ cal$Weight.gained..grams, data = cal) # Y ~ X
summary(reg)
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)
View(pred)

# gg plot for adding Regression line for data
library(ggplot2)
ggplot(data = cal, aes(cal$Weight.gained..grams, cal$Calories.Consumed) ) + geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Evaluation the model for fitness 
cor(pred$fit, cal$Calories.Consumed)
reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse

# Transformation Techniques
# input = log(x); output = y
plot(log(cal$Weight.gained..grams), cal$Calories.Consumed)
cor(log(cal$Weight.gained..grams), cal$Calories.Consumed)
reg_log <- lm(cal$Calories.Consumed ~ log(cal$Weight.gained..grams), data = cal)
summary(reg_log)
confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")
pred <- as.data.frame(pred)
cor(pred$fit, cal$Calories.Consumed)
rmse <- sqrt(mean(reg_log$residuals^2))
rmse
# Regression line for data
ggplot(data = cal, aes(log(cal$Weight.gained..grams), cal$Calories.Consumed) ) + geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Log transformation applied on 'y'
# input = x; output = log(y)
plot(cal$Weight.gained..grams, log(cal$Calories.Consumed))
cor(cal$Weight.gained..grams, log(cal$Calories.Consumed))
reg_log1 <- lm(log(cal$Calories.Consumed) ~ cal$Weight.gained..grams, data = cal)
summary(reg_log1)
predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))
# Exponential function
pred <- exp(predlog)  
pred <- as.data.frame(pred)
cor(pred$fit, cal$Calories.Consumed)
res_log1 = cal$Calories.Consumed - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse
# Regression line for data
ggplot(data = cal, aes(x = cal$Weight.gained..grams, y = log(cal$Calories.Consumed))) + geom_point(color = 'blue') + geom_line(color = 'red', data = cal, aes(x = cal$Weight.gained..grams, y = predlog$fit))
# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)
reg2 <- lm(log(cal$Calories.Consumed) ~ cal$Weight.gained..grams + I(cal$Weight.gained..grams*cal$Weight.gained..grams), data = cal)
summary(reg2)
predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit, cal$Calories.Consumed)
res2 = cal$Calories.Consumed - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
# Regression line for data
ggplot(data = cal, aes(cal$Weight.gained..grams, log(cal$Calories.Consumed)) ) + geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Data Partition
train <- cal[1:7, ]
test <- cal[8:14, ]
log(train$Calories.Consumed)
log(test$Calories.Consumed)
plot(train$Weight.gained..grams., log(train$Calories.Consumed))
plot(test$Weight.gained..grams., log(test$Calories.Consumed))
model <- lm(log(cal$Calories.Consumed) ~ cal$Weight.gained..grams + I(cal$Weight.gained..grams * cal$Weight.gained..grams), data = train)
summary(model)
confint(model,level=0.95)
log_res <- predict(model,interval = "confidence", newdata = test)
# converting log values to original values
predict_original <- exp(log_res) 
predict_original <- as.data.frame(predict_original)
test_error <- test$Calories.Consumed- predict_original$fit # calculate error/residual
test_error
test_rmse <- sqrt(mean(test_error^2))
test_rmse
log_res_train <- predict(model, interval = "confidence", newdata = train)
# converting log values to original values
predict_original_train <- exp(log_res_train) 
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Calories.Consumed - predict_original_train$fit # calculate error/residual
train_error
train_rmse <- sqrt(mean(train_error^2))
train_rmse

