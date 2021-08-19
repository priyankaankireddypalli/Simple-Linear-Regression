# 3
library(readr)
# Importing the dataset
emp <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\emp_data.csv', header = T)
View(emp)
# Performing EDA
summary(emp)
library(Hmisc)
describe(emp)
library(lattice) 
# Graphical exploration
# Dotplot
dotplot(emp$Salary_hike, main = "Dot Plot of Waist Circumferences")
dotplot(emp$Churn_out_rate, main = "Dot Plot of Adipose Tissue Areas")
# Boxplot
boxplot(emp$Salary_hike, col = "dodgerblue4")
boxplot(emp$Churn_out_rate, col = "red", horizontal = T)
# Histogram
hist(emp$Salary_hike)
hist(emp$Churn_out_rate)
# Normal QQ plot
qqnorm(emp$Salary_hike)
qqline(emp$Salary_hike)
qqnorm(emp$Churn_out_rate)
qqline(emp$Churn_out_rate)
hist(emp$Salary_hike, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(emp$Salary_hike))             # add a density estimate with defaults
lines(density(emp$Salary_hike, adjust = 2), lty = "dotted")   # add another "smoother" density
hist(emp$Churn_out_rate, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(emp$Churn_out_rate))             # add a density estimate with defaults
lines(density(emp$Churn_out_rate, adjust = 2), lty = "dotted")   # add another "smoother" density
# Bi-variate analysis
# Scatter plot
plot(emp$Salary_hike,emp$Churn_out_rate, main = "Scatter Plot", col = "Dodgerblue4", col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike", ylab = "Churn_out_rate", pch = 20)  # plot(x,y)
# Correlation Coefficient
cor(emp$Salary_hike, emp$Churn_out_rate)
# Co-variance
cov(emp$Salary_hike, emp$Churn_out_rate)
# Linear Regression model
reg <- lm(emp$Churn_out_rate~ emp$Salary_hike, data = emp)
summary(reg)
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)
View(pred)
# ggplot for adding Regression line for data
library(ggplot2)
ggplot(data = emp, aes(emp$Salary_hike, emp$Churn_out_rate) ) + geom_point() + stat_smooth(method = lm, formula = y ~ x)
# Evaluation the model for fitness 
cor(pred$fit,  emp$Churn_out_rate)
reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse
# Transformation Techniques
# input = log(x); output = y
plot(log(emp$Salary_hike), emp$Churn_out_rate)
cor(log(emp$Salary_hike), emp$Churn_out_rate)
reg_log <- lm(emp$Churn_out_rate ~ log(emp$Salary_hike), data = emp)
summary(reg_log)
confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")
pred <- as.data.frame(pred)
cor(pred$fit, emp$Churn_out_rate)
rmse <- sqrt(mean(reg_log$residuals^2))
rmse
# Regression line for data
ggplot(data = emp, aes(log(emp$Salary_hike), emp$Churn_out_rate) ) + geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Log transformation applied on 'y'
# input = x; output = log(y)
plot(emp$Salary_hike, log(emp$Churn_out_rate))
cor(emp$Salary_hike, log(emp$Churn_out_rate))
reg_log1 <- lm(log(emp$Churn_out_rate) ~ emp$Salary_hike, data = emp)
summary(reg_log1)
predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))
# Exponential function
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit, emp$Churn_out_rate)
res_log1 = emp$Churn_out_rate - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse
# Regression line for data
ggplot(data = emp, aes( emp$Salary_hike, log(emp$Churn_out_rate )) ) + geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)
# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)
reg2 <- lm(log(emp$Churn_out_rate) ~ emp$Salary_hike + I(emp$Salary_hike*emp$Salary_hike), data = emp)
summary(reg2)
predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit, emp$Churn_out_rate)
res2 = emp$Churn_out_rate - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
# Regression line for data
ggplot(data = emp, aes(emp$Salary_hike, log(emp$Churn_out_rate)) ) + geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Data Partition
train <- emp[1:5, ]
test <- emp[6:10, ]
log(train$Churn_out_rate)
log(test$Churn_out_rate)
plot(train$Salary_hike, log(train$Churn_out_rate))
plot(test$Salary_hike, log(test$Churn_out_rate))
model <- lm(log(emp$Churn_out_rate) ~ emp$Salary_hike + I(emp$Salary_hike * emp$Salary_hike), data = train)
summary(model)
confint(model,level=0.95)
log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Churn_out_rate- predict_original$fit # calculate error/residual
test_error
test_rmse <- sqrt(mean(test_error^2))
test_rmse
log_res_train <- predict(model, interval = "confidence", newdata = train)
predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Churn_out_rate - predict_original_train$fit # calculate error/residual
train_error
train_rmse <- sqrt(mean(train_error^2))
train_rmse

