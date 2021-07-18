# 5
library(readr)
# Importing the data
st.at <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\SAT_GPA.csv', header = T)
View(st.at)
# Performing EDA
summary(st.at)
library(Hmisc)
describe(st.at)
library(lattice) 
# Graphical exploration
# Dotplot
dotplot(st.at$SAT_Scores, main = "Dot Plot of Waist Circumferences")
dotplot(st.at$GPA, main = "Dot Plot of Adipose Tissue Areas")
# Boxplot
boxplot(st.at$SAT_Scores, col = "dodgerblue4")
boxplot(st.at$GPA, col = "red", horizontal = T)
# Histogram
hist(st.at$SAT_Scores)
hist(st.at$GPA)
# Normal QQ plot
qqnorm(st.at$SAT_Scores)
qqline(st.at$SAT_Scores)
qqnorm(st.at$GPA)
qqline(st.at$GPA)
hist(st.at$SAT_Scores, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(st.at$SAT_Scores))             # add a density estimate with defaults
lines(density(st.at$SAT_Scores, adjust = 2), lty = "dotted")   # add another "smoother" density
hist(st.at$GPA, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(st.at$GPA))             # add a density estimate with defaults
lines(density(st.at$GPA, adjust = 2), lty = "dotted")   # add another "smoother" density
# Bi-variate analysis
# Scatter plot
plot(st.at$SAT_Scores,st.at$GPA, main = "Scatter Plot", col = "Dodgerblue4", col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary_hike", ylab = "Churn_out_rate", pch = 20)  
# Correlation Coefficient
cor(st.at$SAT_Scores,st.at$GPA)
# Co-variance
cov(st.at$SAT_Scores,st.at$GPA)
# Linear Regression model
reg <- lm(st.at$SAT_Scores~ st.at$GPA, data = st.at) 
summary(reg)
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)
View(pred)
# ggplot for adding Regression line for data
library(ggplot2)
ggplot(data = st.at, aes(st.at$GPA, st.at$GPA) ) + geom_point() + stat_smooth(method = lm, formula = y ~ x)
# Evaluation the model for fitness 
cor(pred$fit,  st.at$GPA)
reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse
# Transformation Techniques
# input = log(x); output = y
plot(log(st.at$SAT_Scores), st.at$GPA)
cor(log(st.at$SAT_Scores), st.at$GPA)
reg_log <- lm(st.at$GPA ~ log(st.at$SAT_Scores), data = st.at)
summary(reg_log)
confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")
pred <- as.data.frame(pred)
cor(pred$fit, st.at$GPA)
rmse <- sqrt(mean(reg_log$residuals^2))
rmse
# Regression line for data
ggplot(data = st.at, aes(log(st.at$SAT_Scores), st.at$GPA) ) + geom_point() + stat_smooth(method = lm, formula = y ~ log(x))
# Log transformation applied on 'y'
# input = x; output = log(y)
plot(st.at$SAT_Scores, log(st.at$GPA))
cor(st.at$SAT_Scores, log(st.at$GPA))
reg_log1 <- lm(log(st.at$GPA) ~ st.at$SAT_Scores, data = st.at)
summary(reg_log1)
predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))
# Exponential function
pred <- exp(predlog)   
pred <- as.data.frame(pred)
cor(pred$fit, st.at$GPA)
res_log1 = st.at$GPA - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse
# Regression line for data
ggplot(data = st.at, aes( st.at$SAT_Scores, log(st.at$GPA )) ) + geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)
# Non-linear models = Polynomial models
# input = x & x^2 (2-degree) and output = log(y)
reg2 <- lm(log(st.at$GPA ) ~ st.at$SAT_Scores + I(st.at$SAT_Scores*st.at$SAT_Scores), data = st.at)
summary(reg2)
predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)
pred <- as.data.frame(pred)
cor(pred$fit,st.at$GPA)
res2 =st.at$GPA - pred$fit
rmse <- sqrt(mean(res2^2))
rmse
# Regression line for data
ggplot(data = st.at, aes(st.at$SAT_Scores, log(st.at$GPA)) ) + geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))
# Data Partition

# Random Sampling
n <- nrow(st.at)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- st.at[train_ind, ]
test <-  st.at[-train_ind, ]

# Non-random sampling
train <- st.at[1:100, ]
test <- st.at[101:200, ]

log(train$GPA)
log(test$GPA)
plot(train$GPA, log(train$SAT_Scores))
plot(test$GPA, log(test$SAT_Scores))
model <- lm(log(st.at$GPA) ~ st.at$SAT_Scores + I(st.at$SAT_Scores * st.at$SAT_Scores), data = train)
summary(model)
confint(model,level=0.95)
log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$GPA- predict_original$fit # calculate error/residual
test_error
test_rmse <- sqrt(mean(test_error^2))
test_rmse
log_res_train <- predict(model, interval = "confidence", newdata = train)
predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$GPA - predict_original_train$fit # calculate error/residual
train_error
train_rmse <- sqrt(mean(train_error^2))
train_rmse

