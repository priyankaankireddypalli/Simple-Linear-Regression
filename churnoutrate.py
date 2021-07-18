# 3
import pandas as pd
import numpy as np  
# Importing the dataset
empData = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\emp_data.csv")
# Performing EDA
empData.describe()
# Graphical Representation
import matplotlib.pyplot as plt 
# BArplot
plt.bar(height = empData.Churn_out_rate, x = np.arange(1, 11, 1))
# Histogram
plt.hist(empData.Churn_out_rate) 
# Boxplot
plt.boxplot(empData.Churn_out_rate) 
plt.bar(height = empData.Salary_hike, x = np.arange(1, 11, 1))
plt.hist(empData.Salary_hike) #histogram
plt.boxplot(empData.Salary_hike) #boxplot
# Scatter plot
plt.scatter(x = empData['Salary_hike'], y = empData['Churn_out_rate'], color = 'green') 
# correlation
np.corrcoef(empData.Salary_hike, empData.Churn_out_rate) 
# Covariance
cov_output = np.cov(empData.Salary_hike, empData.Churn_out_rate)[0, 1]
cov_output
empData.cov()
import statsmodels.formula.api as smf
# Simple Linear Regression
model = smf.ols('Churn_out_rate ~ Salary_hike', data = empData).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(empData['Salary_hike']))
# Regression Line
plt.scatter(empData.Salary_hike, empData.Churn_out_rate)
plt.plot(empData.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res1 = empData.Churn_out_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1
# Model building on Transformed Data
# Log Transformation
# x = log(Salary_hike); y = at
plt.scatter(x = np.log(empData['Salary_hike']), y = empData['Churn_out_rate'], color = 'brown')
np.corrcoef(np.log(empData.Salary_hike), empData.Churn_out_rate) #correlation
model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = empData).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(empData['Salary_hike']))
# Regression Line
plt.scatter(np.log(empData.Salary_hike), empData.Churn_out_rate)
plt.plot(np.log(empData.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res2 = empData.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
# x = Salary_hike; y = log(at)
plt.scatter(x = empData['Salary_hike'], y = np.log(empData['Churn_out_rate']), color = 'orange')
np.corrcoef(empData.Salary_hike, np.log(empData.Churn_out_rate)) #correlation
model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = empData).fit()
model3.summary()
pred3 = model3.predict(pd.DataFrame(empData['Salary_hike']))
pred3_at = np.exp(pred3)
pred3_at
# Regression Line
plt.scatter(empData.Salary_hike, np.log(empData.Churn_out_rate))
plt.plot(empData.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res3 = empData.Churn_out_rate - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
# Polynomial transformation
# x = Salary_hike; x^2 = Salary_hike*Salary_hike; y = log(at)
model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike+ I(Salary_hike*Salary_hike)', data = empData).fit()
model4.summary()
pred4 = model4.predict(pd.DataFrame(empData))
pred4_at = np.exp(pred4)
pred4_at
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = empData.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = empData.iloc[:, 1].values
plt.scatter(empData.Salary_hike, np.log(empData.Churn_out_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res4 = empData.Churn_out_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse
# The best model
from sklearn.model_selection import train_test_split
train, test = train_test_split(empData, test_size = 0.3)
finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()
# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate
# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = np.exp(train_pred)
pred_train_Churn_out_rate
# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
