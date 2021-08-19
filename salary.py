# 4
import pandas as pd 
import numpy as np  
# importing the dataset
salary_data = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\Salary_Data.csv")
# Performing EDA
salary_data.describe()
# Graphical Representation
import matplotlib.pyplot as plt 
# Barplot
plt.bar(height = salary_data.Salary, x = np.arange(1, 31, 1))
# Histogram
plt.hist(salary_data.Salary) 
# Boxplot
plt.boxplot(salary_data.Salary) 
plt.bar(height = salary_data.YearsExperience, x = np.arange(1, 31, 1))
plt.hist(salary_data.YearsExperience) #histogram
plt.boxplot(salary_data.YearsExperience) #boxplot
# Scatter plot
plt.scatter(x = salary_data['YearsExperience'], y = salary_data['Salary'], color = 'green') 
# correlation
np.corrcoef(salary_data.YearsExperience, salary_data.Salary) 
# Covariance
cov_output = np.cov(salary_data.YearsExperience, salary_data.Salary)[0, 1]
cov_output
salary_data.cov()
import statsmodels.formula.api as smf
# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = salary_data).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(salary_data['YearsExperience']))
# Regression Line
plt.scatter(salary_data.YearsExperience, salary_data.Salary)
plt.plot(salary_data.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res1 = salary_data.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1
# Model building on Transformed Data
# Log Transformation
# x = log(YearsExperience); y = Salary
plt.scatter(x = np.log(salary_data['YearsExperience']), y = salary_data['Salary'], color = 'brown')
np.corrcoef(np.log(salary_data.YearsExperience), salary_data.Salary) #correlation
model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = salary_data).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(salary_data['YearsExperience']))
# Regression Line
plt.scatter(np.log(salary_data.YearsExperience), salary_data.Salary)
plt.plot(np.log(salary_data.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res2 = salary_data.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
# x = YearsExperience; y = log(Salary)
plt.scatter(x = salary_data['YearsExperience'], y = np.log(salary_data['Salary']), color = 'orange')
np.corrcoef(salary_data.YearsExperience, np.log(salary_data.Salary)) #correlation
model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = salary_data).fit()
model3.summary()
pred3 = model3.predict(pd.DataFrame(salary_data['YearsExperience']))
pred3_at = np.exp(pred3)
pred3_at
# Regression Line
plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(salary_data.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res3 = salary_data.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
# Polynomial transformation
# x = YearsExperience; x^2 = YearsExperience*YearsExperience; y = log(Salary)
model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary_data).fit()
model4.summary()
pred4 = model4.predict(pd.DataFrame(salary_data))
pred4_at = np.exp(pred4)
pred4_at
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = salary_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = salary_data.iloc[:, 1].values
plt.scatter(salary_data.YearsExperience, np.log(salary_data.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res4 = salary_data.Salary - pred4_at
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
train, test = train_test_split(salary_data, test_size = 0.3)
finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)',  data= train).fit()
finalmodel.summary()
# Predict on test Salary
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary
# Model Evaluation on Test data
test_res = test.Salary - pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT
# Model Evaluation on train data
train_res = train.Salary - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

