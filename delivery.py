# 2
import pandas as pd 
import numpy as np  
# Importing the dataset
delivery_data = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Linear Regression\\delivery_time.csv")
delivery_data=delivery_data.rename(columns={'Delivery Time':'Delivery_Time','Sorting Time':'Sorting_Time'})
# Performing EDA
delivery_data.describe()
# Graphical Representation
import matplotlib.pyplot as plt 
# Barplot
plt.bar(height = delivery_data.Delivery_Time, x = np.arange(1, 22, 1))
# Histogram
plt.hist(delivery_data.Delivery_Time) 
# Boxplot
plt.boxplot(delivery_data.Delivery_Time) 
plt.bar(height = delivery_data.Sorting_Time, x = np.arange(1, 22, 1))
plt.hist(delivery_data.Sorting_Time) #histogram
plt.boxplot(delivery_data.Sorting_Time) #boxplot
# Scatter plot
plt.scatter(x = delivery_data['Sorting_Time'], y = delivery_data['Delivery_Time'], color = 'green') 
# correlation
np.corrcoef(delivery_data.Sorting_Time, delivery_data.Delivery_Time) 
# Covariance
cov_output = np.cov(delivery_data.Sorting_Time, delivery_data.Delivery_Time)[0, 1]
cov_output
delivery_data.cov()
import statsmodels.formula.api as smf
# Simple Linear Regression
model = smf.ols('Delivery_Time ~ Sorting_Time', data = delivery_data).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(delivery_data['Sorting_Time']))
# Regression Line
plt.scatter(delivery_data.Sorting_Time, delivery_data.Delivery_Time)
plt.plot(delivery_data.Sorting_Time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res1 = delivery_data.Delivery_Time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1
# Model building on Transformed Data
# Log Transformation
# x = log(Sorting_Time); y = at
plt.scatter(x = np.log(delivery_data['Sorting_Time']), y = delivery_data['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(delivery_data.Sorting_Time), delivery_data.Delivery_Time) #correlation
model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = delivery_data).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(delivery_data['Sorting_Time']))
# Regression Line
plt.scatter(np.log(delivery_data.Sorting_Time), delivery_data.Delivery_Time)
plt.plot(np.log(delivery_data.Sorting_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res2 = delivery_data.Delivery_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
# x = Sorting_Time; y = log(at)
plt.scatter(x = delivery_data['Sorting_Time'], y = np.log(delivery_data['Delivery_Time']), color = 'orange')
np.corrcoef(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time)) #correlation
model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = delivery_data).fit()
model3.summary()
pred3 = model3.predict(pd.DataFrame(delivery_data['Sorting_Time']))
pred3_at = np.exp(pred3)
pred3_at
# Regression Line
plt.scatter(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time))
plt.plot(delivery_data.Sorting_Time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res3 = delivery_data.Delivery_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
# Polynomial transformation
# x = Sorting_Time; x^2 = Sorting_Time*Sorting_Time; y = log(at)
model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time+ I(Sorting_Time*Sorting_Time)', data = delivery_data).fit()
model4.summary()
pred4 = model4.predict(pd.DataFrame(delivery_data))
pred4_at = np.exp(pred4)
pred4_at
# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = delivery_data.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)
# y = delivery_data.iloc[:, 1].values
plt.scatter(delivery_data.Sorting_Time, np.log(delivery_data.Delivery_Time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error calculation
res4 = delivery_data.Delivery_Time - pred4_at
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
train, test = train_test_split(delivery_data, test_size = 0.3)
finalmodel = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = train).fit()
finalmodel.summary()
# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Delivery_Time = np.exp(test_pred)
pred_test_Delivery_Time
# Model Evaluation on Test data
test_res = test.Delivery_Time - pred_test_Delivery_Time
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse
# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Delivery_Time = np.exp(train_pred)
pred_train_Delivery_Time
# Model Evaluation on train data
train_res = train.Delivery_Time - pred_train_Delivery_Time
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

