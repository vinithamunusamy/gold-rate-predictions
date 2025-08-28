import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix

df = pd.read_csv("/content/Gold vs USDINR.csv")
print(df.tail(5))
print(df.columns)
print(df.dtypes)

df['Goldrate'] = df['Goldrate'].str.replace('₹', '').str.replace(',', '').astype(float)


x = df['Goldrate']
y = df['USD_INR']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("Train Data")
print(x_train)
print(y_train)

print("Test Data")
print(x_test)
print(y_test)

lin_reg = LinearRegression()
lin_reg.fit(x_train.values.reshape(-1,1),y_train)

print("Prediction Input : ")
print(x_test)
y_pred = lin_reg.predict(x_test.values.reshape(-1,1))
print("Prediction Result : ")
print(y_pred)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error :",rmse)

plt.scatter(x,y, color='blue', label= 'Data points')
plt.plot(x, lin_reg.predict(x.values.reshape(-1,1)), color='red', label= 'Regression line')
plt.xlabel('Goldrate')
