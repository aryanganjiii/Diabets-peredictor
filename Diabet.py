import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

x=pd.read_csv("diabates.csv")
Y=x["Y"]
x.drop(columns=["Y"],inplace=True)

model = LinearRegression()

model.fit(x,Y)

plt.plot(Y,label="Real Data")
plt.plot(model.predict(x), label="Regression")
print(model.score(x,Y)*100)
print(model.predict([[float(input("enter age :")),float(input("enter sex :")),float(input("enter BMI :")),float(input("enter bp :")),float(input("enter s1 :")),float(input("enter s2 :")),
float(input("enter s3 :")),float(input("enter s4 :")),float(input("enter s5 :")),float(input("enter s6 :"))]]))

plt.legend()
plt.show()

