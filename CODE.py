import pandas as pd
from sklearn.linear_model import LinearRegression

data ={
  "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "Price": [114348.20, 113574.90, 114485.20, 116770.30, 116364.70, 118231.90, 121660.90, 123227.80, 118389.30]
}
df = pd.DataFrame(data)
x = df[["Day"]] # Features
y= df["Price"] # Target

model = LinearRegression()
model.fit(x, y)
prediction = model.predict([[10]])
print(df)
print("Predicted price for Day 10:",
prediction[0])
