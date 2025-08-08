import pandas as pd
from sklearn.linear_model import
LinearRegression

data ={
  "Day": [1, 2, 3, 4],
  "Price": [101.23, 102.22, 102.62, 103.3]
}
df = pd.DataFrame(data)
x = df[["Day"]] # Features
y= df["Price"] # Target

model = LinearRegression()
model.fit(x, y)
prediction = model.predict([[5]])
print(df)
print("Predicted price for Day 5:",
prediction[0])
