import pandas as pd
from sklearn.linear_model import LinearRegression

data ={
  "Day": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "Price": [104.69, 103.48, 103.36, 102.27, 101.77, 101.12, 102.21, 102.45, 102.69,103.03]
}
df = pd.DataFrame(data)
x = df[["Day"]] # Features
y= df["Price"] # Target

model = LinearRegression()
model.fit(x, y)
prediction = model.predict([[11]])
print(df)
print("Predicted price for Day 11:",
prediction[0])
