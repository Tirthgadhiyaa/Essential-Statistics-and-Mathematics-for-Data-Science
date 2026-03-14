# assessment_module7.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------Numpy


# Q1
arr = np.arange(10)
result_q1 = arr.reshape(2, 5)
print("Q1:", result_q1)


# Q2
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

result_q2 = np.intersect1d(a, b)
print("Q2:", result_q2)


# Q3
a = np.array([2, 6, 1, 9, 10, 3, 27])

result_q3 = np.where((a >= 5) & (a <= 10))
print("Q3:", a[result_q3])


# Q4
np.set_printoptions(threshold=6)

a = np.arange(15)
print("Q4:", a)


#------------------- Pandas

# Q1
ser = pd.Series(np.random.random(10))

print("\nQ5:")
print("Min:", ser.min())
print("25th percentile:", ser.quantile(0.25))
print("Median:", ser.median())
print("75th percentile:", ser.quantile(0.75))
print("Max:", ser.max())


# Q2
data = {
    "Name": ["A","B","C","D"],
    "Age": [21,22,23,24],
    "City": ["Surat","Delhi","Mumbai","Pune"]
}

df = pd.DataFrame(data)
print("\nQ6 DataFrame:")
print(df)


# Q3
np_data = np.random.randint(1,100,(5,3))
df2 = pd.DataFrame(np_data)

print("\nQ7 DataFrame from NumPy:")
print(df2)


# Q4
# Default behavior:
# Columns -> labeled 0,1,2,...
# Rows -> labeled 0,1,2,...


# Q5
# Example loading large CSV
# df_large = pd.read_csv("large_dataset.csv")


# Q6
print("/nQ9 Rows and Columns:", df.shape)


# Q7
print("/nQ10 First rows:")
print(df.head())


# Q8
# If selecting single column from DataFrame

col = df["Age"]
print("/nQ11 Column type:", type(col))
# It returns pandas Series


# Q9
x = [3,4,5,6]
y = [1.5,2,2.5,3]

plt.figure()
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Q12 Line Plot")
plt.show()


# Q10
x = np.arange(0,6,0.3)

plt.figure()
plt.plot(x,x**2,"ro")
plt.plot(x,x**3,"bs")
plt.plot(x,x**4,"g")
plt.plot(x,x**5,":")
plt.xlim(0,6)
plt.ylim(0,125)
plt.title("Q13 Multiple Functions")
plt.show()


# Q11
height = [179,155,191,152,188,177]
names = ['QA','WB','EC','RD','TE','YF']

plt.figure()
plt.bar(names,height)
plt.title("Height Comparison")
plt.show()


# Q12
x = np.random.randn(100000)

plt.figure()
plt.hist(x,bins=10)
plt.title("Histogram 10 bins")
plt.show()

plt.figure()
plt.hist(x,bins=20)
plt.title("Histogram 20 bins")
plt.show()

plt.figure()
plt.hist(x,bins=50)
plt.title("Histogram 50 bins")
plt.show()



# -------------------Theory


# Q13
# Features of TensorFlow:
# 1. Open source machine learning framework
# 2. Supports deep learning and neural networks
# 3. Runs on CPU, GPU and TPU
# 4. Automatic differentiation
# 5. Scalable for large datasets
# 6. Provides high level APIs like Keras


# Q14
# Limitations of TensorFlow:
# 1. Hard to debug compared to simple libraries
# 2. Large memory usage
# 3. Slower for small models
# 4. Complex syntax for beginners


# Q15
# Supervised Learning:
# Machine learning method where model is trained using labeled data.
# Example: Classification, Regression.

# Unsupervised Learning:
# Machine learning method where model finds patterns without labeled output.
# Example: Clustering, Association.