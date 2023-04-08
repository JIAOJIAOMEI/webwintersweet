---
mathjax: true
title: Regression
date: 2023-03-28 15:08:31
tags: 
  - Linear regression
  - Polynomial regression 
  - Linear algebra
  - Span
  - Rank
---

### Rank

In linear algebra, the rank of a matrix is the number of linearly independent rows or columns in the matrix.

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 3/28/23 17:23
import numpy as np

# Define a matrix
A = np.array([[1, 0, 3],
              [-1, 9, 6],
              [3, 2, 9]])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A)

# Print the rank of the matrix
print("Rank of the matrix: ", rank)
```

```
Rank of the matrix:  3
```

To find the rank, we need to :
$$
\begin{equation}
\text { Find the row echelon form of }\left[\begin{array}{ccc}
1 & 0 & 3 \\
-1 & 9 & 6 \\
3 & 2 & 9
\end{array}\right]
\end{equation}
$$
The solution is:

Add row 1 to row $2: R_2=R_2+R_1$.
$$
\left[\begin{array}{lll}
1 & 0 & 3 \\
0 & 9 & 9 \\
3 & 2 & 9
\end{array}\right]
$$
Subtract row 1 multiplied by 3 from row 3 : $R_3=R_3-3 R_1$.
$$
\left[\begin{array}{lll}
1 & 0 & 3 \\
0 & 9 & 9 \\
0 & 2 & 0
\end{array}\right]
$$
Subtract row 2 multiplied by $\frac{2}{9}$ from row 3 : $R_3=R_3-\frac{2 R_2}{9}$.
$$
\left[\begin{array}{ccc}
1 & 0 & 3 \\
0 & 9 & 9 \\
0 & 0 & -2
\end{array}\right]
$$

$$
\begin{equation}
\text { The row echelon form is }\left[\begin{array}{ccc}
1 & 0 & 3 \\
0 & 9 & 9 \\
0 & 0 & -2
\end{array}\right]
\end{equation}
$$

**It means that all three rows or columns are linearly independent**. In other words, none of the rows or columns can be expressed as a linear combination of the other rows or columns.

**If a 3x3 matrix has rank 3, it means that its rows (or columns) span the entire 3D space**, and any vector in 3D can be expressed as a linear combination of the rows (or columns) of the matrix.

Okay, another example
$$
\begin{equation}
\text { Find the row echelon form of }\left[\begin{array}{ccc}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{array}\right]
\end{equation}
$$
Subtract row 1 multiplied by 4 from row $2: R_2=R_2-4 R_1$.
$$
\left[\begin{array}{ccc}
1 & 2 & 3 \\
0 & -3 & -6 \\
7 & 8 & 9
\end{array}\right]
$$
Subtract row 1 multiplied by 7 from row $3: R_3=R_3-7 R_1$.
$$
\left[\begin{array}{ccc}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & -6 & -12
\end{array}\right]
$$
Subtract row 2 multiplied by 2 from row $3: R_3=R_3-2 R_2$.
$$
\left[\begin{array}{ccc}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & 0 & 0
\end{array}\right]
$$
Since the element at row 3 and column 3 (pivot element) equals 0 , we need to swap the rows.
Find the first nonzero element in column 3 under the pivot entry.
As can be seen, there are no such entries.
$$
\begin{equation}
\text { The row echelon form is }\left[\begin{array}{ccc}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & 0 & 0
\end{array}\right]
\end{equation}
$$
so the rank is 2.

**If a 3 x 3 matrix has a rank of 2, it means that only two of its rows or columns are linearly independent, and the third row or column can be expressed as a linear combination of the other two.**

**Geometrically, this means that the rows or columns of the matrix lie in a plane in three-dimensional space.**

### Easy to remember Rank

In simpler terms, imagine that you're going out and you bring three items with you: an umbrella, a raincoat, and a loaf of bread. However, the umbrella and raincoat serve the same purpose of keeping you dry in the rain, so one of them is unnecessary. Therefore, you are effectively only bringing two items with you. This is similar to a 3 x 3 matrix with a rank of 2, where one row or column is redundant and can be expressed as a linear combination of the other two.

### Span

In linear algebra, the span of a set of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ is the set of all possible linear combinations of those vectors. Formally, the span is defined as follows:
$$
\begin{equation}
\operatorname{span}\left(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\right)=\left\{\sum_{i=1}^n c_i \mathbf{v}_i \mid c_i \in \mathbb{R}\right\}
\end{equation}
$$
In words, the span is the set of all possible vectors that can be formed by scaling and adding the given vectors. It can be thought of as a "subspace" of the vector space that contains the given vectors.

Geometrically, the span of a set of vectors is the smallest subspace that contains all those vectors. For example, the span of two non-collinear vectors in $\mathbb{R}^2$ is the entire plane, while the span of two parallel vectors is the line they lie on.

### Features

The Boston dataset contains information on various housing features in 506 neighborhoods around Boston, which can be used to evaluate the price of a house (target) based on different attributes. The dataset includes 13 attributes, and median value of owner-occupied homes in thousands of dollars, is provided in attribute 14.

```python
# load a dataset for prediction from library
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
print(boston.data.shape)
print(boston.feature_names)
print(boston.target)
```

```
Number of Instances: 506 
Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
```

For example, for the first row, it means given the first row of data, the MEDV is 24. we have 506 rows.

```
      CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33  36.2
```

Given a matrix of size 506 x 13, the rank of the matrix must be less than or equal to 13, since the maximum rank of a 506 x 13 matrix is 13. 

**If the rank is equal to 13, then there are indeed 13 independent features in the dataset, and each feature provides unique information that is not redundant with the other features. If the rank is less than 13, for example, if the rank is 10, then it means that some of the features are redundant or can be expressed as linear combinations of other features.** For example, it is intuitive that there might be a relationship between two features, such as the number of convenience stores and the accessibility of transportation, where areas with better transportation tend to have more convenience stores. Then, they are not completely independent of each other and can be considered redundant or providing similar information.

In real-world engineering applications, the relationship between two features may not be completely independent or dependent, but may exist somewhere in between. These relationships can be measured using correlation coefficients or other statistical measures, such as mutual information or covariance.

### Correlation coefficients

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 3/28/23 17:23

# load a dataset for prediction from library
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# plot Correlation coefficients
import seaborn as sns
import matplotlib.pyplot as plt
# give a mask for the upper triangle
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# set font as times new roman
plt.rcParams['font.family'] = 'Times New Roman'
# plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f', cmap='viridis')
plt.savefig('Correlation coefficients.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

![Correlation coefficients](Regression/Correlation%20coefficients.png)

The heatmap provides a visual representation of the strength and direction of the correlation between variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.

### Linear system

```
      CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33  36.2
```

let's recall this dataset. It just shows the first 5 rows, but actually we have 506 rows.

**The Boston housing dataset can be represented as a linear system with 13 unknown variables and 506 equations.**

Let $\mathbf{X}$ be the $506 \times 13$ matrix of input features, where each row corresponds to an observation, and each column corresponds to a feature. The feature columns are denoted as $\mathbf{X} = [\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_{13}}]$.

Let $\mathbf{y}$ be the $506 \times 1$ vector of target values, which corresponds to the median value of owner-occupied homes in $1000's$.

The linear regression model can be written as:
$$
\mathbf{y} = \theta_0 + \theta_1 \mathbf{x_1} + \theta_2 \mathbf{x_2} + \cdots + \theta_{13} \mathbf{x_{13}}
$$
where $\theta_0$ is the intercept or bias term, $\theta_1, \theta_2, \ldots, \theta_{13}$ are the model coefficients or weights, $\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_{13}}$ are the input feature columns.

In matrix notation, the linear regression model can be written as:
$$
\mathbf{y} = \mathbf{X}\mathbf{\theta}
$$
where $\mathbf{X}$ is the $506 \times 13$ matrix of input features, $\mathbf{\theta}$ is the $13 \times 1$ vector of model coefficients, and $\mathbf{\epsilon}$ is the $506 \times 1$ vector of errors.

The goal is to find $\mathbf{\theta}$.

### Normal equation

The solution for the model coefficients $\mathbf{\theta}$ using the normal equation is given by:
$$
\mathbf{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$
In order to find the values of $\mathbf{\theta}$, we need to first compute the matrix product $\mathbf{X}^T\mathbf{X}$ and the vector product $\mathbf{X}^T\mathbf{y}$, and then apply matrix inversion to $(\mathbf{X}^T\mathbf{X})$.

```python
import numpy as np
import pandas as pd

# load a dataset for prediction from library
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# solve this problem by normal equation
# define X and y, X 13 features, y 1 target
X = df.drop('MEDV', axis=1).values
y = df['MEDV'].values
# add a column of 1 to X, why?
# because the first element of theta is the intercept
X = np.hstack((np.ones((X.shape[0], 1)), X))
# compute the normal equation
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# print the result
print('theta: ', theta)
```

```
theta:  [ 3.64594884e+01 -1.08011358e-01  4.64204584e-02  2.05586264e-02
  2.68673382e+00 -1.77666112e+01  3.80986521e+00  6.92224640e-04
 -1.47556685e+00  3.06049479e-01 -1.23345939e-02 -9.52747232e-01
  9.31168327e-03 -5.24758378e-01]
```

### Notation

In a linear regression model, there is typically an error term $\mathbf{\epsilon}$ that represents the difference between the predicted target values and the true target values.

The linear regression model should be written as:
$$
\mathbf{y} = \mathbf{X}\mathbf{\theta} + \mathbf{\epsilon}
$$
where $\mathbf{y}$ is a vector of observed target values, $\mathbf{X}$ is a matrix of input feature values, $\mathbf{\theta}$ is a vector of model coefficients, and $\mathbf{\epsilon}$ is a vector of errors.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data points
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 3, 5, 6, 8, 9])

# Calculate the slope and intercept of the regression line using the least squares method
A = np.vstack([x, np.ones(len(x))]).T
m, b = np.linalg.lstsq(A, y, rcond=None)[0]

# Plot the data points and the regression line
plt.scatter(x, y)
plt.plot(x, m*x + b, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.savefig('Linear Regression.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

<p align="center">
  <img src="Regression/Linear%20Regression.png" alt="Linear Regression" style="zoom:30%;" />
</p>

The error term in a linear regression model accounts for the fact that the predicted values may not perfectly align with the actual values, due to the inherent variability in the data. In other words, the error term allows for some deviation between the predicted values and the true values.

### Polynomial regression

Polynomial regression and linear regression are similar in that they both aim to model the relationship between an input variable and a target variable. However, they differ in the functional form of the model that they use to capture this relationship.

Linear regression models the relationship between the input variable and target variable as a linear function, represented by a straight line. Polynomial regression, on the other hand, models the relationship between the input variable and target variable as a polynomial function of degree n, where n is the highest power of the input variable in the model.

In other words, polynomial regression allows for a more flexible and non-linear relationship between the input and target variables than linear regression. By fitting a polynomial curve to the data, polynomial regression can capture more complex patterns and relationships that may not be apparent in a linear model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Reshape the data into a column vector
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Fit a polynomial regression model to the data
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Plot the original data and the fitted curve
plt.scatter(x, y, color='blue')
plt.plot(x, poly_reg.predict(poly.fit_transform(x)), color='red')
plt.title('Polynomial Regression of Sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Polynomial Regression of Sin(x).png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

<p align="center">
  <img src="Regression/Polynomial Regression of Sin(x).png" alt="Polynomial Regression of Sin(x)" style="zoom:30%;" />
</p>

### Reference

1. UCI Machine Learning Repository (2018). Housing Data Set [Data File]. Retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
2. Strang, G. (2016). Introduction to linear algebra (5th ed.). Wellesley, MA: Wellesley-Cambridge Press.







