---
mathjax: true
title: Linear equations
date: 2023-03-28 09:47:39
categories: Basic math
comments: true
description: This article discusses linear equations and their solutions. It starts by introducing the concept of a linear system and provides an example of how to solve a system of linear equations using the elimination method and matrices. The article also explains what a singular system of linear equations is and gives an example of such a system. It is a basic introduction to linear algebra and is relevant for those interested in fields such as machine learning, data science, and engineering.
tags: 
  - Linear algebra
  - Linear regression
  - Linear equations 
  - Matrix
---

# Linear system

**The number of equations is the same as the number of unknowns.**

## Example

$$
\begin{aligned}
x-2 y & =1 \\
3 x+2 y & =11
\end{aligned}
$$
We can solve the system of linear equations by adding the two equations together to eliminate the $y$ variable:

$$
\begin{aligned}
(x - 2y) + (3x + 2y) &= 1 + 11 \\
4x &= 12 \\
x &= 3
\end{aligned}
$$
Then we can substitute $x = 3$ back into one of the original equations to solve for $y$:

$$
\begin{aligned}
3 - 2y &= 1 \\
-2y &= -2 \\
y &= 1
\end{aligned}
$$
Therefore, the solution to the system of linear equations is $x = 3$ and $y = 1$.

We can also solve the system of equations using matrices. The system of equations can be represented in matrix form as:
$$
\begin{aligned} \begin{bmatrix} 1 & -2 \\ 3 & 2 \ \end{bmatrix} \begin{bmatrix} x \\ y \ \end{bmatrix} &= \begin{bmatrix} 1 \\ 11 \ \end{bmatrix} \end{aligned}
$$
We can solve this matrix equation by finding the inverse of the coefficient matrix and multiplying it with the constant matrix:
$$
\begin{aligned}
\begin{bmatrix}
x \\
y \
\end{bmatrix}
&=
\begin{bmatrix}
1 & -2 \\
3 & 2 \
\end{bmatrix}^{-1}
\begin{bmatrix}
1 \\
11 \
\end{bmatrix} \\
&=
\begin{bmatrix}
3 \\
1\
\end{bmatrix}
\end{aligned}
$$

```python
# Define the system of linear equations
A = np.array([[3, 2], [1, -2]])
b = np.array([11, 1])

# Solve the system of linear equations
x = np.linalg.inv(A).dot(b)
# x = np.linalg.solve(A, b)

# Print the solution
print('x =', x[0])
print('y =', x[1])

# x = 3.0
# y = 1.0
```


# Singular system

A singular system of linear equations is a system of equations that has no unique solution or has infinitely many solutions.
$$
\begin{aligned}
x + y &= 3 \\
2x + 2y &= 6 \
\end{aligned}
$$
