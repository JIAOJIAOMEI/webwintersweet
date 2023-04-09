---
mathjax: true
title: Basic operations of Matrix
date: 2023-03-27 18:42:35
categories: Basic math
comments: true
description: The article covers basic operations of matrices including transpose, vector and matrix addition, scalar multiplication, linear combination, dot product, matrix multiplication, identity matrix, and matrix inverse. It provides definitions, examples, and formulas for each operation.
tags: 
  - Matrix
  - Linear combination
  - Linear algebra
---

# Matrix transpose

The transpose of a matrix is an operation that flips the matrix over its diagonal, i.e., it switches the rows and columns of the matrix. 
$$
\begin{equation}
A_{i, j}^{\top}=A_{i, j}
\end{equation}
$$
Consider the following $2 \times 3$ matrix :
$$
A=\left[\begin{array}{lll}
1 & 2 & 3 \\
4 & 5 & 6
\end{array}\right]
$$
The transpose of $A$, denoted as $A^{\wedge} T$, can be obtained by flipping the rows and columns of A:
$$
A^T=\left[\begin{array}{ll}
1 & 4 \\
2 & 5 \\
3 & 6
\end{array}\right]
$$
Note that the original matrix $A$ has dimensions $2 \times 3$, while the transpose $\mathrm{A}^{\wedge} \mathrm{T}$ has dimensions $3 \times 2$. This is because the number of rows in A becomes the number of columns in $\mathrm{A}^{\wedge} \mathrm{T}$, and vice versa.

# Vector addition

Vector addition is the process of combining two or more vectors into a single vector. The resulting vector is the sum of the individual vectors.
$$
\begin{equation}
\boldsymbol{v}=\left[\begin{array}{r}
1 \\
1 \\
-1
\end{array}\right] \quad \text { and } \quad \boldsymbol{w}=\left[\begin{array}{l}
2 \\
3 \\
4
\end{array}\right] \quad \text { and } \quad \boldsymbol{v}+\boldsymbol{w}=\left[\begin{array}{l}
3 \\
4 \\
3
\end{array}\right]
\end{equation}
$$
# Matrix addition

Matrix addition is the process of combining two or more matrices into a single matrix. The resulting matrix is the sum of the individual matrices.

If we have two matrices A and B with the same dimensions m x n, then their sum, C, is defined as:
$$
\begin{equation}
\boldsymbol{C}=\boldsymbol{A}+\boldsymbol{B}, C_{i, j}=A_{i, j}+B_{i, j}
\end{equation}
$$
Consider the following matrices $\mathrm{A}$ and $\mathrm{B}$ :
$$
\begin{gathered}
A=\left[\begin{array}{ll}
1 & 2 \\
3 & 4 \\
5 & 6
\end{array}\right] \\
B=\left[\begin{array}{cc}
-1 & 2 \\
4 & -3 \\
7 & 1
\end{array}\right]
\end{gathered}
$$
To add these matrices, we add the corresponding elements:
$$
A+B=\left[\begin{array}{cc}
1+(-1) & 2+2 \\
3+4 & 4+(-3) \\
5+7 & 6+1
\end{array}\right]=\left[\begin{array}{cc}
0 & 4 \\
7 & 1 \\
12 & 7
\end{array}\right]
$$
# Scalar multiplication

Scalar multiplication is the process of multiplying a vector or a matrix by a scalar value. The scalar value is a single number that scales or stretches the vector or matrix.

 Vectors can be multiplied by 2 or by -1 or by any number $c$. 
$$
\begin{equation}
2 \boldsymbol{v}=\left[\begin{array}{l}
2 v_1 \\
2 v_2
\end{array}\right]=\boldsymbol{v}+\boldsymbol{v}
\end{equation}
$$

$$
\begin{equation}
-\boldsymbol{v}=\left[\begin{array}{l}
-v_1 \\
-v_2
\end{array}\right]
\end{equation}
$$

In case of a Matrix, let $A$ be a $2 \times 2$ matrix:
$$
A=\left[\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right]
$$
To find the scalar multiplication of A by 2 , we simply multiply each element of the matrix by 2 :
$$
2 A=2\left[\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right]=\left[\begin{array}{ll}
2 & 4 \\
6 & 8
\end{array}\right]
$$
Similarly, we can find the scalar multiplication of $A$ by -1 by multiplying each element of the matrix by -1 :
$$
-1 A=-\left[\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right]=\left[\begin{array}{ll}
-1 & -2 \\
-3 & -4
\end{array}\right]
$$
In general, given a matrix $A$ and a scalar $\mathrm{c}$, the scalar multiplication of $A$ by $\mathrm{c}$ is defined as:
$$
c A=c\left[\begin{array}{cccc}
a_{11} & a_{12} & \ldots & a_{1 n} \\
a_{21} & a_{22} & \ldots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m 1} & a_{m 2} & \ldots & a_{m n}
\end{array}\right]=\left[\begin{array}{cccc}
c a_{11} & c a_{12} & \ldots & c a_{1 n} \\
c a_{21} & c a_{22} & \ldots & c a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
c a_{m 1} & c a_{m 2} & \ldots & c a_{m n}
\end{array}\right]
$$

# Linear combination

Linear combination is a combination of addition and multiplication.

Given a set of vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n$ and a set of scalars $c_1, c_2, \dots, c_n$, a linear combination of the vectors is defined as the sum of the vectors multiplied by their corresponding scalar coefficients:
$$
c_1 \boldsymbol{v}_1+c_2 \boldsymbol{v}_2+\cdots+c_n \boldsymbol{v}_n=\sum_{i=1}^n c_i \boldsymbol{v}_i
$$
This expression represents the linear combination of the vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n$ using the scalar coefficients $c_1, c_2, \dots, c_n$, where the addition of the vectors and the multiplication of the scalars are both included.

For example, consider the following set of vectors:
$$
\boldsymbol{v}_1=\left[\begin{array}{l}
1 \\
2 \\
3
\end{array}\right], \quad \boldsymbol{v}_2=\left[\begin{array}{l}
4 \\
5 \\
6
\end{array}\right], \quad \boldsymbol{v}_3=\left[\begin{array}{l}
7 \\
8 \\
9
\end{array}\right]
$$
A linear combination of these vectors using the scalar coefficients $c_1 = 2$, $c_2 = -1$, and $c_3 = 3$ is:
$$
2 v_1-v_2+3 v_3=2\left[\begin{array}{l}
1 \\
2 \\
3
\end{array}\right]-\left[\begin{array}{l}
4 \\
5 \\
6
\end{array}\right]+3\left[\begin{array}{l}
7 \\
8 \\
9
\end{array}\right]=\left[\begin{array}{l}
29 \\
32 \\
35
\end{array}\right]
$$
## Dot product

Given two vectors $\boldsymbol{v}$ and $\boldsymbol{w}$ of the same dimension, the dot product between $\boldsymbol{v}$ and $\boldsymbol{w}$, denoted as $\boldsymbol{v} \cdot \boldsymbol{w}$, is the sum of the products of the corresponding components:
$$
\begin{equation}
\boldsymbol{v} \cdot \boldsymbol{w}=\sum_{i=1}^n v_i w_i=v_1 w_1+v_2 w_2+\cdots+v_n w_n
\end{equation}
$$
For example, consider the following two vectors:
$$
\boldsymbol{v}=\left[\begin{array}{l}
1 \\
2 \\
3
\end{array}\right], \quad \boldsymbol{w}=\left[\begin{array}{l}
4 \\
5 \\
6
\end{array}\right]
$$
To compute the dot product of $\boldsymbol{v}$ and  $ \boldsymbol {w}$, we multiply the corresponding elements of the two vectors and then sum the resulting products:
$$
v \cdot w=(1 \times 4)+(2 \times 5)+(3 \times 6)=32
$$

## Matrix multiplication

Given two matrices A and B, where A has dimensions $m \cdot n$ and B has dimensions $n \cdot p$, the product of A and B, denoted as C, is a matrix with dimensions $m \cdot p$, where the element in row $i$ and column $j$ is obtained by multiplying the $i$-th row of A with the $j$-th column of B, and then summing the resulting products:
$$
\begin{equation}
C_{i, j}=\sum_{k=1}^n A_{i, k} B_{k, j}, \quad \text { for } 1 \leq i \leq m, 1 \leq j \leq p
\end{equation}
$$
For example, consider the following matrices $A$ and $B$ :
$$
A=\left[\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right], B=\left[\begin{array}{ll}
5 & 6 \\
7 & 8
\end{array}\right]
$$
To multiply $A$ and $B$, we take the dot product of each row of $A$ with each column of $B$ :
$$
\left[\begin{array}{ll}
1 & 2 \\
3 & 4
\end{array}\right]\left[\begin{array}{ll}
5 & 6 \\
7 & 8
\end{array}\right]=\left[\begin{array}{ll}
1 \cdot 5+2 \cdot 7 & 1 \cdot 6+2 \cdot 8 \\
3 \cdot 5+4 \cdot 7 & 3 \cdot 6+4 \cdot 8
\end{array}\right]=\left[\begin{array}{ll}
19 & 22 \\
43 & 50
\end{array}\right]
$$

# Identity matrix

The identity matrix, denoted by $\boldsymbol{I}_n$, is a square matrix of dimension $n$ with ones on the main diagonal and zeros everywhere else. In other words, the entry in the $i$-th row and $j$-th column of $\boldsymbol{I}_n$ is:
$$
\begin{equation}
\left(\boldsymbol{I}_n\right)_{i, j}= \begin{cases}1 & \text { if } i=j \\ 0 & \text { if } i \neq j\end{cases}
\end{equation}
$$
For example, the $3 \times 3$ identity matrix is:
$$
\begin{equation}
\boldsymbol{I}_3=\left[\begin{array}{lll}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{array}\right]
\end{equation}
$$
The identity matrix is a special type of matrix in that it behaves like the number 1 in multiplication. Specifically, if $\boldsymbol{A}$ is a square matrix of dimension $n$, then:
$$
\begin{equation}
\boldsymbol{A} \boldsymbol{I}_n=\boldsymbol{I}_n \boldsymbol{A}=\boldsymbol{A}
\end{equation}
$$

# Matrix inverse

The inverse of a matrix $\boldsymbol{A}$, denoted as $\boldsymbol{A}^{-1}$, is defined as a matrix such that:
$$
\begin{equation}
\boldsymbol{A}^{-1} \boldsymbol{A}=\boldsymbol{I}_n
\end{equation}
$$
where $\boldsymbol{I}_n$ is the identity matrix of dimension $n$. If $\boldsymbol{A}^{-1}$ exists (**not all matrices are invertible**), then the solution to the linear system $\boldsymbol{A x}=\boldsymbol{b}$ is given by:
$$
\begin{equation}
\boldsymbol{x}=\boldsymbol{A}^{-1} \boldsymbol{b}
\end{equation}
$$
This is because we can multiply both sides of the equation $\boldsymbol{A x}=\boldsymbol{b}$ by $\boldsymbol{A}^{-1}$ on the left to obtain:
$$
\begin{equation}
\boldsymbol{A}^{-1} \boldsymbol{A} \boldsymbol{x}=\boldsymbol{A}^{-1} \boldsymbol{b}
\end{equation}
$$
which simplifies to:
$$
\begin{equation}
\boldsymbol{I}_n \boldsymbol{x}=\boldsymbol{x}=\boldsymbol{A}^{-1} \boldsymbol{b}
\end{equation}
$$
If the determinant of $\boldsymbol{A}$ is zero, then the matrix $\boldsymbol{A}$ is singular or non-invertible, and it does not have an inverse. 

# Python examples
```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 3/27/23 22:26

import numpy as np

# Matrix Transpose
A = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix A:")
print(A)
print("Transpose of A:")
print(A.T)

# Vector Addition
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
w = u + v
print("Vector u:", u)
print("Vector v:", v)
print("Vector u + v:", w)

# Matrix Addition
B = np.array([[1, 2, 3], [4, 5, 6]])
C = np.array([[7, 8, 9], [10, 11, 12]])
D = B + C
print("Matrix B:")
print(B)
print("Matrix C:")
print(C)
print("Matrix B + C:")
print(D)

# Scalar Multiplication
k = 3
E = np.array([[1, 2, 3], [4, 5, 6]])
F = k * E
print("Scalar k:", k)
print("Matrix E:")
print(E)
print("Scalar k times E:")
print(F)

# Linear Combination
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
a = 2
b = 3
w = a * u + b * v
print("Vector u:", u)
print("Vector v:", v)
print("Scalar a:", a)
print("Scalar b:", b)
print("Linear combination a*u + b*v:")
print(w)

# Dot Product
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
w = np.dot(u, v)
print("Vector u:", u)
print("Vector v:", v)
print("Dot product of u and v:", w)

# Matrix Multiplication
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8], [9, 10]])
C = np.dot(A, B)
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Matrix product of A and B:")
print(C)

# Identity Matrix
I = np.eye(3)
print("Identity matrix of size 3:")
print(I)

# Matrix Inverse
A = np.array([[1, 2], [3, 4]])
B = np.linalg.inv(A)
print("Matrix A:")
print(A)
print("Inverse of A:")
print(B)
```

# Reference

1. Strang, G. (2016). Introduction to linear algebra (5th ed.). Wellesley, MA: Wellesley-Cambridge Press.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.



















