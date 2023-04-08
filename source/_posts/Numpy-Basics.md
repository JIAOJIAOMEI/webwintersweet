---
mathjax: true
title: Numpy Basics
date: 2023-03-28 06:14:24
categories: Python skills
comments: true
description: The article is a beginner's guide to working with NumPy, a powerful Python library for numerical computing. It covers creating 1D and 2D arrays using various methods, checking the attributes of arrays, basic operations on arrays, indexing and slicing arrays, and stacking and splitting arrays. The article provides code examples and outputs to demonstrate the different features and functions of NumPy.
tags: 
  - Python
  - Numpy 
  - Matrix
---

### Create 1D array using Numpy

```python
import numpy as np

# Create 1D arrays using different methods
a = np.array([1, 2, 3, 4, 5])        # Create 1D array from a list
# [1 2 3 4 5]
b = np.arange(10)                   # Create 1D array using range
# [0 1 2 3 4 5 6 7 8 9]
c = np.linspace(0, 1, 5)             # Create 1D array with equally spaced values
# [0.   0.25 0.5  0.75 1.  ]
d = np.random.rand(5)               # Create 1D array with random values
# [0.785179   0.05506288 0.83677954 0.98587586 0.52964397]
e = np.zeros(5)                     # Create 1D array of zeros
# [0. 0. 0. 0. 0.]
f = np.ones(5)                      # Create 1D array of ones
# [1. 1. 1. 1. 1.]
g = np.full(5, 2)                   # Create 1D array of a specific value
# [2 2 2 2 2]
h = np.empty(5)                     # Create 1D array without initializing its values
# [0. 0. 0. 0. 0.]
```

### Create 2D array using Numpy

```python
import numpy as np

# Create 2D arrays using different methods
a = np.array([[1, 2, 3], [4, 5, 6]])         # Create 2D array from nested lists
b = np.zeros((3, 3))                        # Create 2D array of zeros
c = np.ones((2, 2))                         # Create 2D array of ones
d = np.eye(3)                               # Create 2D identity matrix
e = np.random.rand(2, 3)                    # Create 2D array with random values
f = np.array([[1, 2], [3, 4]])              # Create 2D array from a list of lists
g = np.empty((2, 2))                        # Create uninitialized 2D array
```

```
# a
[[1 2 3]
 [4 5 6]]
 # b
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
 # c
[[1. 1.]
 [1. 1.]]
 # d
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
 # e
[[0.23691216 0.7258241  0.25003681]
 [0.52863985 0.92896228 0.81953756]]
 # f
[[1 2]
 [3 4]]
 # g
[[0. 0.]
 [0. 0.]]
```

### Checking attributes of a 2D array

```python
import numpy as np

# Create a 2D array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Check the size, shape, and number of dimensions of the array
print("Size of the array: ", a.size)
print("Shape of the array: ", a.shape)
print("Number of dimensions: ", a.ndim)

# Check the data type of the array
print("Data type of the array: ", a.dtype)

# Check the number of bytes used by each element in the array
print("Bytes per element: ", a.itemsize)

# Check the total number of bytes used by the array
print("Total number of bytes used by the array: ", a.nbytes)
```

```
Size of the array:  6
Shape of the array:  (2, 3)
Number of dimensions:  2
Data type of the array:  int64
Bytes per element:  8
Total number of bytes used by the array:  48
```

### Comprehension

```python
# List comprehension example
# Create a list of squares of numbers from 1 to 10
squares = [x**2 for x in range(1, 11)]
print("List of squares: ", squares)

# Dictionary comprehension example
# Create a dictionary of squares of numbers from 1 to 10
squares_dict = {x: x**2 for x in range(1, 11)}
print("Dictionary of squares: ", squares_dict)

# Set comprehension example
# Create a set of squares of numbers from 1 to 10
squares_set = {x**2 for x in range(1, 11)}
print("Set of squares: ", squares_set)
```

```python
List of squares:  [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
Dictionary of squares:  {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}
Set of squares:  {64, 1, 4, 36, 100, 9, 16, 49, 81, 25}
```

### Basic operations of a Matrix

```python
import numpy as np

# Create a 2D array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Perform some basic operations on the array
print("Array to the power of 2: ", a ** 2)        # Exponentiation
print("Array divided by 2: ", a / 2)              # Division
print("Array modulo 2: ", a % 2)                  # Modulo
print("Array integer divided by 2: ", a // 2)     # Integer division
print("Array greater than 5: ", a > 5)            # Comparison
print("Array equal to 6: ", a == 6)               # Comparison
print("Array plus 100: ", a + 100)               # Addition
```

```
Array to the power of 2:  [[ 1  4  9]
 [16 25 36]]
Array divided by 2:  [[0.5 1.  1.5]
 [2.  2.5 3. ]]
Array modulo 2:  [[1 0 1]
 [0 1 0]]
Array integer divided by 2:  [[0 1 1]
 [2 2 3]]
Array greater than 5:  [[False False False]
 [False False  True]]
Array equal to 6:  [[False False False]
 [False False  True]]
Array plus 100:  [[101 102 103]
 [104 105 106]]
```

### Index and slice for 1D array

```python
import numpy as np

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])

# Indexing the array
print("Element at index 0: ", a[0])         # Indexing by position
print("Element at index -1: ", a[-1])       # Indexing by negative position
print("Element at index 2: ", a[2])         # Indexing by position
print("Element at index -3: ", a[-3])       # Indexing by negative position

# Slicing the array
print("Elements from index 1 to 3: ", a[1:4])    # Slicing using start and end positions
print("Elements from index 0 to end: ", a[:])     # Slicing from start to end
print("Every second element: ", a[::2])           # Slicing with a step of 2
print("Reversed array: ", a[::-1])                # Slicing with a negative step
```

```
Element at index 0:  1
Element at index -1:  5
Element at index 2:  3
Element at index -3:  3
Elements from index 1 to 3:  [2 3 4]
Elements from index 0 to end:  [1 2 3 4 5]
Every second element:  [1 3 5]
Reversed array:  [5 4 3 2 1]
```

### Index and slice for 2D array

```python
import numpy as np

# Create a 2D array
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing the array
print("Element at row 0, column 1: ", a[0, 1])        # Indexing a single element
print("Element at row -1, column -2: ", a[-1, -2])    # Indexing a single element using negative indices
print("First row: ", a[0, :])                          # Indexing a row
print("Second column: ", a[:, 1])                      # Indexing a column

# Slicing the array
print("Subarray from rows 0 to 1, columns 1 to 2: ")
print(a[0:2, 1:3])                                    # Slicing a subarray
print("First two rows: ")
print(a[:2, :])                                        # Slicing the first two rows
print("Last two columns: ")
print(a[:, -2:])                                       # Slicing the last two columns
```

```
Element at row 0, column 1:  2
Element at row -1, column -2:  8
First row:  [1 2 3]
Second column:  [2 5 8]
Subarray from rows 0 to 1, columns 1 to 2: 
[[2 3]
 [5 6]]
First two rows: 
[[1 2 3]
 [4 5 6]]
Last two columns: 
[[2 3]
 [5 6]
 [8 9]]
```

### Index and slice for 3D array

```python
import numpy as np

# Create a 3D array
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Indexing the array
print("Element at position 1, 0, 1: ", a[1, 0, 1])     # Indexing a single element
print("Last element in the last row of the last matrix: ", a[-1, -1, -1])  # Indexing a single element using negative indices
print("First matrix: ")
print(a[0, :, :])                                      # Indexing the first matrix
print("Second row of the second matrix: ")
print(a[1, 1, :])                                      # Indexing the second row of the second matrix

# Slicing the array
print("Subarray from the first matrix: ")
print(a[0, :, :])                                      # Slicing a subarray
print("Subarray from the second matrix, rows 0 to 1, columns 0 to 1: ")
print(a[1, 0:2, 0:2])                                  # Slicing a subarray
print("Last row of the last matrix: ")
print(a[-1, -1, :])                                     # Slicing the last row of the last matrix
```

```
Element at position 1, 0, 1:  6
Last element in the last row of the last matrix:  8
First matrix: 
[[1 2]
 [3 4]]
Second row of the second matrix: 
[7 8]
Subarray from the first matrix: 
[[1 2]
 [3 4]]
Subarray from the second matrix, rows 0 to 1, columns 0 to 1: 
[[5 6]
 [7 8]]
Last row of the last matrix: 
[7 8]
```

### Stack and split

```py
import numpy as np

# Create two 2D arrays
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

# vstack: Stack arrays vertically (row-wise)
c = np.vstack((a, b))
print("Vertically stacked array: ")
print(c)

# hstack: Stack arrays horizontally (column-wise)
d = np.hstack((a, b))
print("Horizontally stacked array: ")
print(d)

# dstack: Stack arrays depth-wise (along third axis)
e = np.dstack((a, b))
print("Depth-wise stacked array: ")
print(e)

# concatenate: Join arrays along an existing axis
f = np.concatenate((a, b), axis=0)
print("Concatenated array along axis 0: ")
print(f)

# row_stack: Stack arrays vertically (row-wise)
g = np.row_stack((a, b))
print("Vertically stacked array using row_stack: ")
print(g)

# column_stack: Stack arrays horizontally (column-wise)
h = np.column_stack((a, b))
print("Horizontally stacked array using column_stack: ")
print(h)

# hsplit: Split arrays horizontally (column-wise)
i = np.hsplit(d, 2)
print("Horizontally split arrays: ")
print(i)

# vsplit: Split arrays vertically (row-wise)
j = np.vsplit(c, 2)
print("Vertically split arrays: ")
print(j)

# dsplit: Split arrays depth-wise (along third axis)
k = np.dsplit(e, 2)
print("Depth-wise split arrays: ")
print(k)
```

```
Vertically stacked array: 
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
Horizontally stacked array: 
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
Depth-wise stacked array: 
[[[ 1  7]
  [ 2  8]
  [ 3  9]]

 [[ 4 10]
  [ 5 11]
  [ 6 12]]]
Concatenated array along axis 0: 
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
Vertically stacked array using row_stack: 
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
Horizontally stacked array using column_stack: 
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
Horizontally split arrays: 
[array([[1, 2, 3],
       [4, 5, 6]]), array([[ 7,  8,  9],
       [10, 11, 12]])]
Vertically split arrays: 
[array([[1, 2, 3],
       [4, 5, 6]]), array([[ 7,  8,  9],
       [10, 11, 12]])]
Depth-wise split arrays: 
[array([[[1],
        [2],
        [3]],

       [[4],
        [5],
        [6]]]), array([[[ 7],
        [ 8],
        [ 9]],

       [[10],
        [11],
        [12]]])]
```

### Broadcasting

```python
import numpy as np

# Broadcasting with scalars
a = np.array([1, 2, 3, 4])
b = 2
print(a + b)  # Output: [3 4 5 6]

# Broadcasting with arrays of different shapes
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([10, 20, 30])
print(c + d)  # Output: [[11 22 33] [14 25 36]]

# Broadcasting with arrays of different ranks
e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
f = np.array([10, 20, 30])
print(e + f)  # Output: [[11 22 33] [14 25 36] [17 28 39]]

# Broadcasting with arrays of different sizes
g = np.array([[1, 2, 3], [4, 5, 6]])
h = np.array([10])
print(g + h)  # Output: [[11 12 13] [14 15 16]]

# Broadcasting with arrays of different sizes and shapes
i = np.array([[1, 2, 3], [4, 5, 6]])
j = np.array([[10], [20]])
print(i + j)  # Output: [[11 12 13] [24 25 26]]
```

