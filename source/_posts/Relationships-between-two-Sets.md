---
mathjax: true
title: Relationships between two Sets
categories: Basic math
comments: true
description: The code shows how to create Venn diagrams in Python using the matplotlib_venn library. It also demonstrates how to calculate cardinality and visualize the Cartesian product of two sets.
date: 2023-04-03 15:21:07
tags:
  - Set
  - Math
  - Probability
  - Python
  - Basics
---
### Unions, intersections, and complements

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 4/3/23 16:59

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# create sets A and B
A = set([1, 2, 3])
B = set([4, 5, 6, 7, 8])

# create subplots
fig, axs = plt.subplots(nrows=2, ncols=2)

# plot 1: A and B are independent
venn2([A, B], set_labels=('A', 'B'),set_colors=('skyblue', 'orange'), ax=axs[0, 0])
axs[0, 0].set_title('Independent Sets')

# plot 2: A and B have some common parts
A = set([1, 2, 3, 4, 5])
B = set([4, 5, 6, 7, 8])
venn2([A, B], set_labels=('A', 'B'), set_colors=('skyblue', 'orange'), alpha=0.7, ax=axs[0, 1])
axs[0, 1].set_title('Overlapping Sets')

# plot 3: A and B are equal
A = set([1, 2, 3, 4, 5])
B = set([1, 2, 3, 4, 5])
venn2([A, B], set_labels=('A', 'B'), set_colors=('skyblue', 'orange'), alpha=0.7, ax=axs[1, 0])
axs[1, 0].set_title('Equal Sets')

# plot 4: A belongs to B
A = set([1, 2, 3])
B = set([1, 2, 3, 4, 5])
venn2([A, B], set_labels=('A', 'B'), set_colors=('skyblue', 'orange'), alpha=0.7, ax=axs[1, 1])
axs[1, 1].set_title('Subset')

# adjust layout
plt.tight_layout()

# show plots
plt.savefig('Venn Diagrams for numbers.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

<div style="text-align:center">
    <img src="Relationships-between-two-Sets/Venn%20Diagrams%20for%20numbers.png" alt="Venn Diagrams for numbers" style="zoom:67%;" />
</div>
###  Cardinality

$$
\begin{equation}
|A \cup B|=|A|+|B|-|A \cap B| \text {. }
\end{equation}
$$

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 4/3/23 17:10

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# create sets A and B
A = set([1, 2, 3, 4, 5])
B = set([4, 5, 6, 7, 8])

# calculate cardinality using the equation |A ∪ B| = |A| + |B| - |A ∩ B|
cardinality_union = len(A.union(B))
cardinality_A = len(A)
cardinality_B = len(B)
cardinality_intersection = len(A.intersection(B))
cardinality_sum = cardinality_A + cardinality_B - cardinality_intersection

# print the cardinality values
print("|A| =", cardinality_A)
print("|B| =", cardinality_B)
print("|A ∩ B| =", cardinality_intersection)
print("|A ∪ B| =", cardinality_union)
print("|A| + |B| - |A ∩ B| =", cardinality_sum)

# plot the equation using Venn diagrams
venn2([A, B], set_labels=('A', 'B'), set_colors=('skyblue', 'orange'), alpha=0.7)
plt.title('|A ∪ B| = |A| + |B| - |A ∩ B|')
plt.annotate('|A|', xy=(-0.6, 0), fontsize=14)
plt.annotate('|B|', xy=(0.5, 0), fontsize=14)
plt.savefig('Venn Diagrams for cardinality.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
```

<div style="text-align:center"> <img src="Relationships-between-two-Sets/Venn%20Diagrams%20for%20cardinality.png" alt="Venn Diagrams for cardinality" style="zoom:37%;" /> </div>



```markdown
|A| = 5
|B| = 5
|A ∩ B| = 2
|A ∪ B| = 8
|A| + |B| - |A ∩ B| = 8
```

### Cartesian product

> The Cartesian product of two sets $A$ and $B$ is the set
> $$
> A \times B=\{(a, b): a \in A, b \in B\}
> $$
> For example, $[0,1] \times[0,1]$ is the square $\{(x, y): x, y \in[0,1]\}$, and $\mathbb{R} \times \mathbb{R}=\mathbb{R}^2$ is two-dimensional Euclidean space.

```python
# Name: Mei Jiaojiao
# Profession: Artificial Intelligence
# Time and date: 4/3/23 17:37

import numpy as np
import matplotlib.pyplot as plt

# create two sets A and B
A = np.linspace(0, 1, 11)
B = np.linspace(0, 1, 11)

# compute the Cartesian product of A and B
C = [(a, b) for a in A for b in B]

# plot the resulting set
x, y = zip(*C)
plt.scatter(x, y, s=5)

# set plot limits and labels
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('A')
plt.ylabel('B')
plt.savefig('Cartesian Product of A and B.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.title('Cartesian Product of A and B')

# show plot
plt.show()
```

<div style="text-align:center"> <img src="Relationships-between-two-Sets/Cartesian%20Product%20of%20A%20and%20B.png" alt="Cartesian Product of A and B" style="zoom:33%;" /> </div>


### Reference

1. Blitzstein, J. K., & Hwang, J. (2019). Introduction to Probability (2nd ed.). CRC Press.

