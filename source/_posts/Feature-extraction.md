---
title: Feature extraction and One-hot embedding
mathjax: true
hidden: false
comments: true
date: 2023-04-27 18:29:46
categories: Machine learning
tags:
  - Machine learning
  - Linear algebra
  - Feature extraction
  - Artificial intelligence
  - Deep learning
  - One-hot embedding
  - Multi-hot embedding
description: Feature extraction refers to the process of extracting meaningful and representative information from raw data that can be used for data analysis, pattern recognition, machine learning and other tasks. In fields such as computer vision, natural language processing, and signal processing, feature extraction is a crucial step.
top: 15
---

# What is feature extraction

Feature extraction is the process of identifying and selecting the most relevant information from a given set of data and transforming it into numerical features that can be used for further analysis, modeling, or comparison. The goal of feature extraction is to capture the essential characteristics of the data that are most informative for a specific task or problem, while reducing the complexity and redundancy of the data. 

When describing a person, an event, a scene, etc., we often use some data, such as describing midterm exam scores of someone where Chinese, Mathematics, and English scores are 120, 143, and 130 respectively, with an average score of 131 and a variance of $(11^2+12^2+1^2) /3 = 88.6$. We can use a vector [120, 143, 130, 131, 88.6] to represent the grades of a student. This process is called feature extraction.

For another example, let's say a programmer is 24 years old, 187 cm tall, earns a salary of 3000 euros, has a partner, and owns an independent property. We can represent this programmer using a vector [24, 187, 3000, 1, 1], where the last two values indicate the presence or absence of a partner and independent property respectively (1 represents "yes" and 0 represents "no"). This vector can be used as a feature vector to represent the characteristics of this programmer for further analysis, modeling, or comparison with other programmers.

# To be relevant and necessary

In feature extraction, it is necessary to identify and select features that are relevant and informative for the specific task or problem, while also considering the limitations and requirements of the data representation. For example, the color of a vehicle can be considered a feature, but it would not be practical to represent it using a three-dimensional vector with values ranging from 0 to 255, as in real life, there are usually only a limited number of common colors for vehicles, such as red, black, gray, white, green, etc. In fact, it may be possible to represent the color of a vehicle using just one dimension or feature, such as a categorical variable that indicates the specific color or a numerical value that corresponds to a specific color code. 

Therefore, in feature extraction, it is important to carefully consider the necessary and sufficient features that can effectively capture the essential characteristics of the data while minimizing the complexity and redundancy of the feature representation.

# One-hot embedding

Before introducing one-hot embedding, it should be noted that some features may not be directly convertible into numerical features, such as a person's interests or profession. For example, a person may be interested in both computer science and law, but there are no obvious numerical features to describe these interests. In such cases, we need a method to convert these categorical features into manageable numerical features, which is where one-hot embedding comes in.

For example, suppose we are analyzing a dataset of job applicants, and one of the categorical features is the applicant's highest education level, which can take on values such as "High School Diploma", "Bachelor's Degree", "Master's Degree", or "Ph.D." We can use one-hot embedding to convert this categorical feature into a set of numerical features that can be used for further analysis or modeling. We can represent each possible education level as a binary vector of length four, where the position of the 1 corresponds to the specific education level. For example, the one-hot encoded vectors for each education level would be:

- "High School Diploma": [1, 0, 0, 0]
- "Bachelor's Degree": [0, 1, 0, 0]
- "Master's Degree": [0, 0, 1, 0]
- "Ph.D.": [0, 0, 0, 1]

If a person has two master's degrees, we cannot simply use the same one-hot encoding approach, since it would require two positions to be 1, violating the one-hot constraint. In this case, we can use a modified approach called "multi-hot" encoding, where we assign a non-zero value to each position that corresponds to a degree the person holds. For example, we can represent a person with two master's degrees using the following multi-hot encoded vector:

- "Master's Degree": [0, 0, 2, 0]

This means that the person holds two master's degrees, which are represented by the values 2 at the third position of the vector. Note that in multi-hot encoding, the values of the vector are no longer binary and can take on any non-negative value. This approach is useful for cases where a categorical feature can have multiple values or when the frequency of occurrence of a value is important.

# Limitations and drawbacks

One-hot embedding has several limitations and drawbacks:

- Sparsity: With a large number of unique words in a text corpus, the resulting one-hot or multi-hot encoded vectors can be extremely sparse, with most elements being 0. This can lead to computational inefficiencies and difficulty in analyzing the data.
- Order: These encoding methods do not take into account the order of the words in a sentence, leading to loss of information about the sentence structure and context. For example, "I love pizza" and "Pizza loves me" would have the same multi-hot encoding, even though the meaning is opposite.
- Semantics: One-hot and multi-hot encodings cannot capture semantic similarity between words. For example, "car" and "vehicle" are semantically similar, but would have different one-hot or multi-hot encodings.
- Dimensionality: With a large vocabulary, the one-hot or multi-hot encoding vectors can become very high-dimensional, leading to issues with memory and computational resources.

To address these issues, alternative encoding methods such as word embeddings and contextualized embeddings have been developed, which can capture the semantic relationships and context of words in a more efficient and effective way. 

# Applications

## Example 1

While one-hot and multi-hot encodings have some limitations, they can still be useful in certain cases. For example, if we want to represent gender as a binary feature, using 0/1 encoding could imply a ranking or hierarchy between the values, which may not be appropriate. Instead, using a one-hot encoding where each gender is represented by a vector with only one non-zero value can better capture the categorical nature of the feature. For example, representing gender as "male" and "female" using one-hot encoding would result in:

Male: $[0]^{\mathrm{T}}$. Female: $[1]^{\mathrm{T}}$. 

Male: $[0,1]^{\mathrm{T}}$. Female: $[1,0]^{\mathrm{T}}$.

In this case, the second encoding approach would be more appropriate as it accurately represents the categorical nature of gender without implying any hierarchy or ranking between the values.

## Example 2

Suppose we want to represent age as a categorical feature, where the difference between, say, 15 and 16 years old is not significant. We can divide the age range of 0 to 100 into four categories and use a 4-dimensional vector to represent them. Here are the feature vectors for each age category:

- Children: $[1,0,0,0]^{\mathrm{T}}$, for ages 0 to 18
- Young Adults: $[0,1,0,0]^{\mathrm{T}}$, for ages 18 to 40
- Middle-aged Adults: $[0,0,1,0]^{\mathrm{T}}$, for ages 40 to 60
- Elderly: $[0,0,0,1]^{\mathrm{T}}$, for ages 60 and above

This approach allows us to represent age as a categorical feature with a more compact and informative feature vector, which can be useful in various data analysis and machine learning tasks.

But the difference between 17 and 19 years old may not be significant enough to warrant them being classified as completely separate categories.

To address this issue, we can improve the age category representation by allowing for some overlap between the categories. This can be achieved by defining the age ranges to have some overlap, so that adjacent age categories share some common age ranges. For example, we can define the age categories as "Children (0-18)", "Young Adults (15-40)", "Middle-aged Adults (35-60)", and "Elderly (55-100)". Using this approach, a person who is 36 years old would be represented by a feature vector of $[0,1,1,0]^{\mathrm{T}}$, indicating that they exhibit characteristics of both the "Young Adults" and "Middle-aged Adults" age categories.

# Summary

In summary, feature extraction is highly dependent on the specific context and problem being addressed. There is no one-size-fits-all solution or algorithm, and the most appropriate approach depends on the specific requirements and limitations of the data representation. One-hot and multi-hot embeddings have some limitations, such as sparsity, order, and semantics, but they can still be useful in certain cases. 

# Reference

1. 卢菁. (2021). 速通机器学习。电子工业出版社.

