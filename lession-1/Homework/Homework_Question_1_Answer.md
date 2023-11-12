
# Homework Question 1 Answer

## Question
How would you normalize the data related to children born in the past year using the field 'mothers_age'? Why did you choose that method?

## Answer
Given the distribution of the 'mothers_age' data, sourced from the March of Dimes [March of Dimes Data](https://www.marchofdimes.org/peristats/data?reg=99&top=2&stop=2&lev=1&slev=1&obj=1), where:
- 4.4% of births were to women under 20
- 46.7% to women aged 20-29
- 45.4% to women aged 30-39
- 3.6% to women aged 40 and older

The data is not uniformly distributed and appears to be skewed with the majority of data concentrated in the middle age groups.

Given this distribution pattern, I would choose Z-Score Normalization (Standardization) as the normalization technique. This method transforms the data to have a mean of 0 and a standard deviation of 1, making it suitable for machine learning algorithms that assume a normal distribution of input features. It is particularly useful here due to the concentration of data in the middle age groups and the presence of lower percentages at the extremes, which may indicate skewness.

Z-Score Normalization is beneficial because it maintains the relative distribution of ages while standardizing their scale. This is important as the majority of the data lies within a specific age range, and this method will ensure that the model does not bias towards the less represented age groups. Additionally, if the model being used is sensitive to the scale of input features (like in algorithms that use gradient descent), standardizing the data helps in faster convergence.

In conclusion, Z-Score Normalization is the most appropriate for this dataset due to its skewed distribution and the need to standardize the age data for effective modeling, especially in models where the assumption of normally distributed input features improves performance and accuracy.

## Sample Data Sets
```python
mothers_age = [4.4, 46.7, 45.4, 3.6]  # Percentages of births by age group
```

## Sample Code
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example 'mothers_age' data
mothers_age = np.array(mothers_age).reshape(-1, 1)  # Reshaped for compatibility with StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply Z-score normalization
mothers_age_normalized = scaler.fit_transform(mothers_age)

print("Normalized Mothers' Age Data:", mothers_age_normalized)
```
