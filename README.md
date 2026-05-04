# Module 11 — Statistical Analysis and Inference Foundations

**Session Time:** 120 minutes  
**Primary Tools:** Python, Pandas, NumPy, SciPy, Matplotlib  
**Environment:** Google Colab or local Jupyter Lab

---

## Module Overview

In previous modules, students explored datasets, cleaned data, validated assumptions, and began looking for patterns. This module introduces the next major analytical skill:

> **Reasoning under uncertainty.**

Real-world data is noisy. Analysts rarely have access to an entire population. Instead, they usually work with a sample and use that sample to make a careful claim about a larger group.

This module focuses on the foundational ideas behind statistical inference:

- Sampling variability
- The Central Limit Theorem
- Confidence intervals
- Basic hypothesis testing


---

## Prerequisites

Students should already be comfortable with:

- Python fundamentals
- Lists, functions, and loops
- Pandas DataFrames
- Basic exploratory data analysis
- Summary statistics such as mean, median, and standard deviation
- Reading CSV files into Python

---

## Learning Objectives

By the end of this module, students will be able to:

- Explain the difference between a **sample** and a **population**
- Describe why different samples from the same population can produce different results
- Use simulation to demonstrate **sampling variability**
- Explain the Central Limit Theorem at a conceptual level
- Build and interpret a confidence interval
- Explain the purpose of a hypothesis test
- Run a simple hypothesis test in Python
- Communicate statistical uncertainty in plain language

---

## Big Idea

Exploratory data analysis helps us describe what we see in a dataset.

Statistical inference helps us answer a deeper question:

> **How confident are we that what we see in our sample reflects something true about the larger population?**

This module helps students move from:

> “The sample average is 52.”

to:

> “Based on this sample, we estimate that the true population average is likely between 49 and 55.”

That shift is the foundation of statistical reasoning.

---

## Session Breakdown

| Segment | Topic | Duration |
|---:|---|---:|
| 1 | Why Statistical Inference Matters | 10 min |
| 2 | Samples, Populations, and Sampling Variability | 20 min |
| 3 | Central Limit Theorem | 25 min |
| 4 | Confidence Intervals | 25 min |
| 5 | Basic Hypothesis Testing | 25 min |
| 6 | Wrap-Up and Reflection | 15 min |

---

# 1. Why Statistical Inference Matters

In real analysis, we often ask questions like:

- What is the average customer spend?
- Did a marketing campaign increase purchases?
- Are users satisfied with a product?
- Is one group behaving differently than another?

In most cases, we only observe part of the full picture.

That means we need tools for handling uncertainty.

---

## Key Vocabulary

| Term | Meaning |
|---|---|
| Population | The full group we want to understand |
| Sample | The subset of the population we actually observe |
| Parameter | A true value about the population |
| Statistic | A value calculated from a sample |
| Inference | Using sample data to make a careful claim about the population |

---

## Example

If a company has 100,000 customers but surveys 500 of them:

- The **100,000 customers** are the population
- The **500 surveyed customers** are the sample
- The true average satisfaction score for all customers is a population parameter
- The average satisfaction score from the 500 surveyed customers is a sample statistic

The sample statistic is useful, but it is not guaranteed to equal the true population parameter.

---

# 2. Setup

The following code can run in either:

- Google Colab
- Jupyter Lab
- Jupyter Notebook

If running locally, students may need to install the required libraries first.

```python
# If needed, uncomment and run this cell:
# !pip install pandas numpy scipy matplotlib
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

# Make results reproducible
np.random.seed(42)
```

---

# 3. Create a Simulated Population

For teaching purposes, we will create a fake population of customer purchase amounts.

In real projects, we usually do not have the full population. But simulation lets us create one so we can clearly see how sampling works.

```python
# Create a fake population of 100,000 customer purchase amounts
population_size = 100_000

purchase_amounts = np.random.gamma(
    shape=2.0,
    scale=25.0,
    size=population_size
)

population = pd.DataFrame({
    "purchase_amount": purchase_amounts
})

population.head()
```

---

## Inspect the Population

```python
population["purchase_amount"].describe()
```

```python
true_population_mean = population["purchase_amount"].mean()
true_population_std = population["purchase_amount"].std()

true_population_mean, true_population_std
```

---

## Visualize the Population

```python
plt.figure(figsize=(8, 5))
plt.hist(population["purchase_amount"], bins=50)
plt.title("Population Distribution of Purchase Amounts")
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")
plt.show()
```


---

# 4. Sampling Variability

## Teaching Goal

Students should understand that different samples produce different results.

A sample mean is not the truth. It is an estimate.

```python
sample_1 = population.sample(n=100, random_state=1)
sample_2 = population.sample(n=100, random_state=2)
sample_3 = population.sample(n=100, random_state=3)

sample_1_mean = sample_1["purchase_amount"].mean()
sample_2_mean = sample_2["purchase_amount"].mean()
sample_3_mean = sample_3["purchase_amount"].mean()

sample_1_mean, sample_2_mean, sample_3_mean
```

---

## Repeated Sampling Simulation

Now we will take many samples and calculate the mean of each sample.

```python
sample_size = 100
num_samples = 1_000

sample_means = []

for i in range(num_samples):
    sample = population["purchase_amount"].sample(n=sample_size, replace=True)
    sample_mean = sample.mean()
    sample_means.append(sample_mean)

sample_means = np.array(sample_means)

sample_means[:10]
```

```python
sample_means.mean(), sample_means.std()
```

---

## Visualize the Sample Means

```python
plt.figure(figsize=(8, 5))
plt.hist(sample_means, bins=40)
plt.axvline(true_population_mean, linestyle="--", label="True Population Mean")
plt.title("Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```


---

# 5. Central Limit Theorem

## Teaching Goal

Students should understand the Central Limit Theorem conceptually.

The Central Limit Theorem says:

> If we repeatedly take random samples from a population and calculate the mean of each sample, the distribution of those sample means will become approximately normal as sample size increases.

This is true even if the original population is not normally distributed, assuming the sample size is large enough and the samples are independent.

---

## Compare Different Sample Sizes

```python
def get_sample_means(data, sample_size, num_samples=1000):
    means = []

    for i in range(num_samples):
        sample = data.sample(n=sample_size, replace=True)
        means.append(sample.mean())

    return np.array(means)
```

```python
means_n_10 = get_sample_means(population["purchase_amount"], sample_size=10)
means_n_30 = get_sample_means(population["purchase_amount"], sample_size=30)
means_n_100 = get_sample_means(population["purchase_amount"], sample_size=100)
means_n_500 = get_sample_means(population["purchase_amount"], sample_size=500)
```

```python
plt.figure(figsize=(8, 5))
plt.hist(means_n_10, bins=40)
plt.title("Distribution of Sample Means: n = 10")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
plt.hist(means_n_30, bins=40)
plt.title("Distribution of Sample Means: n = 30")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
plt.hist(means_n_100, bins=40)
plt.title("Distribution of Sample Means: n = 100")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

```python
plt.figure(figsize=(8, 5))
plt.hist(means_n_500, bins=40)
plt.title("Distribution of Sample Means: n = 500")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

---

## Compare Standard Errors

The standard deviation of the sample means is called the **standard error**.

It measures how much sample means vary from sample to sample.

```python
summary = pd.DataFrame({
    "sample_size": [10, 30, 100, 500],
    "mean_of_sample_means": [
        means_n_10.mean(),
        means_n_30.mean(),
        means_n_100.mean(),
        means_n_500.mean()
    ],
    "standard_error": [
        means_n_10.std(),
        means_n_30.std(),
        means_n_100.std(),
        means_n_500.std()
    ]
})

summary
```


---

# 6. Confidence Intervals

## Teaching Goal

Students should understand a confidence interval as a range of plausible values for the population parameter.

A confidence interval gives us a way to communicate uncertainty.

Instead of saying:

> The average purchase amount is $50.

We say:

> Based on this sample, we estimate the true average purchase amount is likely between $46 and $54.

---

## Confidence Interval Formula

For a sample mean, a simple confidence interval is:

$$
\bar{x} \pm t^* \times \frac{s}{\sqrt{n}}
$$

Where:

| Symbol | Meaning |
|---|---|
| $\bar{x}$ | Sample mean |
| $t^*$ | Critical value from the t-distribution |
| $s$ | Sample standard deviation |
| $n$ | Sample size |
| $\frac{s}{\sqrt{n}}$ | Standard error |

---

## Build a Confidence Interval Manually

```python
sample = population["purchase_amount"].sample(n=100, random_state=11)

sample_mean = sample.mean()
sample_std = sample.std()
n = len(sample)

confidence_level = 0.95
alpha = 1 - confidence_level

t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
standard_error = sample_std / np.sqrt(n)

margin_of_error = t_critical * standard_error

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

sample_mean, ci_lower, ci_upper
```

```python
print(f"Sample mean: ${sample_mean:.2f}")
print(f"95% confidence interval: (${ci_lower:.2f}, ${ci_upper:.2f})")
print(f"True population mean: ${true_population_mean:.2f}")
```

---

## Interpret the Confidence Interval

A correct interpretation:

> Based on this sample and method, we estimate that the true population mean purchase amount is between the lower and upper bounds of the interval.

A more technical interpretation:

> If we repeated this sampling process many times, about 95% of the confidence intervals created this way would contain the true population mean.

A common incorrect interpretation:

> There is a 95% probability that the true mean is inside this specific interval.

That wording is not technically correct in classical frequentist statistics.

---

## Use SciPy to Calculate the Same Interval

```python
confidence_interval = stats.t.interval(
    confidence=0.95,
    df=n - 1,
    loc=sample_mean,
    scale=standard_error
)

confidence_interval
```

---

# 7. Confidence Interval Simulation


We will create many confidence intervals and check how often they contain the true population mean.

```python
def create_confidence_interval(sample, confidence=0.95):
    sample_mean = sample.mean()
    sample_std = sample.std()
    n = len(sample)

    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    standard_error = sample_std / np.sqrt(n)

    margin_of_error = t_critical * standard_error

    lower = sample_mean - margin_of_error
    upper = sample_mean + margin_of_error

    return lower, upper
```

```python
num_intervals = 100
sample_size = 100

intervals = []

for i in range(num_intervals):
    sample = population["purchase_amount"].sample(n=sample_size, replace=True)
    lower, upper = create_confidence_interval(sample)

    contains_true_mean = lower <= true_population_mean <= upper

    intervals.append({
        "lower": lower,
        "upper": upper,
        "contains_true_mean": contains_true_mean
    })

intervals_df = pd.DataFrame(intervals)
intervals_df.head()
```

```python
coverage_rate = intervals_df["contains_true_mean"].mean()
coverage_rate
```

```python
print(f"Percent of intervals containing the true mean: {coverage_rate:.1%}")
```

---

# 8. Basic Hypothesis Testing

## Teaching Goal

Students should understand hypothesis testing as a structured way to ask:

> Could this result reasonably happen by chance?

We will focus on the logic of hypothesis testing, not heavy formulas.

---

## Key Terms

| Term | Meaning |
|---|---|
| Null hypothesis | The default assumption; usually says there is no effect or no difference |
| Alternative hypothesis | The claim we are looking for evidence to support |
| p-value | How surprising our result would be if the null hypothesis were true |
| Significance level | The cutoff we use to decide whether the result is statistically significant |
| Statistically significant | Unlikely enough under the null hypothesis that we reject the null |

---

## Business Example

Suppose the company claims that the average customer purchase amount is $50.

We collect a sample and want to test whether the true average purchase amount is different from $50.

Our hypotheses are:

$$
H_0: \mu = 50
$$

$$
H_A: \mu \neq 50
$$

Where:

- $H_0$ is the null hypothesis
- $H_A$ is the alternative hypothesis
- $\mu$ is the true population mean

---

## Run a One-Sample t-Test

```python
sample = population["purchase_amount"].sample(n=100, random_state=25)

hypothesized_mean = 50

t_statistic, p_value = stats.ttest_1samp(
    a=sample,
    popmean=hypothesized_mean
)

t_statistic, p_value
```

```python
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("The sample provides evidence that the true mean is different from $50.")
else:
    print("Fail to reject the null hypothesis.")
    print("The sample does not provide strong evidence that the true mean is different from $50.")
```

---

## Interpret the p-value

A p-value is not the probability that the null hypothesis is true.

A better beginner-friendly interpretation:

> If the true average purchase amount really were $50, the p-value tells us how surprising it would be to observe a sample result this extreme or more extreme.

If the p-value is small, the result is surprising under the null hypothesis.

If the p-value is large, the result is not very surprising under the null hypothesis.

---

## Important Warning

Statistical significance does not automatically mean practical importance.

A result can be statistically significant but too small to matter in the real world.

Example:

> A website change increases average purchase amount by $0.03.

With a huge sample size, that difference might be statistically significant.

But from a business perspective, it may not be meaningful.

---

# 9. Wrap-Up Reflection

Use these questions to close the module:

1. Why does sampling variability matter?
2. Why is the sample mean usually not exactly equal to the population mean?
3. What does the Central Limit Theorem help us understand?
4. Why is a confidence interval more informative than a single sample mean?
5. What does a p-value tell us?
6. Why should statistical significance not be treated as the same thing as practical importance?

---


# Optional Extension: Two-Sample Hypothesis Test

If time allows, instructors can introduce a two-sample test.

Example scenario:

> Did customers in Group A spend a different amount than customers in Group B?

```python
group_a = np.random.gamma(shape=2.0, scale=25.0, size=200)
group_b = np.random.gamma(shape=2.0, scale=28.0, size=200)

group_a.mean(), group_b.mean()
```

```python
t_statistic, p_value = stats.ttest_ind(
    group_a,
    group_b,
    equal_var=False
)

t_statistic, p_value
```

```python
alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("The groups appear to have different average purchase amounts.")
else:
    print("Fail to reject the null hypothesis.")
    print("The sample does not provide strong evidence of a difference between groups.")
```

## Instructor Note

This optional extension previews A/B testing without requiring students to fully master experimental design yet.

---

# Resources

- [SciPy Statistics Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NumPy Random Sampling Documentation](https://numpy.org/doc/stable/reference/random/index.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [OpenIntro Statistics](https://www.openintro.org/book/os/)
- [StatQuest: Confidence Intervals](https://www.youtube.com/user/joshstarmer)
