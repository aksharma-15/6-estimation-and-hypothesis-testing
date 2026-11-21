# 6-estimation-and-hypothesis-testing
Fundamentals of Central Limit Theorem, population, samples and hypothesis testing for data science and machine learning.

---

# Statistical Inference & Hypothesis Testing Fundamentals

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

An empirical, simulation-driven exploration of foundational statistical inference concepts using synthetic ride-hailing telemetry data. This repository bridges theoretical probability distributions with practical data science workflows, demonstrating how sample statistics behave relative to true population parameters.

---

## 🧠 Overview & Core Concepts

### 1. Population vs. Sample
In statistical modeling, we rarely have access to every data point in a system. Understanding the distinction between the ground truth and our observed subset is critical:
* **Population (`N`):** The complete set of all observations of interest (e.g., all `4,328` simulated Uber rides over a 30-day window). The metrics derived from this set—such as the true mean (`mu`) and true standard deviation (`sigma`)—are fixed **parameters**.
* **Sample (`n`):** A randomly drawn subset of the population (e.g., `n = 200` rides). Metrics calculated from a sample—such as the sample mean (`x_bar`) and sample standard deviation (`s`)—are **statistics** used as point estimates for the unknown population parameters.

### 2. Sampling Variability & The Law of Large Numbers
Because a sample is a random subset, its summary statistics will naturally fluctuate from one draw to the next—a phenomenon known as **Sampling Variability**.
* **Small Samples (`n = 20`):** Exhibit high variance across independent draws. For instance, three random draws of 20 rides yielded average fares of `$305.57`, `$335.93`, and `$293.75`.
* **Large Samples (`n = 500`):** In accordance with the **Law of Large Numbers (LLN)**, as sample size increases, the sample mean converges tightly around the true population mean (`$317.86`, `$319.84`, and `$320.66` vs. the true population mean of `$318.45`).

### 3. The Central Limit Theorem (CLT)
The **Central Limit Theorem** is the cornerstone of frequentist inference. It states that if you draw sufficiently large random samples of size `n` from *any* population—regardless of how heavily skewed or non-normal the underlying distribution is—the **sampling distribution of the sample means** will approximate a **Normal (Gaussian) distribution** centered at the true population mean (`mu`).

In this notebook:
* **Underlying Distribution:** The raw `fare` variable is heavily right-skewed (`skewness = 2.64`) due to rare surge-pricing events (`2%` probability).
* **Sampling Distribution:** By taking `1,000` independent random samples of size `n = 50` and plotting their averages, the skewness drops dramatically to `0.46`, yielding a symmetric bell curve centered right on the ground-truth mean (`$318.53` vs. `$318.45`).

### 4. Standard Error of the Mean (SEM)
While the standard deviation (`s`) measures the dispersion of individual observations within a single sample, the **Standard Error (`SE`)** quantifies the precision of the sample mean as an estimate of the population mean (i.e., the standard deviation of the sampling distribution).

```text
SE = s / sqrt(n)
```

* `SE` = Standard Error of the Mean
* `s` = Sample Standard Deviation (estimating population `sigma`)
* `n` = Sample Size

**Empirical Verification:** 
In our simulation with `n = 50` across `1,000` trials, the empirical standard deviation of the sample means was `~13.93`. Multiplying this by `sqrt(50)` yields `98.48`, closely matching the true population standard deviation of `97.91`. Reducing `n` to `10` increases the `SE` to `31.91`, whereas increasing `n` to `100` shrinks the `SE` to `~9.79`—demonstrating the inverse square-root relationship between sample size and estimation uncertainty.

### 5. Confidence Intervals & Hypothesis Testing
Building directly on the CLT and Standard Error, the notebook sets the foundation for formal statistical inference:
* **Confidence Intervals (CIs):** Using the normality of the sampling distribution, we construct intervals around a sample mean (`x_bar +/- z * SE`). Across repeated sampling, a `95%` confidence interval will capture the true population parameter `mu` approximately `95%` of the time (visualized via the custom `show_intervals` helper function).
* **Hypothesis Testing:** Enables data-driven decision-making by evaluating how likely an observed sample statistic is under a baseline assumption—the **Null Hypothesis (`H_0`)**—versus an **Alternative Hypothesis (`H_a`)**. Using `scipy.stats`, we quantify this surprise via **test statistics** (e.g., `t`-scores or `z`-scores) and **p-values** to determine whether observed shifts in metrics (like trip duration or surge fares) are statistically significant or merely artifacts of sampling variability.

---

## 📊 Dataset Architecture

The notebook utilizes a deterministic synthetic data generator (`load_uber(seed=11)`) modeling **30 days (`720` hourly slots)** of urban ride-hailing operations (`4,328` total rides):

| Feature | Data Type | Generation Logic & Statistical Properties |
| :--- | :--- | :--- |
| `hour_slot` | `int` | Sequential hour index (`0` to `719`), arrival counts modeled via `Poisson(lambda=6)`. |
| `hour` | `int` | Hour of the day (`0` to `23`). |
| `is_peak` | `int` (binary) | `1` during rush hours (`8–10 AM`, `5–7 PM`), `0` otherwise. |
| `trip_minutes` | `float` | Approximately Normal (`mu=24`, `sigma=6`), left-truncated at `4.0` minutes. |
| `fare` | `float` | Linear function of duration (`45 + 11 * trip + Normal(0, 25)`) with `2%` surge multiplier (`2.0x–2.8x`), creating heavy right-skew. |
| `surge` | `int` (binary) | `1` if surge pricing was active (`p = 0.02`), `0` otherwise. |
| `cancelled` | `int` (binary) | Conditional Bernoulli (`p = 0.20` during peak hours, `p = 0.08` off-peak). |
| `is_fraud` | `int` (binary) | Ground-truth fraudulent trip indicator (`p = 0.01`). |
| `flagged` | `int` (binary) | Fraud detection system alert (`90%` true positive rate, `5%` false positive rate). |

---

## 💡 Key Analytical Takeaways

1. **Reproducibility via PRNG Seeding:** Explicitly passing `random_state` / `seed` ensures deterministic subsampling during debugging, whereas omitting it demonstrates true stochastic variance across Monte Carlo runs.
2. **Skewness Resilience:** Non-normal business metrics (like revenue, fares, or session lengths) do not invalidate parametric tests (`t`-tests, `z`-tests) provided the sample size `n` is large enough (`n >= 30` as a rule of thumb, though highly skewed data may require `n >= 50`) for the CLT to take effect.
3. **Diminishing Returns on Sample Size:** Because `SE` scales with `1 / sqrt(n)`, cutting estimation error in half requires **quadrupling (`4x`)** the sample size—a crucial cost-benefit consideration when designing A/B tests and data collection pipelines.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the core scientific computing stack:

```bash
pip install numpy pandas matplotlib scipy jupyter
```

### Running the Notebook
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/your-username/statistical-inference-clt.git](https://github.com/your-username/statistical-inference-clt.git)
   cd statistical-inference-clt
   ```
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open the `.ipynb` file and run the cells sequentially from top to bottom.
