# Customer Churn Prediction

**Aryan Pathania (Ar3missss)**

An end-to-end machine learning pipeline to predict customer churn using a real-world telecom dataset of 7,000+ records. Covers data cleaning, feature engineering, exploratory data analysis, model training, and evaluation.

---

## Dataset

Telco Customer Churn — available on Kaggle  
*https://www.kaggle.com/datasets/blastchar/telco-customer-churn*

---

## Objectives

- What does the churn distribution look like in the dataset?
- Which features have the strongest relationship with churn?
- How do tenure, monthly charges, and contract type affect churn?
- Which demographic and service features are most associated with churn?
- Can we build an ML model that reliably predicts churn?
- Which model performs best — Logistic Regression, Random Forest, or XGBoost?
- What are the most important features driving churn predictions?

---

## Tech Stack

- **Language** — Python
- **Data** — Pandas, NumPy
- **Visualisation** — Matplotlib, Seaborn
- **Modelling** — Scikit-learn, XGBoost
- **Notebook** — Jupyter

## Key Findings

- ~27% of customers churned — the dataset has a moderate class imbalance
- Month-to-month contract customers churn at ~43%, far higher than annual or two-year customers (~3–11%)
- New customers (0–1 yr tenure) are the highest-risk group; churn drops steadily with tenure
- Churned customers pay on average ~$13/month more than retained customers
- Customers without Online Security or Tech Support churn significantly more
- Senior citizens churn at ~41% vs ~24% for non-seniors; gender has almost no effect
- `TotalCharges`, `tenure`, `MonthlyCharges`, and `Contract` are the top predictors across both tree-based models


## Results

Model  =  Logistic Regression , Accuracy = ~80% , ROC-AUC = ~0.85        
Model  =  Random Forest , Accuracy = ~79% , ROC-AUC = ~0.83 
Model  =  XGBoost , Accuracy = ~89% , ROC-AUC = ~0.91

- XGBoost achieved the best performance and was selected as the final model.



## How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place it in the `data/` folder
3. Create the output directory: `outputs/churn/`
4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost notebook
```

5. Launch the notebook

```bash
jupyter notebook churn-prediction.ipynb
```

---

