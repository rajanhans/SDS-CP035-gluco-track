# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

**Q:** Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
**A:** No missing values were found across the 253,680 instances. A total of **24,206 rows (~9.5%)** were flagged as duplicates and removed to avoid bias from repeated observations. No formatting issues were present. However, all features were stored as `float64`; several should instead be integers or categoricals.

**Q:** Are all data types appropriate (e.g., numeric, categorical)?  
**A:** Not fully. While numeric measures (e.g., BMI, MentHlth, PhysHlth) were correctly stored as floats, many **binary** and **ordinal** variables were misclassified as continuous. We corrected these using domain-informed mappings.

**Feature types overview**

| Type                 | Columns                                                                                                                                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Binary categorical   | Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk                                                                   |
| Ordinal / Categorical| GenHlth, Age, Education, Income                                                                                                                                                                                                           |
| Numeric              | BMI, MentHlth, PhysHlth                                                                                                                                                                                                                   |

**Q:** Did you detect any constant, near-constant, or irrelevant features?  
**A:** No constant features. At this stage nothing was dropped as irrelevant; all were retained for modeling and later importance checks.

---

### 🎯 2. Target Variable Assessment

**Q:** What is the distribution of `Diabetes_binary`?  
**A:** After removing duplicates, the target remains **imbalanced**: **84.7% = 0 (no diabetes)** vs **15.3% = 1 (diabetes)**.

**Q:** How might this imbalance influence metrics/strategy?  
**A:** Accuracy is misleading. We’ll emphasize **Recall**, **Precision**, **F1**, **ROC-AUC**, and **PR-AUC**, and use strategies like **class weights** and **resampling (SMOTE)**.

---

### 📊 3. Feature Distribution & Quality

**Q:** Which numerical features are skewed or contain outliers?  
**A:** Using a Z-score > 3, **15,328** potential outlier rows were flagged across **BMI, MentHlth, PhysHlth**. MentHlth and PhysHlth are strongly right-skewed; BMI is moderately right-skewed.

**Q:** Any unrealistic/problematic values?  
**A:** None outside expected ranges, but many statistically unusual values exist (especially MentHlth, PhysHlth). These may affect certain models.

**Q:** Helpful transformations?  
**A:**  
- MentHlth, PhysHlth: **log(x+1)** or **sqrt**.  
- BMI: **log/sqrt/Box-Cox/Yeo-Johnson** or **quantile** transforms.  
Choice will be validated empirically.

---

### 📈 4. Feature Relationships & Patterns

**Q:** Categorical patterns vs `Diabetes_binary`?  
**A:**  
- **GenHlth**: Worse self-reported health → higher diabetes prevalence.  
- **PhysActivity**: No activity 21.14% diabetic vs 11.61% if active.  
- **Smoker**: Slightly higher prevalence among smokers (16.29% vs 12.06%).

**Q:** Pairwise relationships/multicollinearity?  
**A:** Moderate, intuitive correlations (e.g., GenHlth with PhysHlth ~0.42; Income with Education ~0.45). None exceeded 0.7–0.8, so no severe multicollinearity concerns.

---

### 🧰 5. EDA Summary & Preprocessing Plan

**Top takeaways**
- **Severe class imbalance** (15.3% positive).  
- **Skewness/outliers** in MentHlth/PhysHlth (and some in BMI).  
- **GenHlth** strongly associated with diabetes.  
- **Lifestyle** factors (PhysActivity, Smoking) show signal.  
- **Correlations** moderate and manageable.

**Planned preprocessing**
- **Scaling** numeric features (BMI, MentHlth, PhysHlth; also Age/Income if used continuously).  
- **Encoding**: Binary already 0/1; Ordinal (GenHlth, Education, Age bands, Income bands) preserved as ordered integers.  
- **Exclusion**: None at this stage. Consider PCA if needed.

**Cleaned data shape**: **(229,474 rows, 22 columns)** after duplicate removal.

---

## ✅ Week 2: Feature Engineering & Preprocessing

### 🏷️ 1. Encoding

- **Binary columns** (already 0/1):  
  `['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex','Diabetes_binary']` → **kept as is**.
- **Ordinal columns**:  
  - `GenHlth`: 1=Excellent … 5=Poor (**preserve order**).  
  - `Education`: 1=None … 6=College Grad (**preserve order**).  
- **Nominal columns**: None identified → **no one-hot** needed (if introduced later, we’ll one-hot to avoid false ordering).
- **Scaling**: Applied to continuous features (BMI, MentHlth, PhysHlth, etc.) with **StandardScaler** (fit on train only).

### ✨ 2. Feature Creation

- **`BMI_category`** (CDC cutoffs):  
  - Underweight < 18.5; Normal 18.5–24.9; Overweight 25–29.9; Obese ≥ 30.  

  | Category    | Count  | Percent |
  |-------------|--------|---------|
  | Overweight  | 93,749 | ~37%    |
  | Obese       | 87,851 | ~35%    |
  | Normal      | 68,953 | ~27%    |
  | Underweight | 3,127  | ~1%     |

  *Insight:* ~72% Overweight/Obese → strong nonlinearity; the category feature can help.

- **`TotalHealthDays` = `PhysH**

### ✂️ 3. Data Splitting

- **80/20 split**, **stratified** on `Diabetes_binary`.  
- Shapes (example from run):  
  - `X_train`: **(202,944, 23)**  
  - `X_test`: **(50,736, 23)**  
  - `y_train`: **(202,944)**  
  - `y_test`: **(50,736)**

**Why split before SMOTE/scaling?** To avoid **data leakage**. Fit resampling/scalers on **train only**.

### ⚖️ 4. Imbalance Handling & Final Preprocessing

- **SMOTE** on training only:  
  - Before: `Counter({0: 174,667, 1: 28,277})`  
  - After:  `Counter({0: 174,667, 1: 174,667})`  
  - `X_train_resampled`: **(349,334, 25)**; `y_train_resampled`: **(349,334)**

- **Scaling**: StandardScaler **fit on train**, then transform train/test.  
  - `X_train_scaled`: **(349,334, 25)**  
  - `X_test_scaled`: **(91,258, 25)**

---

## ✅ Week 3: Model Development & Experimentation

### 🤖 1. Baselines

Initial baselines (test set):

| Model               | Accuracy | Precision | Recall  | F1     | AUC    |
|---------------------|---------:|----------:|--------:|-------:|-------:|
| Naive Bayes         | 0.666    | 0.259     | 0.748   | 0.384  | 0.700  |
| Decision Tree       | 0.727    | 0.238     | 0.437   | 0.308  | 0.605  |
| Logistic Regression | 0.666    | 0.259     | 0.748   | 0.384  | 0.700  |

**Notes:** NB shows strong **recall** (catching positives), LR offers a better balance, DT underperforms on AUC.

### 🧪 2. Extended models & comparison

Logged runs (via MLflow):

| Model                | Accuracy | Precision | Recall  | F1     | AUC    |
|----------------------|---------:|----------:|--------:|-------:|-------:|
| Naive Bayes          | 0.644    | 0.256     | 0.813   | 0.389  | 0.715  |
| Decision Tree        | 0.750    | 0.254     | 0.412   | 0.314  | 0.608  |
| Logistic Regression  | 0.727    | 0.297     | 0.701   | 0.417  | 0.716  |
| Random Forest        | **0.780**| **0.308** | 0.466   | 0.371  | 0.648  |
| Gradient Boosting    | 0.720    | 0.295     | 0.730   | **0.420** | **0.724** |
| k-Nearest Neighbors  | 0.720    | 0.268     | 0.583   | 0.367  | 0.663  |

**Takeaways:**  
- **Recall priority** → NB & GB strongest at catching positives;  
- **Overall balance** → **Gradient Boosting** has best F1/AUC; **Random Forest** best Accuracy/Precision.

### 📈 3. Experiment tracking (MLflow)

- Tracked: algorithm, key hyperparameters, train/test metrics (Accuracy, Precision, Recall, F1, AUC), timestamp, data version.  
- Benefit: **side-by-side comparability**, reproducibility, and quick identification of promising settings.

### 🕵️ 4. Error Analysis

- **Confusion matrix** focus: minimize **FN** (missed diabetics).  
- Example (Logistic Regression): more **FP (11,728)** than **FN (2,116)**. Threshold tuning and class weights can rebalance.


### 📝 5. Model Selection & Insights

- **Current best direction:** **Gradient Boosting** (best F1/AUC) with tuning; LR and RF are competitive.  
- **Top insights:**  
  1) **Class imbalance drives** metric choice; SMOTE helped learning.  
  2) **Precision–Recall trade-off** dictates thresholding for healthcare.  
  3) **Ensembles** (GB, RF) generally dominate single trees.  
  4) **Feature engineering** (BMI_category, TotalHealthDays) adds interpretability and signal.  
  5) **Hyperparameter tuning** likely to improve GB/LR/RF further.

**Non-technical summary:** The model screens for diabetes risk well, emphasizing **catching true cases** (recall). Some healthy people may be flagged (precision trade-off), which is manageable via inexpensive follow-up tests and threshold adjustments.

---

## ✅ Week 4: Tuning & Finalization (Preview)

## 🛠️ 1. Hyperparameter Tuning

**Q: Which hyperparameters did you tune for your models, and what methods (e.g., grid search, random search) did you use?**
A: 
For Gradient Boosting, the hyperparameters tuned were n_estimators, learning_rate, and max_depth.
For Logistic Regression, the hyperparameter tuned was C.
For Random Forest, the hyperparameters tuned were n_estimators and max_depth.
The method used for hyperparameter tuning for all three models was Grid Search with 5-fold Stratified Cross-Validation.

![alt text](image-3.png)

**Q: How did you select the range or values for each hyperparameter?**
A:
The ranges or values for each hyperparameter were selected based on common practices and a starting point for exploring the parameter space for each specific algorithm. They represent a small subset of possible values to demonstrate the tuning process within a reasonable computation time. For a more exhaustive search, wider ranges and potentially more values would be explored.
Gradient Boosting: n_estimators (100, 200) for exploring the impact of more trees; learning_rate (0.01, 0.1) for different step sizes; max_depth (3, 4) for controlling tree complexity.
Logistic Regression: C (0.1, 1.0, 10.0) to explore different levels of regularization strength (smaller values mean stronger regularization).
Random Forest: n_estimators (100, 200) for different numbers of trees; max_depth (10, 20) for controlling tree depth.

**Q: What impact did hyperparameter tuning have on your model’s performance?**
A:
Gradient Boosting: Tuning led to slight increases in Accuracy, Precision, F1-score, and AUC on the test set compared to the initial model, but a decrease in Recall.
Logistic Regression: Tuning resulted in very minimal changes across most metrics on the test set, suggesting that the default hyperparameters (or those used initially) were already close to the optimal within the defined grid, or the grid was too limited to show a significant impact.
Random Forest: Tuning resulted in a decrease in Accuracy, Precision, F1-score, and AUC on the test set, but a significant increase in Recall. This indicates a trade-off where the tuned model is better at identifying positive cases but at the cost of overall accuracy and precision within this specific parameter search.

## 🔄 2. Cross-Validation
**Q: How did you use cross-validation to assess model stability and generalization?**
A:  cross-validation was a key part of the hyperparameter tuning process to find generalized parameters, and its use provided some implicit insights into stability across training folds.


**Q: What were the results of your cross-validation, and did you observe any variance across folds?**
A:Cross-validation Results for Best Tuned Models:

Gradient Boosting:
  Best parameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
  Mean cross-validation recall: 0.8037
  Standard deviation of cross-validation recall: 0.0036

Logistic Regression:
  Best parameters: {'C': 0.1, 'penalty': 'l2'}
  Mean cross-validation recall: 0.7771
  Standard deviation of cross-validation recall: 0.0023

Random Forest:
  Best parameters: {'max_depth': 20, 'n_estimators': 200}
  Mean cross-validation recall: 0.8975
  Standard deviation of cross-validation recall: 0.0021

Observation on Variance Across Folds:
A smaller standard deviation relative to the mean indicates less variance across the cross-validation folds, suggesting better model stability and more reliable performance estimates on unseen data.
Based on the standard deviations above, you can observe the degree of variance in recall performance across the 5 cross-validation folds for each of the best tuned models.

**Q: Why is cross-validation important in this context?**
A: In the context of this diabetes prediction task, where the dataset might have underlying complexities and potential imbalances, using cross-validation helps ensure that the performance metrics we obtain are not just a result of a lucky data split but are more representative of the model's true capability to generalize to new patients' data. It gives us more confidence in the chosen model and hyperparameters.

## 🏆 3. Final Model Selection
**Q: How did you choose your final model after tuning and validation?**
A: Considering the critical need for high Recall in this application, the Tuned Gradient Boosting model was identified as the most suitable choice among the evaluated models because it achieved the highest Recall while also maintaining a strong F1-score and AUC, indicating a good balance of performance relevant to the problem.

**Q: Did you retrain your final model on the full training set before evaluating on the test set? Why or why not?**
A: while the best hyperparameters were identified using cross-validation on the training data, the final models were not explicitly retrained on the full training set before evaluating on the test set in this notebook. This is a deviation from a common best practice aimed at maximizing the model's performance using all available training data.

**Q: What were the final test set results, and how do they compare to your validation results?**
A: Final Test Set Results (from metrics_df):
                            ROC AUC  	PR AUC	   F1 (Macro)	Balanced Accuracy	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	        0.787777    0.337009    0.619486	      0.716043	           NaN	     NaN	     NaN	 NaN	    NaN
Random Forest	            0.753711    0.280562    0.618815	      0.648228	           NaN	     NaN	     NaN	 NaN	    NaN
XGBoost                 	0.782618    0.335176    0.616632	      0.706232	           NaN	     NaN         NaN	 NaN        NaN
Gradient Boosting (Tuned)	  NaN	     NaN	     NaN	         NaN	         0.723786   0.295193	0.7080  0.41666    0.717179
Logistic Regression (Tuned)   NaN	     NaN	     NaN	         NaN	         0.727097	0.296962	0.7010	0.417207   0.716198
Random Forest (Tuned)	      NaN	     NaN	     NaN	         NaN             0.758436	0.309847	0.5978	0.408151   0.691129

Cross-validation Results for Best Tuned Models (from cv_results_df):

                              Best Parameters	                               Mean CV Recall	   Std Dev CV Recall
Gradient Boosting (Tuned)	 {'learning_rate': 0.1, 'max_depth': 4, 'n_esti...	 0.803661	         0.003619
Logistic Regression (Tuned)  {'C': 0.1, 'penalty': 'l2'}	                     0.777107	         0.002254
Random Forest (Tuned)	     {'max_depth': 20, 'n_estimators': 200}              0.897519	         0.002102

while cross-validation provided an estimate of performance during tuning, the final test set results offer the most realistic measure of how the tuned models are expected to perform on new, unseen data. The observed drops in performance highlight the importance of having a separate, untouched test set for final evaluation.

## 📊 4. Feature Importance & Interpretation

**Q: How did you assess feature importance for your final model?**
A: For tree-based models like Gradient Boosting, we can access the feature_importances_ attribute after the model has been trained. This attribute provides a numerical score for each feature, indicating its relative importance in the model's decision-making process. 

**Q: Which features were most influential in predicting diabetes risk, and do these results align with domain knowledge?**
A:
Feature Importances for Tuned Gradient Boosting Model:
GenHlth                    0.313986
HighBP                     0.149589
Age                        0.114815
BMI_category_Normal        0.106818
BMI_category_Overweight    0.064931
HvyAlcoholConsump          0.039628
Income                     0.033734
PhysActivity               0.029984
BMI_category_Obese         0.027193
HighChol                   0.023692
NoDocbcCost                0.018038
Education                  0.016177
TotalHealthDays            0.011958
Fruits                     0.011690
Smoker                     0.009383
CholCheck                  0.008054
Sex                        0.007266
Stroke                     0.004465
DiffWalk                   0.003309
Veggies                    0.002634
HeartDiseaseorAttack       0.001907
AnyHealthcare              0.000750

The most influential features identified by the tuned Gradient Boosting model – particularly General Health, High Blood Pressure, Age, and BMI category – strongly align with established medical domain knowledge about key risk factors for diabetes. This increases confidence in the model's ability to capture meaningful relationships in the data.


**Q: How would you explain your model’s decision process to a non-technical audience?**
A: In simple terms, our model is like an expert that uses a weighted checklist of health factors to calculate a risk score, and then predicts your diabetes risk based on that score. The factors that get the most "weight" on the checklist are the ones we identified as most influential.

It's important to remember that this is a prediction based on the data it learned from, not a definitive diagnosis. It's a tool to help identify individuals who might be at higher risk and could benefit from further medical evaluation.


