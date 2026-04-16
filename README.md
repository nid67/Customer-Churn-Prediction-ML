### **PROJECT OVERVIEW:**
- **Dataset**: Telco Customer Churn (7,043 customers, 20 features)
- **Target Variable**: Churn (Binary Classification - Yes/No)
- **Objective**: Predict which customers are likely to leave the telecom service

### **DATASET FEATURES:**
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Services: InternetService, Contract Type, PhoneService, OnlineSecurity, TechSupport, etc.
- Account Info: Tenure (months), MonthlyCharges, TotalCharges, PaymentMethod
- Churn Rate: ~26.5% (Class Imbalanced)

### **MODELS USED:**
1. **Logistic Regression** - Baseline model for binary classification
2. **Random Forest** - Ensemble method for better performance

### **KEY TECHNIQUES IMPLEMENTED:**

**1. Class Imbalance Handling:**
   - Applied SMOTE (Synthetic Minority Oversampling Technique)
   - Used class weights in models
   - Original: 5,174 non-churn vs 1,869 churn
   - After SMOTE: Balanced 1:1 ratio

**2. Data Preprocessing:**
   - One-hot encoded all categorical variables
   - Handled missing values in TotalCharges
   - Standardized numeric features using StandardScaler

**3. Model Evaluation Metrics:**
   - Accuracy: % of correct predictions
   - Precision: % of predicted churners who actually churned
   - Recall/Sensitivity: % of actual churners identified
   - Specificity: % of non-churners correctly identified
   - F1-Score: Harmonic mean of Precision and Recall
   - AUC-ROC: Area under Receiver Operating Characteristic curve

**4. Threshold Tuning:**
   - Analyzed ROC curves to find optimal decision threshold
   - Evaluated F1-Score, Precision, and Recall at different thresholds
   - Default threshold = 0.5, but optimal threshold varies per model

**5. Business-Level Insights:**
   - Segmented customers into Risk Categories: Low/Medium/High
   - Based on predicted churn probability
   - Analyzed actual churn rate in each segment
   - Provides actionable recommendations for customer retention

### **VISUALIZATIONS GENERATED:**
1. **01_class_distribution.png** - Original churn rate (26.5% imbalanced)
2. **02_feature_distributions.png** - Distributions of top numeric features
3. **03_correlation_heatmap.png** - Correlation matrix of key features with churn
4. **04_confusion_matrices.png** - True/False positives and negatives for both models
5. **05_roc_curves.png** - ROC curves comparing both models (AUC comparison)
6. **06_precision_recall_curves.png** - Precision-Recall trade-off analysis
7. **07_threshold_tuning_*.png** - Optimal threshold analysis for each model
8. **08_feature_importance.png** - Top important features from Random Forest
9. **09_model_comparison.png** - Side-by-side metrics comparison
10. **10_risk_segments_*.png** - Customer segmentation by churn risk

### **EXPECTED RESULTS:**
- Both models achieve ~77-78% Accuracy
- Logistic Regression: Better interpretability
- Random Forest: Better generalization, captures non-linear patterns
- Threshold tuning improves F1-Score over default 0.5


- "Explain why Logistic Regression might be preferred over Random Forest for this use case"
- "How should I interpret the correlation heatmap for business decisions?"
- "What are the business implications of threshold tuning?"

