"""
Customer Churn Prediction Using Supervised Machine Learning
Includes: Class Imbalance Handling, Threshold Tuning, Business Insights
Models: Logistic Regression, Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score, precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==================== 1. LOAD DATASET ====================
def load_churn_dataset(filepath):
    """Load real Telco Customer Churn dataset"""
    df = pd.read_csv(filepath)
    
    # Handle TotalCharges - convert to numeric, replace empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert Churn to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Drop customerID as it's not a feature
    df = df.drop('customerID', axis=1)
    
    return df

# ==================== 2. DATA PREPROCESSING ====================
def preprocess_data(df):
    """Preprocess Telco dataset - encode categorical variables"""
    df_processed = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

# ==================== 3. CLASS IMBALANCE HANDLING ====================
def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE for class imbalance"""
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"✓ SMOTE Applied:")
    print(f"  Original distribution: {y_train.value_counts().to_dict()}")
    print(f"  Balanced distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    return X_train_balanced, y_train_balanced

# ==================== 4. MODEL TRAINING ====================
def train_models(X_train, y_train, X_test, y_test):
    """Train LR and RF models with class weights"""
    models = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    return models

# ==================== 5. EVALUATION METRICS ====================
def evaluate_models(models, X_test, y_test):
    """Comprehensive model evaluation"""
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Specificity': specificity,
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return results

# ==================== 6. THRESHOLD TUNING ====================
def find_optimal_threshold(y_test, y_pred_proba, metric='f1'):
    """Find optimal classification threshold"""
    thresholds = np.arange(0, 1.01, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_test, y_pred_adjusted)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred_adjusted, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred_adjusted, zero_division=0)
        scores.append(score)
    
    optimal_threshold = thresholds[np.argmax(scores)]
    return optimal_threshold, np.array(scores), thresholds

# ==================== 7. BUSINESS-LEVEL INSIGHTS ====================
def business_analysis(y_test, y_pred_proba, model_name):
    """Analyze churn risk segments and business impact"""
    df_analysis = pd.DataFrame({
        'Actual_Churn': y_test,
        'Churn_Probability': y_pred_proba
    })
    
    # Risk segmentation
    df_analysis['Risk_Segment'] = pd.cut(
        df_analysis['Churn_Probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print(f"\n{'='*60}")
    print(f"BUSINESS ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    for segment in ['Low Risk', 'Medium Risk', 'High Risk']:
        segment_data = df_analysis[df_analysis['Risk_Segment'] == segment]
        if len(segment_data) > 0:
            churn_rate = segment_data['Actual_Churn'].mean()
            customers = len(segment_data)
            print(f"\n{segment}:")
            print(f"  Customers: {customers} ({customers/len(df_analysis)*100:.1f}%)")
            print(f"  Actual Churn Rate: {churn_rate*100:.1f}%")
            print(f"  Avg Churn Probability: {segment_data['Churn_Probability'].mean():.3f}")
    
    return df_analysis

# ==================== 8. VISUALIZATIONS ====================
def plot_class_distribution(df):
    """Plot class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before SMOTE
    ax = axes[0]
    df['Churn'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Class Distribution (Original)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Churn Status')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No Churn (0)', 'Churn (1)'], rotation=0)
    for i, v in enumerate(df['Churn'].value_counts().values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Percentage
    ax = axes[1]
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie(
        churn_pct.values, labels=['No Churn', 'Churn'],
        autopct='%1.1f%%', colors=colors, startangle=90
    )
    ax.set_title('Class Distribution (%)', fontsize=12, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distributions(df_processed):
    """Plot distributions of numeric features"""
    # Get only numeric columns (excluding Churn)
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_features:
        numeric_features.remove('Churn')
    
    # Limit to first 6 features for visualization
    numeric_features = numeric_features[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numeric_features):
        ax = axes[idx]
        df_processed[feature].hist(ax=ax, bins=30, color='#3498db', edgecolor='black')
        ax.set_title(f'Distribution of {feature}', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('02_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df_processed):
    """Plot correlation heatmap - only top features correlated with Churn"""
    # Get correlation with Churn
    correlation_with_churn = df_processed.corr()['Churn'].sort_values(ascending=False)
    
    # Select only top 10 features (5 positive + 5 negative correlation)
    top_positive = correlation_with_churn.head(6).index.tolist()
    top_negative = correlation_with_churn.tail(5).index.tolist()
    top_features = top_positive + top_negative
    
    # Create correlation matrix for only these features
    corr_matrix = df_processed[top_features].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap with clean design
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, square=True, ax=ax,
        cbar_kws={'label': 'Correlation'},
        linewidths=1, linecolor='white',
        vmin=-1, vmax=1, annot_kws={'size': 10, 'weight': 'bold'}
    )
    ax.set_title('Feature Correlation Matrix - Key Features Only', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, len(results), figsize=(14, 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = confusion_matrix(y_test, metrics['y_pred'])
        
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            cbar=False, square=True,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        
        # Add metrics text
        tn, fp, fn, tp = cm.ravel()
        metrics_text = f"TP:{tp} FP:{fp}\nFN:{fn} TN:{tn}"
        ax.text(1, -0.25, metrics_text, transform=ax.transAxes,
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('04_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        roc_auc = metrics['AUC-ROC']
        
        ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(results, y_test):
    """Plot Precision-Recall curves"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name, metrics in results.items():
        precision, recall, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
        f1 = metrics['F1-Score']
        
        ax.plot(recall, precision, linewidth=2.5, label=f'{model_name} (F1 = {f1:.3f})')
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('06_precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_tuning(y_test, y_pred_proba, model_name):
    """Plot threshold tuning analysis"""
    optimal_f1, f1_scores, thresholds = find_optimal_threshold(y_test, y_pred_proba, metric='f1')
    optimal_precision, precision_scores, _ = find_optimal_threshold(y_test, y_pred_proba, metric='precision')
    optimal_recall, recall_scores, _ = find_optimal_threshold(y_test, y_pred_proba, metric='recall')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, f1_scores, 'b-', linewidth=2.5, label='F1-Score', marker='o', markersize=4)
    ax.plot(thresholds, precision_scores, 'g--', linewidth=2.5, label='Precision', marker='s', markersize=4)
    ax.plot(thresholds, recall_scores, 'r-.', linewidth=2.5, label='Recall', marker='^', markersize=4)
    
    ax.axvline(optimal_f1, color='b', linestyle=':', alpha=0.7, label=f'Optimal F1 Threshold: {optimal_f1:.2f}')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default Threshold: 0.50')
    
    ax.set_xlabel('Classification Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Threshold Tuning Analysis - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'07_threshold_tuning_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_f1

def plot_feature_importance(models, feature_names):
    """Plot feature importance for Random Forest"""
    if 'Random Forest' not in models:
        return
    
    model = models['Random Forest']
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]  # Top 10 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.set_title('Top 10 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(importance[indices]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('08_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results):
    """Compare all models with key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC-ROC']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = [results[model][metric] for model in results.keys()]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(results.keys(), values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('09_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_segments(df_analysis, model_name):
    """Plot risk segmentation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk segment distribution
    ax = axes[0]
    segment_counts = df_analysis['Risk_Segment'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    segment_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(f'Customer Distribution by Risk Segment - {model_name}', fontweight='bold')
    ax.set_xlabel('Risk Segment')
    ax.set_ylabel('Number of Customers')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for i, v in enumerate(segment_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # Churn rate by segment
    ax = axes[1]
    churn_by_segment = df_analysis.groupby('Risk_Segment')['Actual_Churn'].agg(['sum', 'count'])
    churn_rate = (churn_by_segment['sum'] / churn_by_segment['count'] * 100).reindex(
        ['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    bars = ax.bar(churn_rate.index, churn_rate.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(f'Actual Churn Rate by Risk Segment - {model_name}', fontweight='bold')
    ax.set_xlabel('Risk Segment')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'10_risk_segments_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # 1. Load dataset
    dataset_path = "Telco-Customer-Churn.csv"
    df = load_churn_dataset(dataset_path)
    
    # 2. Preprocess data
    df_processed = preprocess_data(df)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    feature_names = X.columns.tolist()
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train models
    models = train_models(X_train_scaled, y_train_balanced, X_test_scaled, y_test)
    
    # 7. Evaluate models
    results = evaluate_models(models, X_test_scaled, y_test)
    print("\nMODEL EVALUATION METRICS")
    print("="*70)
    
    # Display only metrics (exclude prediction arrays)
    metrics_only = {}
    for model_name, result in results.items():
        metrics_only[model_name] = {k: v for k, v in result.items() if k not in ['y_pred', 'y_pred_proba']}
    
    metrics_df = pd.DataFrame(metrics_only).T
    print(metrics_df.round(4).to_string())
    print("="*70)
    
    # 8. Generate visualizations
    plot_class_distribution(df)
    plot_feature_distributions(df_processed)
    plot_correlation_heatmap(df_processed)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_feature_importance(models, feature_names)
    plot_model_comparison(results)
    
    # Threshold tuning
    print("\nTHRESHOLD TUNING RESULTS")
    print("="*70)
    for model_name, metrics in results.items():
        optimal_threshold = plot_threshold_tuning(y_test, metrics['y_pred_proba'], model_name)
        print(f"{model_name}: Optimal F1 Threshold = {optimal_threshold:.2f}")
    print("="*70)
    
    # Risk segmentation analysis
    print("\nRISK SEGMENTATION ANALYSIS")
    print("="*70)
    for model_name, metrics in results.items():
        df_analysis = business_analysis(y_test, metrics['y_pred_proba'], model_name)
        plot_risk_segments(df_analysis, model_name)
    print("="*70)
    
    # Save results
    metrics_df.round(4).to_csv('model_evaluation_metrics.csv')
    df.to_csv('customer_churn_dataset.csv', index=False)
