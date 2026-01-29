# Customer Churn Prediction Using PySpark

Machine learning pipeline using Apache Spark to predict telecommunications customer churn with 83.7% AUC-ROC accuracy.

## Overview

Analyzed 7,043 customer records from a telecommunications company using distributed machine learning. Compared Logistic Regression and Random Forest algorithms to identify at-risk customers before they churn, achieving strong predictive performance suitable for enterprise deployment.

## Key Results

- **Accuracy:** 79.9% (Logistic Regression), 79.1% (Random Forest)
- **AUC-ROC:** 83.8% (Logistic Regression), 83.7% (Random Forest)
- **Business Impact:** Identified potential $1.67M in annual revenue at risk
- **Key Predictors:** Contract type (16.3%), tenure (14.5%), service charges (9.9%)

## Tech Stack

- **Python** - Core programming language
- **Apache Spark (PySpark)** - Distributed computing framework  
- **Spark MLlib** - Machine learning library
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization

## Project Structure
```
customer-churn-prediction/
├── Churn_Prediction_Final_Project.ipynb    # Complete analysis notebook
├── Churn_Prediction_Research_paper.pdf     # Academic research paper
├── Churn_Prediction_Presentation.pptx      # Project presentation
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

## Key Findings

### Model Performance
Both Logistic Regression and Random Forest achieved strong results:
- Logistic Regression slightly outperformed Random Forest (79.9% vs 79.1% accuracy)
- Linear relationships were sufficient for this dataset
- Models demonstrate production-ready performance

### Business Insights
1. **Contract Type Impact:** Month-to-month contracts show 42.7% churn vs 2.8% for two-year contracts
2. **Tenure Effect:** Churn drops from 47.4% (0-12 months) to 9.5% (48+ months)
3. **Service Bundling:** Customers with multiple services show lower churn rates
4. **Payment Method:** Electronic check users more likely to churn than automatic payment users

### Feature Engineering
Created meaningful business features:
- **Tenure Groups:** Segmented customer lifecycle stages
- **Charges Per Service:** Identified value perception issues
- **Service Count:** Measured customer engagement level

## Technical Implementation

### Data Processing
- Handled 7,043 customer records with 21 variables
- Addressed missing values in billing data
- Applied StandardScaler for feature normalization
- Used StringIndexer for categorical encoding

### Model Pipeline
```
Data Ingestion → Feature Engineering → Vector Assembly → 
Feature Scaling → Model Training → Evaluation
```

### Algorithms Compared
1. **Logistic Regression**
   - L2 regularization (λ = 0.01)
   - Highly interpretable coefficients
   - Fast training and prediction

2. **Random Forest**
   - 50 trees with max depth of 10
   - Built-in feature importance
   - Captures non-linear relationships

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Install Apache Spark
# Download from: https://spark.apache.org/downloads.html
```

## Usage
```bash
# Open Jupyter notebook
jupyter notebook Churn_Prediction_Final_Project.ipynb

# Or run with Spark
spark-submit --master local[*] churn_prediction_script.py
```

## Dataset

- **Source:** Telco Customer Churn (Kaggle)
- **Size:** 7,043 customers
- **Features:** 21 variables (demographics, billing, services)
- **Target:** Binary churn indicator (26.5% churn rate)

## Results Visualization

The project includes comprehensive visualizations:
- Churn rate by contract type
- Customer tenure vs churn probability
- ROC curves comparing models
- Feature importance rankings
- Confusion matrices
- Business impact analysis

## Scalability

While this analysis used 7,043 records, the PySpark framework scales to:
- Millions of customer records
- Distributed computing clusters
- Real-time streaming data
- Enterprise production environments

## Future Enhancements

- Deep learning integration for complex pattern recognition
- Real-time churn scoring with Spark Streaming
- A/B testing framework for retention strategies
- Cost-sensitive learning based on customer lifetime value
- Network analysis for referral pattern effects

## Business Applications

This model enables telecommunications companies to:
1. **Proactive Retention:** Identify at-risk customers before they churn
2. **Targeted Marketing:** Focus retention efforts on high-value customers
3. **Resource Optimization:** Allocate budgets to customers most likely to respond
4. **Revenue Protection:** Prevent $1.67M+ annual recurring revenue loss

## Author

**Luis Miranda** - Computer Engineering Graduate  
Florida International University (August 2025, Cum Laude)

- Email: Luismiranda156@gmail.com
- LinkedIn: [linkedin.com/in/luis-miranda-bb68a329a](https://linkedin.com/in/luis-miranda-bb68a329a)

## Acknowledgments

- Florida International University - Department of Engineering
- Kaggle - Telco Customer Churn Dataset

## License

This project is available for educational and research purposes.
