# FicZon Sales Effectiveness - Lead Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 Project Overview

**Team ID:** PTID-CDS-APR-25-2603  
**Project ID:** PRCL-0019

FicZon Inc, a leading IT solutions provider, faces declining sales performance due to market maturity and increased competition. This project implements a Machine Learning-driven solution to automate lead categorization, transforming manual processes into intelligent, data-driven decision-making systems.

### 🎯 Business Objective
Develop a predictive model to categorize leads as **High Potential** or **Low Potential**, enabling:
- Enhanced sales effectiveness through improved lead qualification
- Optimized resource allocation and prioritization
- Reduced dependency on manual categorization processes
- Increased conversion rates and revenue growth

## 🚀 Key Features

- **Automated Lead Classification**: Binary classification (High/Low Potential)
- **Multi-Algorithm Comparison**: 7 different ML algorithms evaluated
- **Data Preprocessing Pipeline**: Comprehensive feature engineering and data cleaning
- **Model Optimization**: Hyperparameter tuning and ensemble methods
- **Production-Ready Model**: Serialized model for deployment

## 📊 Dataset Information

- **Total Records**: 7,422 observations
- **Features**: 9 attributes (3 unique identifiers)
- **Target Variable**: Lead Status (11 categories → 2 binary classes)
- **Data Type**: Categorical and mixed-type features

### Feature Description
| Feature | Description | Type |
|---------|-------------|------|
| `Created` | Activity timestamp | Unique |
| `Product_ID` | Product identifier | Categorical |
| `Source` | Customer engagement channel | Categorical |
| `Mobile` | Customer phone number | Unique |
| `Email` | Customer email address | Unique |
| `Sales_Agent` | Assigned sales representative | Categorical |
| `Location` | Geographic location | Categorical |
| `Delivery_Mode` | Service delivery method | Categorical |
| `Status` | Lead classification (Target) | Binary |

## 🔧 Technical Architecture

### Data Processing Pipeline
```
Raw Data → Missing Value Handling → Label Compression → 
Feature Engineering → Categorical Encoding → Model Training
```

### Model Comparison Results
| Algorithm | Training Accuracy | Testing Accuracy | Tuned Accuracy |
|-----------|------------------|------------------|----------------|
| Logistic Regression | 71.08% | 69.09% | - |
| K-Nearest Neighbors | 76.64% | 67.21% | 69.16% |
| Decision Tree | 83.91% | 68.62% | 68.42% |
| Random Forest | 83.91% | 68.96% | **72.26%** |
| Gradient Boosting | 74.13% | 70.77% | - |
| XGBoost | 78.37% | 70.44% | 72.32% |
| Neural Network | 74.84% | 69.76% | - |

**🏆 Best Performing Model**: Gradient Boosting Classifier (70.77% test accuracy)

## 📁 Project Structure

```
FicZon-Sales-Effectiveness/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
│       └── preprocess.csv
├── notebooks/
│   └── FICZON-SALES-EFFECTIVENESS.ipynb
├── models/
│   └── gbm_classifier_model.pkl
├── reports/
│   └── SWEETVIZ_REPORT.html
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── evaluation.py
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- MySQL Server
- Jupyter Notebook

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ficzon-sales-effectiveness.git
cd ficzon-sales-effectiveness
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Database Configuration**
```python
# Update database credentials in the notebook
connection = mysql.connector.connect(
    host='your-host',
    user='your-username', 
    password='your-password',
    database='project_sales'
)
```

## 📈 Usage

### 1. Data Extraction & Preprocessing
```python
# Load and preprocess data
import pandas as pd
import pickle

# Load preprocessed data
data = pd.read_csv('data/processed/preprocess.csv')

# Load trained model
with open('models/gbm_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)
```

### 2. Making Predictions
```python
# Predict lead category
prediction = model.predict(new_lead_data)
probability = model.predict_proba(new_lead_data)

# Interpret results
lead_category = "High Potential" if prediction[0] == 1 else "Low Potential"
confidence = max(probability[0]) * 100
```

### 3. Model Evaluation
```python
from sklearn.metrics import classification_report, accuracy_score

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
```

## 📊 Key Insights

### Data Analysis Findings
- **21%** of customers prefer Call source with Mode5 delivery
- **28%** of customers originate from Bangalore
- **20%** of leads are classified as "Junk Leads"
- **Product ID 18** represents 23% of all products sold

### Model Performance Insights
- **Ensemble methods** (Random Forest, Gradient Boosting, XGBoost) show superior performance
- **Hyperparameter tuning** improved accuracy by 2-4% across models
- **Class imbalance** affects minority class precision but maintains good recall
- **Gradient Boosting** provides the best balance of accuracy and generalization

## 🔮 Future Enhancements

- [ ] **Real-time Prediction API**: Deploy model as REST API service
- [ ] **Feature Importance Analysis**: Implement SHAP values for explainability
- [ ] **Advanced Ensemble Methods**: Explore stacking and voting classifiers
- [ ] **Automated Retraining**: Implement MLOps pipeline for model updates
- [ ] **A/B Testing Framework**: Compare model performance against manual processes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Data Science Team**  
Team ID: PTID-CDS-APR-25-2603

## 📞 Contact

For questions or collaboration opportunities:
- **Email**: [team@ficzon.com](mailto:team@ficzon.com)
- **Project Lead**: [Your Name](mailto:your.email@company.com)

## 🙏 Acknowledgments

- FicZon Inc for providing the dataset and business context
- Scikit-learn community for excellent ML libraries
- Sweetviz for automated EDA reporting

---

**⭐ If this project helped you, please give it a star!**