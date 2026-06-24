# Customer-Churn-Prediction
 1. Problem Statement

Customer retention is one of the major challenges faced by companies, especially in industries such as telecom, banking, e-commerce, and subscription-based services. Losing existing customers (customer churn) directly affects company revenue because acquiring a new customer is more expensive than retaining an existing one.

The objective of this project is to build a Machine Learning based system that can predict whether a customer is likely to leave a service in the future based on their historical behavior, usage patterns, and account information.

The system analyzes customer-related factors such as:

Customer tenure
Contract type
Monthly charges
Service usage
Payment method
Customer support interactions
Account information

and predicts:

Churn → Customer is likely to leave

No Churn → Customer is likely to continue

The main goal is not only to predict churn accurately but also to identify why a customer is leaving so that companies can take preventive retention actions.

2. Domain Understanding
Domain: Machine Learning + Customer Analytics

Customer churn prediction is a binary classification problem in Machine Learning.

The input data contains customer attributes, and the model learns patterns from previous customer behavior.

Example:

A customer with:

Low tenure
High monthly charges
Multiple complaints
Month-to-month contract

may have a higher probability of leaving.

Companies can use these predictions to:

Provide personalized offers
Improve customer support
Reduce customer loss
Increase customer lifetime value

3.Existing base paper work:
Base Paper:

Customer Churn Prediction Using Machine Learning with Explainable AI

The base paper proposed a machine learning framework for telecom customer churn prediction.

The authors performed:

Data preprocessing
Feature selection
Model training
Performance evaluation
Model explainability analysis

The paper compared multiple machine learning algorithms:

Logistic Regression
Decision Tree
Random Forest
XGBoost

Among all models, XGBoost achieved the best performance.

Results:

Model	Accuracy	F1 Score
Logistic Regression	87.8%	0.153
Decision Tree	92.2%	0.614
Random Forest	95.4%	0.796
XGBoost	96.2%	0.836

The authors selected XGBoost as the final prediction model because it provided better accuracy and classification performance.

The paper also implemented SHAP (SHapley Additive exPlanations) to explain the prediction results and identify which customer features influence churn decisions
