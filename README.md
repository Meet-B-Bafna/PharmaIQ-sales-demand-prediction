#PharmaIQ-sales-demand-prediction
Pharmacy sales analysis and demand prediction using Python, Pandas, and Machine Learning

#Overview
PharmaIQ is a simple data science and machine learning project that analyzes pharmacy sales data and predicts future demand for medicines. The goal is to help identify sales trends and support better business decisions.

#Features
1.Data cleaning and preprocessing using Python
2.Analysis of top-selling and low-performing medicines
3.Revenue calculation and business insights
4.Visualization of sales trends using charts
5.Machine learning model to predict medicine demand

#Technologies Used
1.Python
2.Pandas
3.Matplotlib
4.Scikit-learn

#Dataset
The dataset contains pharmacy sales records with the following columns:
1.Date
2.Medicine Name
3.Category
4.Quantity Sold
5.Price
6.Store ID
7.Discount
8.Profit

#Project Workflow
1. Data Loading and Cleaning
2. Feature Engineering (Month, Day, Time Index, Previous Sales)
3. Exploratory Data Analysis
4. Data Visualization
5. Model Training using Random Forest
6. Prediction and Evaluation
  
# Machine Learning
A Random Forest Regressor model is used to predict the quantity of medicines sold based on:
1.Price
2.Category
3.Discount
4.Time-based features
5.Previous sales

# Evaluation
1.Model performance is evaluated using Mean Absolute Error (MAE)
2.The model shows moderate accuracy due to limited feature relationships in the dataset


#Conclusion
This project demonstrates the end-to-end workflow of a basic machine learning project, including data analysis, visualization, and prediction. It is designed as a beginner-friendly project for learning and showcasing skills.
