
 Car Price Prediction Model

This project focuses on building a machine learning model to predict car prices based on various attributes like car model, engine size, condition, fuel type, and other characteristics. Multiple models such as Support Vector Regressor (SVR), Random Forest Regressor, and Decision Tree Regressor have been implemented and evaluated.

 Project Structure
1. Data Preprocessing
    Categorical and Numerical Separation: The dataset is split into categorical and numerical columns to facilitate preprocessing.
    Feature Engineering: The categorical variables are converted into dummy variables for use in machine learning models.
    Data Cleaning: Missing values are handled by replacing them with appropriate statistics (mean, median, or mode).
   
2. Modeling
    Three machine learning models are implemented to predict the car prices:
      Support Vector Regressor (SVR)
      Random Forest Regressor
      Decision Tree Regressor
   
   The models are trained using the cleaned data, and their performance is evaluated using the following metrics:
    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE)

3. Data
   The data used for this project includes information on various car models, their attributes (such as fuel type, safety rating, engine size), and their corresponding price. It has both categorical and numerical features that are used to predict car prices.

 Instructions

 Requirements

1. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikitlearn seaborn matplotlib
   ```

 Steps

1. Load the Data:
    Make sure the cleaned DataFrame `finaldf` contains the required data for training. This should include the feature columns and the target column (`Price`).
   
2. Preprocess Data:
    Categorical features are transformed into dummy variables using `pd.get_dummies()`.
    The dataset is split into training and testing sets using `train_test_split()`.

3. Train Models:
    Three regression models (SVR, Random Forest, Decision Tree) are trained on the dataset.
    The models are trained on 80% of the data and evaluated on 20% of the data.

4. Evaluate Models:
    The performance of each model is assessed based on the MAE, MSE, and RMSE metrics.
   
5. Check Price Range:
    The minimum and maximum prices in the dataset are printed to validate the data.

 Example Code Snippet:

```python
 Split the data into predictors and target variable
X = finaldf.drop('Price', axis=1)
y = finaldf['Price']

 Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

 Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 Train and evaluate models
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

 Get predictions and evaluate model performance
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

 Print evaluation metrics for each model
print(f"Support Vector Machine MAE: {svm_mae}")
print(f"Random Forest RMSE: {rf_rmse}")
```

 Performance Metrics:
1. Decision Tree
The Decision Tree model is our baseline model. Decision trees are commonly used for regression analysis because they are simple to understand and interpret. The performance metrics for the Decision Tree model are:
•	MAE: 2392.3247
•	MSE: 11448996.7983
•	RMSE: 3383.6366
2. Support Vector Machine (SVM)
Support Vector Machines are supervised learning models that can be used for regression by minimizing the error between predicted and actual values. The performance metrics for the SVM model are:
•	MAE: 6894.5803
•	MSE: 63247833.0477
•	RMSE: 7952.8506
Compared to the Decision Tree, the SVM model has higher errors across all metrics, indicating that it may not be as effective for this particular dataset.
3. Random Forest
Random Forest is an ensemble learning method that creates multiple decision trees to make a forest. It helps in improving the accuracy of the model. The performance metrics for the Random Forest model are:
•	MAE: 1918.3656
•	MSE: 7407168.8633
•	RMSE: 2721.6114
The Random Forest model outperforms the Decision Tree in all metrics, indicating a better fit for the data.



 Price Range:
The minimum and maximum price values in the dataset are printed to provide context for the model’s output.



 Next Steps
 Experiment with other machine learning algorithms such as Gradient Boosting or Neural Networks.
 Tune hyperparameters of the existing models using crossvalidation to potentially improve performance.
 Enhance data cleaning techniques or add feature engineering steps like binning numerical features.

 Contact:
For any issues or suggestions, feel free to reach out.
