# Airbnb Listings Price Predictions

## Overview and Forecasting Problem

Airbnb is an online marketplace that connects people who want to rent out their homes with people who are looking for accommodations in specific locations.
<br><br>
The goal of this problem is to develop a model that can accurately predict the listed prices of Airbnb properties in Melbourne based on various characteristics. The dataset provided contains information about different Airbnb properties, including features such as the property type, number of bathrooms, number of bedrooms and beds, minimum and maximum nights of stay, and availability.
<br><br>
By leveraging the characteristics in the dataset and building a model, the objective is to learn the relationship between these features and the listed prices. The model will be trained to predict the price of a property based on its specific characteristics.

The steps involved in developing the model are as follows:

**Exploratory Data Analysis**
- The dataset will be analysed to understand the distribution of features and identify potential relationships between the features and the listed prices.

**Data Preprocessing**
- Missing values in the dataset will be identified and either removed or imputed. Categorical variables will be encoded using techniques such as one-hot encoding, and numerical features may be scaled if required.

**Feature Selection and Engineering**
- Relevant features that have a significant impact on price prediction will be selected. Additionally, new features may be created through feature engineering techniques, such as extracting information from existing features or combining multiple features.

**Model Selection**
- Regression models such as Linear Regression and GradientBoostingRegressor will be considered for training. These models will be trained on the preprocessed dataset and their performance will be evaluated using root mean squared error (RMSE) and cross-validation scores.

**Model Tuning**
- Hyperparameter tuning will be performed using techniques like GridSearchCV to find the best set of hyperparameters for each model in order to optimize the models' performance.

**Model Evaluation**
- The models will be evaluated to assess their performance in predicting the listed prices of Airbnb properties.

By developing a robust model, we will be able to accurately predict the listed prices of Airbnb properties in Melbourne. This model can provide valuable insights for both hosts and guests, enabling them to make more informed decisions about pricing and choosing suitable accommodation options.

### Evaluation Criteria
The criteria to assess prediction performance is RMSE. This performance metric measures the average distance between predictions obtained by a model and actual target values. Thus, the lower the distance (and the smaller RMSE), the better the prediction quality. It also has the advantage of being in the same unit as the predicted variable which makes it easy to interpret.

## Summary and Results

We attempted to train several regression models, including RandomForestRegressor, GradientBoostRegressor, LASSO, and elastic net, with hyperparameter tuning using GridSearchCV. The top 10 features, as determined by their importance in the regression model, were selected for training.

RandomForestRegressor performed the best based on the Root Mean Squared Error (RMSE) and cross-validation (CV) scores. However evaluating the models on unseen data it was observed that they did not generalise well, even though the RMSE and CV scores appeared reasonable. This discrepancy indicated that the models were overfitting the training set.

To address this issue, we removed price outliers from the dataset and retrained the models. However, even after this preprocessing step, the models performed poorly on the test data, suggesting that some of the outliers were present in the test set. This was evident from the very low RMSE score, indicating a high level of overfitting.

We then experimented with different combinations of features to train the models. Although there was some improvement in the RMSE and CV scores, the models still failed to significantly improve the test scores. To refine the feature selection process, we relied on the rankings provided by XGBoost and eliminated the features that had little impact on the price prediction. Additionally, we made some hyperparameter changes, such as reducing the number of estimators and increasing the depth of the RandomForestRegressor.

Through these adjustments, we were able to achieve a test score of 3409.51 using RandomForestRegressor. Despite efforts to train and tune various regression models, we were not able to generalise the models to unseen data to achieving high accuracy in price prediction. Further refinement of model and feature selection and hyperparameters may be necessary to enhance the model's performance.
