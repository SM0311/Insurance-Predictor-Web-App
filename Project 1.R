# Load necessary libraries
library(caret)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(stargazer)
library(leaps)

# Load the dataset
insurance <- read.csv("D:/Conestoga/Predictive Analytics/Semester 1/Multivariate Statiscts/Project/insurance Test.csv")

# Display the first few rows and summary of the dataset
head(insurance)
summary(insurance)

# Train and test split (70% train, 30% test)
set.seed(123)  # For reproducibility
train_index <- createDataPartition(insurance$charges, p = 0.7, list = FALSE)
train_data <- insurance[train_index, ]
test_data <- insurance[-train_index, ]

# Visualize relationships between variables
ggplot(insurance, aes(x = age, y = charges)) + geom_point() + labs(x = "Age", y = "Charges") + ggtitle("Scatter plot of Age vs Charges")
ggplot(insurance, aes(x = smoker, y = charges)) + geom_point() + labs(x = "Smoker", y = "Charges") + ggtitle("Scatter plot of Smoker vs Charges")
ggplot(insurance, aes(x = children, y = charges)) + geom_point() + labs(x = "Children", y = "Charges") + ggtitle("Scatter plot of Children vs Charges")
ggplot(insurance, aes(x = region, y = charges)) + geom_point() + labs(x = "Region", y = "Charges") + ggtitle("Scatter plot of Region vs Charges")
ggplot(insurance, aes(x = bmi, y = charges)) + geom_point() + labs(x = "BMI", y = "Charges") + ggtitle("Scatter plot of BMI vs Charges")
ggplot(insurance, aes(x = bmi, y = charges)) + geom_point() + geom_smooth(method = "lm", se = FALSE, color = "blue") + labs(x = "BMI", y = "Charges") + ggtitle("Scatter plot of BMI vs Charges with Regression Line")

# Check the structure of the data and convert appropriate columns to factors
str(insurance)
insurance$sex <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)

# Build the linear model and summarize it
model <- lm(charges ~ age + sex + bmi + children + smoker + region, data = insurance)
summary(model)

# Best subset selection
best_subsetmodel <- regsubsets(charges ~ age + sex + bmi + children + smoker + region, data = insurance)
plot(best_subsetmodel, scale = "adjr2")
summary(best_subsetmodel)

# Train the model with cross-validation
Train_Model2 <- train(
  form = charges ~ age + sex + bmi + children + smoker + region,
  data = insurance,
  method = "lm",
  trControl = trainControl(method = "cv", number = 6)
)
summary(Train_Model2)

# Predict on the test data
prediction <- predict(Train_Model2, test_data)
summary(prediction)

# Actual charges from the test data
actual <- test_data$charges

predicted <- prediction

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(actual - predicted))
print(paste("Mean Absolute Error (MAE):", mae))

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((actual - predicted)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Calculate R-squared (R²)
rss <- sum((predicted - actual) ^ 2)  # Residual Sum of Squares
tss <- sum((actual - mean(actual)) ^ 2)  # Total Sum of Squares
r_squared <- 1 - (rss/tss)
print(paste("R-squared (R²):", r_squared))

## So the model's accuracy is almost 73%.


