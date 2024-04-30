library("readxl")
library(lmtest)
library(sandwich)
library(ggplot2)
library(boot)
library(leaps)
library(caret)
library(dplyr)
library(car) 
library(glmnet)
library(step_dummy)
library("np.exp")
library(Metrics)

# Read the data from the Excel file
file_path <- "C:\\Users\\46704\\Desktop\\Data utbildning\\R Programmering\\Kunskapskontroll 6\\BMW Data.xlsx"
cars1 <- read_excel(file_path)# View the structure of the data to understand its columns

str(cars1)

#--------------------------------------------------------------

cars1$Modell <- factor(cars1$Modell)
cars1$Motorstorlek <- factor(cars1$Motorstorlek)

#--------------------------------------------------------------

# Create the LM model 1
lm1_model <- lm(Pris ~ ., data = cars1)

# Result
summary(lm1_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm1_model)

vif(lm1_model)

#--------------------------------------------------------------

cars1$log_Pris <- log(cars1$Pris)

# Create the LM model 2
lm2_model <- lm(log_Pris ~ . - Pris, data = cars1)

# Result
summary(lm2_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm2_model)

vif(lm2_model)

#--------------------------------------------------------------

# Calculate studentized residuals
student_resid <- rstudent(lm2_model)

# Plot studentized residuals
plot(student_resid, type = "p", main = "Studentized Residuals Plot", xlab = "Observation", ylab = "Studentized Residual")

# Fins outliers
outliers_student <- which(abs(student_resid) > 3)  # Определяем выбросы как стандартизированные остатки, превышающие 2

# Outputting outliers indices, if any
if (length(outliers_student) > 0) {
  cat("Outliers found at observations:", outliers_student, "\n")
} else {
  cat("No outliers found.\n")
}

# Remove outliers from cars1
cars_no_outliers <- cars1[-outliers_student, ]

# Check the structure of the new dataset without outliers
str(cars_no_outliers)

#--------------------------------------------------------------

# Create the LM model 3
lm3_model <- lm(log_Pris ~ . - Pris, data = cars_no_outliers)

# Result
summary(lm3_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm3_model)

vif(lm3_model)


# Calculate studentized residuals
student_resid <- rstudent(lm3_model)

# Plot studentized residuals
plot(student_resid, type = "p", main = "Studentized Residuals Plot", 
     xlab = "Observation", ylab = "Studentized Residual")

#--------------------------------------------------------------

# Fit the linear regression model using best subset selection
best_subset <- regsubsets(log_Pris ~ . - Pris, data = cars_no_outliers, nvmax = 45, method = "exhaustive")

# Summary of the best subset selection
summary_best_subset <- summary(best_subset)

# Plot Cp values
plot(summary_best_subset$cp)

# Find the index with the minimum Cp value
min_cp_index <- which.min(summary_best_subset$cp)
min_cp_value <- min(summary_best_subset$cp)
cat("Minimum Cp Index:", min_cp_index, "\n")
cat("Minimum Cp Value:", min_cp_value, "\n")

# Extract the coefficients at the minimum Cp index
min_cp_model <- coef(best_subset, id = min_cp_index)

# Get the names of predictors with non-zero coefficients
selected_features <- names(min_cp_model[min_cp_model != 0])

# Print the selected features
print("Selected features:")
print(selected_features)


# Update the Län variable to create dummy regions
cars_no_outliers$Län1 <- ifelse(cars_no_outliers$Län == "Gävleborg", "Gävleborg",
                     ifelse(cars_no_outliers$Län == "Halland", "Halland",
                            ifelse(cars_no_outliers$Län== "Västernorrland" , "Västernorrland",
                                   ifelse(cars_no_outliers$Län == "Västerbotten", "Västerbotten",
                                          ifelse(cars_no_outliers$Län == "Skaraborg", "Skaraborg",
                                                 ifelse(cars_no_outliers$Län == "Värmland", "Värmland", "1Övriga")
                                          )
                                   )
                            )
                     )
)

str(cars_no_outliers)


# Create the LM model 4
lm4_model <- lm(log_Pris ~ . - Pris - Län,  data = cars_no_outliers)

# Result
summary(lm4_model)
 
# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm4_model)

vif(lm4_model)

#--------------------------------------------------------------

# Define feature matrix X and response vector y
X <- model.matrix(lm2_model, data = cars_no_outliers) # Adjust this to your actual feature matrix
y <- cars_no_outliers$log_Pris  # Adjust this to your actual response vector

# Create a cross-validated Lasso model
lasso_model.cv <- cv.glmnet(X, y, alpha = 1)

# Get the coefficients of the cross-validated Lasso model
lasso_coef <- coef(lasso_model.cv)

# Extract the non-zero coefficients (selected variables)
selected_variables1 <- rownames(lasso_coef)[apply(lasso_coef, 1, any)]

# Print the selected variables
print(selected_variables1)


# Update the Län variable to create dummy regions
cars_no_outliers$Län2 <- ifelse(cars_no_outliers$Län == "Skaraborg", "Skaraborg",
                                ifelse(cars_no_outliers$Län == "Halland", "Halland",
                                       ifelse(cars_no_outliers$Län== "Värmland" , "Värmland",
                                              ifelse(cars_no_outliers$Län == "Västerbotten", "Västerbotten", "1Övriga")
                                       )
                                )
)


# Create the LM model 5
lm5_model <- lm(log_Pris ~ . - Pris - Län - Län1,  data = cars_no_outliers)

# Result
summary(lm5_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm5_model)

vif(lm5_model)

#--------------------------------------------------------------

# Determining threshold
threshold <- 0.075

# Calculating influence
leverages <- hatvalues(lm5_model)

# Displaying observations with influence above the threshold
high_leverage <- which(leverages > threshold)
print(high_leverage)

# Delete high leverage points from the dataset
cars_no_high_leverage <- cars_no_outliers[-high_leverage, ]

# Check the structure of the new dataset without high leverage points
str(cars_no_high_leverage)


# Create the LM model 6
lm6_model <- lm(log_Pris ~ . - Pris - Län - Län1,  data = cars_no_high_leverage)

# Result
summary(lm6_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm6_model)

vif(lm6_model)

#--------------------------------------------------------------

# Set seed for reproducibility
set.seed(123)

# Splitting indices for train-validation-test split
train_index <- sample(nrow(cars_no_outliers), 0.6 * nrow(cars_no_outliers))  # 60% for training
validation_test_index <- setdiff(1:nrow(cars_no_outliers), train_index)

# Further split the remaining data into validation and test sets
validation_index <- sample(validation_test_index, 0.5 * length(validation_test_index))  # 20% for validation
test_index <- setdiff(validation_test_index, validation_index)

# Create train-validation-test datasets
train_data <- cars_no_outliers[train_index, ]
validation_data <- cars_no_outliers[validation_index, ]
test_data <- cars_no_outliers[test_index, ]

#--------------------------------------------------------------

# Create the LM model 3
lm3_model <- lm(log_Pris ~ . - Pris - Län1 - Län2, data = train_data)

# Выведите результаты модели
summary(lm3_model)

# Постройте диагностические графики
par(mfrow = c(2, 2))
plot(lm3_model)

vif(lm3_model)

#--------------------------------------------------------------
# Best subset
# Create the LM model 4
lm4_model <- lm(log_Pris ~ . - Pris - Län - Län2,  data = train_data)

# Выведите результаты модели
summary(lm4_model)

# Постройте диагностические графики
par(mfrow = c(2, 2))
plot(lm4_model)

vif(lm4_model)

#--------------------------------------------------------------
#Lasso model
# Create the LM model 5
lm5_model <- lm(log_Pris ~ . - Pris - Län - Län1,  data = train_data)

# Выведите результаты модели
summary(lm5_model)

# Постройте диагностические графики
par(mfrow = c(2, 2))
plot(lm5_model)

vif(lm5_model)

#--------------------------------------------------------------

# Calculate RMSE on Validation set
pre_val_lm3_model <- predict(lm3_model, newdata = validation_data)
pre_val_lm4_model <- predict(lm4_model, newdata = validation_data)
pre_val_lm5_model <- predict(lm5_model, newdata = validation_data)   

val_rmse_lm3_model <- rmse(validation_data$log_Pris, pre_val_lm3_model)  
val_rmse_lm4_model<- rmse(validation_data$log_Pris, pre_val_lm4_model)
val_rmse_lm5_model <- rmse(validation_data$log_Pris, pre_val_lm5_model)


results <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3"),
  RMSE_val_data = c(val_rmse_lm3_model, val_rmse_lm4_model, val_rmse_lm5_model),
  Adj_R_squared = c(summary(lm3_model)$adj.r.squared, summary(lm4_model)$adj.r.squared, summary(lm5_model)$adj.r.squared),
  BIC = c(BIC(lm3_model), BIC(lm4_model), BIC(lm5_model))
)

results

#--------------------------------------------------------------

# Set seed for reproducibility
set.seed(123)

# Splitting indices for train-validation-test split
train_index1 <- sample(nrow(cars_no_high_leverage), 0.6 * nrow(cars_no_high_leverage))  # 60% for training
validation_test_index1 <- setdiff(1:nrow(cars_no_high_leverage), train_index1)

# Further split the remaining data into validation and test sets
validation_index1 <- sample(validation_test_index1, 0.5 * length(validation_test_index1))  # 20% for validation
test_index1 <- setdiff(validation_test_index1, validation_index1)

# Create train-validation-test datasets
train_data1 <- cars_no_high_leverage[train_index1, ]
validation_data1 <- cars_no_high_leverage[validation_index1, ]
test_data1 <- cars_no_high_leverage[test_index1, ]

#--------------------------------------------------------------

# Create the LM model 6
lm6_model <- lm(log_Pris ~ . - Pris - Län - Län1,  data = train_data1)

# Result
summary(lm6_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm6_model)

vif(lm6_model)

#--------------------------------------------------------------
# Calculate RMSE on Validation set
pre_val_lm6_model <- predict(lm6_model, newdata = validation_data1)

val_rmse_lm6_model <- rmse(validation_data1$log_Pris, pre_val_lm6_model)
results <- data.frame(
  Model = c("Model 4"),
  RMSE_val_data = c(val_rmse_lm6_model),
  Adj_R_squared = c(summary(lm6_model)$adj.r.squared),
  BIC = c(BIC(lm6_model))
)

results

#--------------------------------------------------------------

# Evaluating lm6_model on the test data
pre_test_lm6_model <- predict(lm6_model, newdata = test_data1) 
test_rmse_lm6_model <- rmse(test_data1$log_Pris, pre_test_lm6_model)

print(test_rmse_lm6_model)


lm6_model_test <- data.frame(actual = test_data1$log_Pris, predicted = pre_test_lm6_model)

# Plotting the data
ggplot(lm6_model_test, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Actual vs Predicted Prices", x = "Actual Prices", y = "Predicted Prices") +
  theme_minimal()

#--------------------------------------------------------------

# Exclude 'Län' and 'Pris' columns
cars_no_high_leverage <- subset(cars_no_high_leverage, select = -c(Län, Pris, Län1))

# Check the structure of the modified dataset
str(cars_no_high_leverage)

#--------------------------------------------------------------
# Create the LM model 6
lm6_model <- lm(log_Pris ~ .,  data = cars_no_high_leverage)

# Result
summary(lm6_model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm6_model)

vif(lm6_model)

#--------------------------------------------------------------
# Prediction
input <- data.frame(
  Modell = "1",
  Årsmodell = 2021,
  Miltal = 12003,
  Drivmedel = "Bensin",
  Växellåda = "Manuell",
  Län2 = "Halland",
  Motorstorlek = "1.6"
)

# Predict 
predict_input <- predict(lm6_model, newdata = input)
print(predict_input)

# Converting Price
predicted_price <- exp(predict_input)
print(predicted_price)

#--------------------------------------------------------------

# Confidence and Prediction Intervall
confidence_intervals <- predict(lm6_model, newdata = input, interval = "confidence", level = 0.95)
prediction_intervals <- predict(lm6_model, newdata = input, interval = "prediction", level = 0.95)

confidence_intervals
prediction_intervals

#--------------------------------------------------------------

# Convert log 
predicted_price = exp(12.32024)
confidence_interval_lower = exp(12.22545)
confidence_interval_upper = exp(12.41504)

prediction_interval_lower = exp(11.94079)
prediction_interval_upper = exp(12.6997)

# Print transformed prices
print(predicted_price)

cat("Confidence Interval: [", confidence_interval_lower, ", ", confidence_interval_upper, "] \n")
cat("Prediction Interval: [", prediction_interval_lower, ", ", prediction_interval_upper, "] \n")

#--------------------------------------------------------------

# Calculate the frequency of each category
vaxellada_counts <- table(cars1$Drivmedel)

# Calculate the percentage of each category
percentage <- prop.table(vaxellada_counts) * 100

# Create a bar plot
barplot(vaxellada_counts, 
        main = "Distribution of Växellåda",
        xlab = "Växellåda", 
        ylab = "Frequency")

# Add frequency values to each bar
text(x = barplot(vaxellada_counts), 
     y = vaxellada_counts, 
     label = vaxellada_counts,
     pos = 3, cex = 0.8, col = "black")

# Add percentage values to each bar
text(x = barplot(vaxellada_counts), 
     y = vaxellada_counts, 
     label = paste(round(percentage, 1), "%"),
     pos = 1, cex = 0.8, col = "blue")

#--------------------------------------------------------------

# Given numbers
numbers <- c(293262.4, 315001.7, 353543.9, 357392.4, 404480.5, 477688, 613495.3)

# Calculate the percentage increase
percentage_increase <- c(NA, diff(numbers) / numbers[-length(numbers)] * 100)

# Create a line plot
plot(numbers, type = "o", col = "blue", ylim = c(0, max(numbers) * 1.1), xlab = "Index", ylab = "Value", main = "Line Diagram with Percentage Increase")

# Add the percentage increase as text at each point
text(x = 1:length(numbers), y = numbers, labels = paste0(round(percentage_increase, 2), "%"), pos = 3)