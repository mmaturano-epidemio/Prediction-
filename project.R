# ==============================================================================
# ---- Machine Learning Course Project ----
# ==============================================================================

# ---- Load libraries and training set ---- 
library(caret)
library(data.table)
library(corrplot)
library(ggplot2)

dt <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
            na.strings = c("NA", "#DIV/0!", ""))

# ---- Clean dataset ----
dim(dt)

# The outcome variable "classe" must be a factor for classification
dt[, classe := as.factor(classe)]

# Remove variables that are irrelevant as predictors (administrative and time data)
dt <- dt[, .SD, .SDcols = !c("V1", "user_name", "raw_timestamp_part_1", 
                             "raw_timestamp_part_2", "cvtd_timestamp", 
                             "new_window", "num_window")]

# Identify and remove columns with a high percentage of missing values (>10%)
# These sparse columns will not make reliable predictors
dt <- dt[, .SD, .SDcols = !colMeans(is.na(dt))[colMeans(is.na(dt)) > .1] |> names()]

# Verify the presence of near-zero variance (NZV) predictors
nzv_metrics <- nearZeroVar(dt[, .SD, .SDcols = is.numeric], saveMetrics = TRUE)
# nzv_metrics[nzv_metrics$nzv == TRUE, ] # Returns empty. No NZV columns found.

# Handle Outliers: 
# During preliminary PCA, a severe outlier was identified indicating sensor failure.
# We locate and remove the observation with physically impossible sensor readings.
outlier_idx <- which.max(dt$accel_forearm_y)
dt <- dt[-outlier_idx]

# Save the pre-processed dataset for reproducibility and faster loading
saveRDS(dt, "weight_lifting_clean.rds")

# ==============================================================================
# ---- Exploratory Data Analysis (EDA) ----
# ==============================================================================

dt <- readRDS("weight_lifting_clean.rds") 

# Assess class distribution to rule out severe imbalance
dt[, .N, by = classe][, .(classe, percentage = 100 * N / sum(N))]

# Explore predictor collinearity using a correlation matrix
cor_matrix <- cor(dt[, .SD, .SDcols = is.numeric])

# Filter out weak correlations to isolate highly correlated predictor pairs
cor_filtered <- cor_matrix
cor_filtered[abs(cor_filtered) < 0.7] <- 0

corrplot(cor_filtered, method = "color", type = "lower", 
         diag = FALSE, tl.cex = 0.5, addCoef.col = NULL)
# Conclusion: Collinearity is present among sensors of the same device, but 
# overall, it does not dominate the dataset.

# Perform Principal Component Analysis (PCA) to assess linear separability
pca_res <- prcomp(dt[, .SD, .SDcols = is.numeric], scale. = TRUE)
pca_data <- data.table(pca_res$x[, 1:2], classe = dt$classe)

# PCA Visualization
ggplot(pca_data, aes(x = PC1, y = PC2, color = classe)) +
  geom_point(alpha = 0.3) +
  theme_minimal() +
  labs(title = "PCA of Pre-processed Dataset (PC1 vs PC2)")

# Scree Plot to evaluate variance explained by each principal component
plot(pca_res, type = "l", main = "Scree Plot: Variance by Component")

# Conclusion from EDA: Data separation is highly non-linear. Classes overlap 
# significantly in the principal component space. A linear model will likely underperform.

# ==============================================================================
# ---- Model Training Setup ----
# ==============================================================================

set.seed(1234) # For reproducibility
inTrain <- createDataPartition(dt$classe, p = 0.75, list = FALSE)
training <- dt[inTrain, ]
testing  <- dt[-inTrain, ]

# Set up 5-fold Cross-Validation to prevent overfitting and estimate out-of-sample error
control <- trainControl(method = "cv", number = 5)

# ---- Model 1: Random Forest ----
# We chose Random Forest due to its ability to handle non-linear relationships 
# and robustness against correlated predictors.
# We use the "ranger" method for a faster, optimized C++ implementation of Random Forest.

set.seed(1234)
mod_rf <- train(classe ~ ., 
                data = training, 
                method = "ranger", 
                trControl = control,
                importance = "impurity") # Calculates variable importance


# ---- Save the trained model ----
# We do this once the training finishes because model generation is a highly time consuming process 
saveRDS(mod_rf, "rf_model_ranger.rds")

# ---- Load the model in future sessions ----
# This is instantaneous and avoids re-training
mod_rf <- readRDS("rf_model_ranger.rds")
# Print the model summary
print(mod_rf)

# Extract variable importance from the ranger model
# Note: Ensure you used importance = "impurity" or "permutation" in the train function
var_imp <- varImp(mod_rf, scale = FALSE)

# Plotting the top 20 predictors
plot(var_imp, top = 20, main = "Top 20 Variable Importance")

# Converting to data.table for easier plotting
imp_dt <- as.data.table(var_imp$importance, keep.rownames = "Predictor")
setorder(imp_dt, -Overall)

ggplot(imp_dt[1:15], aes(x = reorder(Predictor, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Most Important Predictors", x = "Predictor", y = "Importance (Gini)")

# ==============================================================================
# ---- Model Validation & Out-of-Sample Error ----
# ==============================================================================

# Use the model to predict on the 25% "testing" set we held out initially
test_predictions <- predict(mod_rf, testing)

# Generate the Confusion Matrix to compare predictions against actual values
conf_matrix <- confusionMatrix(test_predictions, testing$classe)
print(conf_matrix)

# 1. Accuracy and Out-of-Sample Error Calculation
# The Accuracy is the proportion of correct classifications
accuracy <- conf_matrix$overall['Accuracy']
out_of_sample_error <- 1 - accuracy

# Display results clearly
cat("Estimated Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Estimated Out-of-Sample Error:", round(out_of_sample_error * 100, 2), "%\n")

# 2. Kappa Statistic
# This accounts for the possibility of the agreement occurring by chance.
# Values > 0.80 indicate almost perfect agreement.
cat("Kappa Statistic:", conf_matrix$overall['Kappa'], "\n")

# ==============================================================================
# ---- Model Testing ----
# ==============================================================================

# Load the official test set
final_test <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                    na.strings = c("NA", "#DIV/0!", ""))

# Apply the same column selection as in 'dt'
# (Make sure final_test has the same columns as your training set, minus 'classe')
final_test_clean <- final_test[, .SD, .SDcols = names(training)[-which(names(training) == "classe")]]
saveRDS(final_test_clean, "final_test_clean.rds")

# Final Predictions
final_preds <- predict(mod_rf, final_test_clean)

submission_results <- data.frame(
  problem_id = final_test$problem_id,
  prediction = final_preds
)

print(submission_results)
