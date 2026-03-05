install.packages(c(
  "tidyverse", "caret", "pROC", "PRROC",
  "ranger", "e1071", "glmnet", "DALEX", "ingredients", "xgboost"
))

library(tidyverse)
library(caret)
library(pROC)
library(PRROC)
library(ranger)
library(glmnet)
library(xgboost)
library(e1071)
library(DALEX)
library(ingredients)


set.seed(42)

#load data 
df <- hospital_readmissions_30k

summary(df)

# Drop ID 
if ("patient_id" %in% names(df)) df <- df %>% select(-patient_id)

df <- df %>% drop_na()


#mutate
df <- df %>%
  mutate(
    readmitted_30_days = factor(readmitted_30_days, levels = c("Yes","No")),
    diabetes = factor(diabetes),
    hypertension = factor(hypertension),
    gender = factor(gender),
    blood_pressure = factor(blood_pressure),
    discharge_destination = factor(discharge_destination),
    age_group = cut(age,
                    breaks = c(-Inf, 39, 59, 79, Inf),
                    labels = c("<=39", "40-59", "60-79", "80+"))
  )

# 5000 
set.seed(123)
idx_5k <- createDataPartition(df$readmitted_30_days,
                              p = 5000/nrow(df),
                              list = FALSE)
df <- df[idx_5k, ]

# 3) Train/Test split

set.seed(123)
idx <- createDataPartition(df$readmitted_30_days, p = 0.8, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]




# EDA 

df %>%
  count(readmitted_30_days) %>%
  mutate(pct = n / sum(n))

df %>%
  group_by(age_group) %>%
  summarise(rate = mean(readmitted_30_days == "Yes"), .groups = "drop")


# Train/Test Split 

set.seed(123)
idx <- createDataPartition(df$readmitted_30_days, p = 0.8, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]


# Dummy encoding 

dummies <- dummyVars(readmitted_30_days ~ ., data = train_df, fullRank = TRUE)

X_train <- predict(dummies, newdata = train_df) %>% as.data.frame()
X_test  <- predict(dummies, newdata = test_df) %>% as.data.frame()

y_train <- train_df$readmitted_30_days
y_test  <- test_df$readmitted_30_days


# Remove near-zero variance predictors

nzv <- nearZeroVar(X_train)
X_train2 <- X_train[, -nzv, drop = FALSE]
X_test2  <- X_test[, -nzv, drop = FALSE]

train_data2 <- X_train2 %>% mutate(readmitted_30_days = y_train)




# Modelings

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  allowParallel = TRUE
)

#GLMNET
set.seed(123)
model_glmnet <- train(
  readmitted_30_days ~ .,
  data = train_data2,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center", "scale"),  
  tuneLength = 5
)
model_glmnet

# Random Forest (ranger)
set.seed(123)
model_rf <- train(
  readmitted_30_days ~ .,
  data = train_data2,
  method = "ranger",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 3
)
model_rf


models <- list(GLM = model_glmnet, RF = model_rf)

# Model Performace Comparison 

evaluate_model <- function(model, name) {
  
  prob <- predict(model, X_test, type = "prob")[,"Yes"]
  roc_obj <- roc(y_test, prob)
  
  tibble(
    Model = name,
    ROC_AUC = as.numeric(auc(roc_obj)),
    Accuracy = mean((prob > 0.5) == (y_test == "Yes"))
  )
}

results <- bind_rows(
  lapply(names(models), function(n) evaluate_model(models[[n]], n))
)

results

# Outperforming model 
best_name <- results$Model[which.max(results$ROC_AUC)]
best_model <- models[[best_name]]

best_name





# Explainability + Fairness 
# Predict function (safe numeric output) 
predict_proba <- function(model, newdata) {
  as.numeric(predict(model, newdata, type = "prob")[, "Yes"])
}

# Create explainer (USE X_train2)
explainer <- explain(
  best_model,
  data = X_train2,
  y = ifelse(y_train == "Yes", 1, 0),
  predict_function = predict_proba,
  verbose = FALSE
)


# Global SHAP explanation

set.seed(42)

X_sample <- X_test2[sample(1:nrow(X_test2), 5), , drop = FALSE]

shap_all <- purrr::map_dfr(1:nrow(X_sample), function(i) {
  predict_parts(
    explainer,
    new_observation = X_sample[i, , drop = FALSE],
    type = "shap"
  )
})

shap_summary <- shap_all %>%
  group_by(variable) %>%
  summarise(mean_abs = mean(abs(contribution)), .groups = "drop") %>%
  arrange(desc(mean_abs))

shap_summary


# Breakdown (highest-risk patient)
prob <- predict(best_model, X_test2, type = "prob")[, "Yes"]
high_risk_index <- which.max(prob)

bd <- predict_parts(
  explainer,
  new_observation = X_test2[high_risk_index, , drop = FALSE],
  type = "break_down"
)

plot(bd)


# Ceteris Paribus

if ("length_of_stay" %in% colnames(X_test2)) {
  cp <- predict_profile(
    explainer,
    new_observation = X_test2[high_risk_index, , drop = FALSE],
    variables = c("length_of_stay"),
    type = "ceteris_paribus"
  )
  plot(cp)
}


# Fairness Evaluation (by age_group)
prob_test <- predict(best_model, X_test2, type = "prob")[, "Yes"]

pred_test <- factor(
  ifelse(prob_test > 0.5, "Yes", "No"),
  levels = c("No", "Yes")
)

fairness_age <- test_df %>%
  mutate(pred = pred_test) %>%
  group_by(age_group) %>%
  summarise(
    n = n(),
    predicted_positive_rate = mean(pred == "Yes"),
    TP = sum(pred == "Yes" & readmitted_30_days == "Yes"),
    FP = sum(pred == "Yes" & readmitted_30_days == "No"),
    TN = sum(pred == "No"  & readmitted_30_days == "No"),
    FN = sum(pred == "No"  & readmitted_30_days == "Yes"),
    TPR = TP / (TP + FN + 1e-9),
    FPR = FP / (FP + TN + 1e-9),
    .groups = "drop"
  )

fairness_age

# Disparity gaps 
fairness_gaps <- fairness_age %>%
  summarise(
    TPR_gap = max(TPR) - min(TPR),
    FPR_gap = max(FPR) - min(FPR),
    Positive_rate_gap = max(predicted_positive_rate) - min(predicted_positive_rate)
  )

fairness_gaps



