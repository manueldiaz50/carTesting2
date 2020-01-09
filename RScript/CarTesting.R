######################################
# Libraries
######################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(fastICA)) install.packages("fastICA", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

######################################
# Create train and validation datasets
######################################

# load data
dl <- tempfile()
download.file("https://raw.githubusercontent.com/manueldiaz50/carTesting2/master/Data/train.csv", dl)
data <- read.csv(dl, header = TRUE)

# convert categorical variables to binary
data_bin <- data %>% select(ID, y, X0, X1, X2, X3, X4, X5, X6, X8) %>% 
  gather(key = "feature", value = "value", -y, -ID) %>% 
  mutate(present = 1, value = paste(feature, value, sep ="_") ) %>%
  select(-feature) %>%
  spread(value, present, fill = 0)
data_bin <- data %>% select(-y, -X0, -X1, -X2, -X3, -X4, -X5, -X6, -X8) %>%
  left_join(data_bin, by = "ID") 

# leave only the ground truth and the features. Remove ID field that will not be used in the model
data_bin <- data_bin %>% select(-ID)

# create train and validation datasets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data_bin$y, times = 1, p = 0.1, list = FALSE)
carTesting <- data_bin[-test_index,]
validation <- data_bin[test_index,]

rm(test_index, dl)

################################
# Loss function
################################

# Computes the RMSE for vectors of predictions and their corresponding true ratings

RMSE <- function(true_y, predicted_y){
  sqrt(mean((true_y - predicted_y)^2, na.rm = TRUE))
}


################################
# Variable reduction
################################

# Computes a new predictor space based in Principal Components Analysis
#
# Parameters:
#   data: dataset
#   v: percentage of the variance conveyed by the selected components
#
# Returns a list with the following components:
#   data: transformed dataset with reduced list of predictors
#   rotation: rotation vector that will be used to transform new data for prediction
#   means = column means also to be used to transform new data for prediction
#   n_components: final number of predictors
#
CarTesting_PCA <- function(data, v = 0.99) {
  
  # remove output variable and convert the predictors space into a matrix
  m <- data %>% select(-y)
  m <- as.matrix(m)
  # principal component analysis
  pca <- prcomp(m)
  
  # get the k components that account for the %v of the variance 
  variance_explained <- summary(pca)$importance[3,] 
  variance_explained <- variance_explained[variance_explained <= v]
  variance_explained[which.max(variance_explained)]
  k <- which.max(variance_explained)

  # transform to the reduced predictors space
  data_pca <- as_tibble(pca$x[, 1:k]) %>%
    # plug back the output variable y
    mutate(y = data$y)
  
  r <- list(data = data_pca, rotation = pca$rotation, means = colMeans(m), n_components = k)
  
  return(r)

}

# Computes a new predictor space based in Independent Components Analysis
#
# Parameters:
#   data: dataset
#   s: desired number of sources
#
# Returns a list with the following components:
#   data: transformed dataset with reduced list of predictors
#   X, K, W: transormation matrices
#   
CarTesting_ICA <- function(data, s = 26) {
  
  # remove the output variable y
  m <- data %>% select(-y)
  # independent components analysis
  ica <- fastICA(m, s)
  
  # transform to the new predictor space
  data_ica <- as_tibble(ica$S) %>% 
    # plug back the ouput variable y
    mutate(y = data$y)
  
  r <- list(data = data_ica, X = ica$X, K = ica$K, W = ica$W)
  
  return(r)

}

# Computes a new predictor space based in a regression tree analysis
#
# Parameters:
#   data: dataset
#
# Returns a list with the following components:
#   data: transformed dataset with reduced list of predictors
#   features: the names of the selected predictors
#   
CarTesting_VarImp <- function(data) {

  # fit a regression tree
  fit_rpart <- train(y ~ ., method = "rpart", data = data)
  
  # get the names of the important predictors
  features_list <- names(fit_rpart$finalModel$variable.importance)
  
  # Reduce the data and keep only the predictors in the list
  data_reduced <- data %>% select(y, features_list)

  r <- list(data = data_reduced, features = features_list)
  
  return(r)
  
}

################################
# Train function
################################

# Fits the models with the reduced predictors spaces
#
# Parameters:
#   data: dataset with the predictors and the output variable
#   
# Returns a list with the following components:
#   fit_pca: lm model trained with PCA predictors
#   fit_ica: lm model trained with ICA predictors
#   fit_varimp: lm model trained with important variables after regression tree analysis
#   fit_varimp_rf: random forest model trained with important variables
#   pca: PCA predictors and transformation elements
#   ica: ICA predictors and transformation elements
#   varimp: important variables predictors
#
CarTesting_Train <- function(data) {

  # dimension reduction - PCA
  pca <- CarTesting_PCA(data)
  data_pca <- pca$data
  data_pca <- data_pca
  np_pca <- ncol(data_pca)
  
  # dimension reduction - ICA
  ica <- CarTesting_ICA(data, s = 26)
  data_ica <- ica$data
  
  # dimension reduction - regression tree important variables
  varimp <- CarTesting_VarImp(data)
  data_varimp <- varimp$data
  
  # train linear regression models with the different features spaces
  fit_pca_lm <- train(y ~ ., method = "lm", data = data_pca)
  fit_ica_lm <- train(y ~ ., method = "lm", data = data_ica)
  fit_varimp_lm <- train(y ~ ., method = "lm", data = data_varimp)
  
  # train random forests with varimp and ica predictors
  fit_varimp_rf <- randomForest(y ~ ., data = data_varimp) 
  fit_ica_rf <- randomForest(y ~ ., data = data_ica)
  
  r <- list(fit_pca = fit_pca_lm, fit_ica = fit_ica_lm, fit_varimp = fit_varimp_lm, fit_varimp_rf = fit_varimp_rf,
            fit_ica_rf = fit_ica_rf,
            pca = pca, ica = ica, varimp = varimp)
}


################################
# Predict function
################################

# Computes predictions with a model ensemble produced by the function CarTesting_Train
#
# Parameters:
#   fit_CarTesting: model ensemble
#   data: dataset
#   w_pca: weight of PCA prediction in the final ensemble
#   w_varimp: weight of the important variables prediction in the final ensemble
#   w_ica: weight of the ICA prediction in the final ensemble
#   w_rf and w_rf2: weight of the random forests predictions in the final ensemble
#
# Returns:
#   pred: vector with the predicted output values.
#
CarTesting_Predict <- function(fit_CarTesting, data, w_pca = 1, w_varimp = 1, w_ica = 0, w_rf = 0, w_rf2 = 0) {

  # pull all the models from the ensemble
  fit_pca <- fit_CarTesting$fit_pca
  fit_ica <- fit_CarTesting$fit_ica
  fit_ica_rf <- fit_CarTesting$fit_ica_rf
  fit_varimp <- fit_CarTesting$fit_varimp
  fit_varimp_rf <- fit_CarTesting$fit_varimp_rf
  
  # pull all the predictors spaces
  predictors_pca <- fit_CarTesting$pca
  predictors_ica <- fit_CarTesting$ica
  predictors_varimp <- fit_CarTesting$varimp
  
  # predict with pca predictors
  #
  # tranform the input predictors to the pca reduced dimension space
  col_means <- predictors_pca$means
  data_reduced <- data %>% select(-y)
  data_reduced <- as.matrix(data_reduced)
  data_reduced <- sweep(data_reduced, 2, col_means) %*% predictors_pca$rotation
  data_reduced <- as_tibble(data_reduced[, 1:predictors_pca$n_components]) %>% mutate(y = data$y)
  # compute the prediction
  pred_lm_pca <- predict(fit_pca, data_reduced, method = "raw")
 
  # predict with ica predictors
  #
  # transform the input predictors to the ica reduced space
  X <- data %>% select(-y) %>% as.matrix()
  S_hat <- X %*% predictors_ica$K %*% predictors_ica$W
  data_reduced <- as_tibble(S_hat) %>% mutate(y = data$y)
  # compute the prediction
  pred_lm_ica <- predict(fit_ica, data_reduced, method = "raw")
  pred_rf_ica <- predict(fit_ica_rf, data_reduced, method = "raw")
  
  # predict on important variables
  #
  # pick only the important predictors
  data_reduced <- data %>% select(y, predictors_varimp$features)
  # compute a prediction with linear regression
  pred_lm_varimp <- predict(fit_varimp, data_reduced, method = "raw")
  # compute a prediction with random forests algorithm
  pred_rf_varimp <- predict(fit_varimp_rf, data_reduced, method = "raw")
  
  # predictions assembly
  pred <-  (pred_lm_pca * w_pca + pred_lm_varimp * w_varimp + pred_lm_ica * w_ica + pred_rf_varimp * w_rf +
              pred_rf_ica * w_rf2) / 
    (w_pca + w_varimp + w_ica + w_rf + w_rf2)
  
  return(pred)
  
}

##################
# Cross-validation
##################

options(digits = 5)

# Create new train and test dataset
#
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = carTesting$y, times = 1, p = 0.2, list = FALSE)
train_cv <- carTesting[-test_index,]
test_cv <- carTesting[test_index,]

# Train the models ensemble on train_cv dataset 1.40 minutes
#
set.seed(1, sample.kind="Rounding")
fit_ensemble <- CarTesting_Train(train_cv)

# Find optimal weight for PCA model - 1.96 minutes
#
rmses <- sapply(1:50, function(w) {
  pred_ensemble <- CarTesting_Predict(fit_ensemble, test_cv, w_pca = w, w_varimp = 1, w_ica = 1, w_rf = 1, w_rf2 = 1)
  rmse <- RMSE(test_cv$y, pred_ensemble)
  return(rmse)
})
qplot(x = 1:50, y = rmses, xlab = "PCA weight", ylab = "rmse")
w_pca <- which.min(rmses)
expectedRMSE <- min(rmses)
results <- tibble(Parameter = "PCA weight", Value = w_pca, Expected_RMSE = expectedRMSE)

# Find optimal weight for VarImp model - 1.96 minutes
#
rmses <- sapply(1:50, function(w) {
  pred_ensemble <- CarTesting_Predict(fit_ensemble, test_cv, w_pca = w_pca, w_varimp = w, w_ica = 1, w_rf = 1, 
                                      w_rf2 = 1)
  rmse <- RMSE(test_cv$y, pred_ensemble)
  return(rmse)
})
qplot(x = 1:50, y = rmses, xlab = "VarImp weight", ylab = "rmse")
w_varimp <- which.min(rmses)
expectedRMSE <- min(rmses)
results <- bind_rows(results, tibble(Parameter = "VarImp weight", Value = w_varimp, Expected_RMSE = expectedRMSE))

# Find optimal weight for ICA model - 1.96 minutes
#
rmses <- sapply(1:50, function(w) {
  pred_ensemble <- CarTesting_Predict(fit_ensemble, test_cv, w_pca = w_pca, w_varimp = w_varimp, w_ica = w, w_rf = 1,
                                      w_rf2 = 1)
  rmse <- RMSE(test_cv$y, pred_ensemble)
  return(rmse)
})
qplot(x = 1:50, y = rmses, xlab = "ICA weight", ylab = "rmse")
w_ica <- which.min(rmses)
expectedRMSE <- min(rmses)
results <- bind_rows(results, tibble(Parameter = "ICA weight", Value = w_ica, Expected_RMSE = expectedRMSE))

# Find optimal weight for random forests model - 1.96 minutes
#
rmses <- sapply(1:50, function(w) {
  pred_ensemble <- CarTesting_Predict(fit_ensemble, test_cv, w_pca = w_pca, w_varimp = w_varimp, w_ica = w_ica, 
                                      w_rf = w, w_rf2 = 1)
  rmse <- RMSE(test_cv$y, pred_ensemble)
  return(rmse)
})
qplot(x = 1:50, y = rmses, xlab = "RF weight", ylab = "rmse")
w_rf <- which.min(rmses)
expectedRMSE <- min(rmses)
results <- bind_rows(results, tibble(Parameter = "RF weight", Value = w_rf, Expected_RMSE = expectedRMSE))

# Find optimal weight for random forests model 2 - 1.96 minutes
#
rmses <- sapply(1:50, function(w) {
  pred_ensemble <- CarTesting_Predict(fit_ensemble, test_cv, w_pca = w_pca, w_varimp = w_varimp, w_ica = w_ica, 
                                      w_rf = w_rf, w_rf2 = w)
  rmse <- RMSE(test_cv$y, pred_ensemble)
  return(rmse)
})
qplot(x = 1:50, y = rmses, xlab = "RF 2 weight", ylab = "rmse")
w_rf2 <- which.min(rmses)
expectedRMSE <- min(rmses)
results <- bind_rows(results, tibble(Parameter = "RF 2 weight", Value = w_rf2, Expected_RMSE = expectedRMSE))

# print the final parameters values
results %>% knitr::kable()

######################################################
# Train the final model in the full carTesting dataset
######################################################

fit_ensemble_final <- CarTesting_Train(carTesting)

######################################
# Prediction on the validation dataset
######################################

pred <- CarTesting_Predict(fit_ensemble_final, validation, w_pca = w_pca, w_ica = w_ica, w_varimp = w_varimp, 
                           w_rf = w_rf, w_rf2 = w_rf2)
finalRMSE <- RMSE(validation$y, pred)
finalRMSE

###############################
# RMSE:    7.911
###############################


