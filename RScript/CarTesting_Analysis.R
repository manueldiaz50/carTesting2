######################################
# Libraries
######################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(fastICA)) install.packages("fastICA", repos = "http://cran.us.r-project.org")
if(!require(kknn)) install.packages("kknn", repos = "http://cran.us.r-project.org")



######################################
# Create train and validation datasets
######################################

# load data
dl <- tempfile()
download.file("https://raw.githubusercontent.com/manueldiaz50/CarTesting/master/Data/train.csv", dl)
data <- read.csv(dl, header = TRUE)

# convert categorical variables to binary
data_bin <- data %>% select(ID, y, X0, X1, X2, X3, X4, X5, X6, X8) %>% 
  gather(key = "feature", value = "value", -y, -ID) %>% 
  mutate(present = 1, value = paste(feature, value, sep ="_") ) %>%
  select(-feature) %>%
  spread(value, present, fill = 0)
data_bin <- data %>% select(-y, -X0, -X1, -X2, -X3, -X4, -X5, -X6, -X8) %>%
  left_join(data_bin, by = "ID") 

# leave only the ground truth and the features. Remove ID field
data_bin <- data_bin %>% select(-ID)

# create train and validation datasets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data_bin$y, times = 1, p = 0.1, list = FALSE)
train <- data_bin[-test_index,]
validation <- data_bin[test_index,]

rm(test_index, dl)

#####################################
# Exploratory analysis
#####################################

tally(data)
head(data)
str(data)


# outcome Y

data %>% ggplot(aes(y)) + geom_histogram(bins = 50) # 2 or 3 modes suggest clusters
data %>% select(y) %>% summary()
# y         
# Min.   : 72.11  
# 1st Qu.: 90.82  
# Median : 99.15  
# Mean   :100.67  
# 3rd Qu.:109.01  
# Max.   :265.32  
sd(data$y)
# 12.67938

# features
#
c <- data %>% select(-ID, -y) %>% sapply(class)
t <- data %>% gather(key = "feature", value = "value", -y, -ID)

length(c[c == "factor"])
length(c[c != "factor"])
# 8 categorical and 368 numerical.

factors <- names(c[c == "factor"])
numerical <- names(c[c != "factor"])

# factors  X0       X1       X2       X3       X4       X5       X6       X8 
t %>% filter(feature %in% factors) %>%
  ggplot(aes(value, y, color = value)) +
  geom_boxplot() + 
  facet_wrap( ~ feature, scales = "free") +
  theme(legend.position = "none")
# most observations are in X4(d) not much info would come from this feature.
t %>% filter(feature %in% factors) %>%
  ggplot(aes(value, fill = feature)) +
  geom_bar(stat = "count") + 
  facet_wrap( ~ feature, scales = "free") +
  theme(legend.position = "none")

# numerical X9 to X385 --> all binary 0 = feature not present / 1 = feature present
t %>% filter(feature %in% numerical) %>%
  ggplot(aes(feature, fill = value)) + 
  geom_bar(stat = "count", position = "stack") + 
  coord_flip() +
  theme_void()

################################
# Loss function
################################

# Computes the RMSE for vectors of predictions and their corresponding true ratings

RMSE <- function(true_y, predicted_y){
  sqrt(mean((true_y - predicted_y)^2, na.rm = TRUE))
}

options(digits = 5)

#############
# Naive Bayes
#############

start <- Sys.time()
mu_hat = mean(train$y)
end <- Sys.time()

rmse <- RMSE(train$y, mu_hat)
rmse
# create a table with the results
rmse_results <- tibble(method = "mu_hat",  runtime = end - start,
                       expectedRMSE = sd(train$y),
                       RMSE = RMSE(validation$y, mu_hat))

############
# Regression
############

# linear regression

start <- Sys.time()
fit_lm <- train(y ~ ., method = "lm", data = train)
pred_lm <- predict(fit_lm, validation, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "linear regression", runtime = end - start,
                                               expectedRMSE = fit_lm$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_lm)))

# logistic regression

start <- Sys.time()
fit_glm <- train(y ~ ., method = "glm", data = train)
pred_glm <- predict(fit_glm, validation, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "logistic regression", runtime = end - start,
                                               expectedRMSE = fit_glm$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_glm)))

# Regression trees

start <- Sys.time()
fit_rpart <- train(y ~ ., method = "rpart", data = train)
pred_rpart <- predict(fit_rpart, validation, method = "raw")
end <- Sys.time()

ggplot(fit_rpart)
plot(fit_rpart$finalModel, margin = 0.1)
text(fit_rpart$finalModel)

rmse_results <- bind_rows(rmse_results, tibble(method = "Regression trees", runtime = end - start,
                                               expectedRMSE = min(fit_rpart$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_rpart)))

### too many features for random forest or kernel algorithms ###

####################
# Variable reduction
####################

# Principal Component Analysis
#

m <- train %>% select(-y)
m <- as.matrix(m)
pca <- prcomp(m)

qplot(1:563, pca$sdev)
variance_explained <- summary(pca)$importance[3,] 
variance_explained <- variance_explained[variance_explained <= 0.99]
variance_explained[which.max(variance_explained)]
k <- which.max(variance_explained)
# 218 dimensions explain 99% of variability

# transform train and validation
train_pca <- as_tibble(pca$x[, 1:k]) %>% mutate(y = train$y)

col_means <- colMeans(m)
validation_pca <- validation %>% select(-y)
validation_pca <- as.matrix(validation_pca)
validation_pca <- sweep(validation_pca, 2, col_means) %*% pca$rotation
validation_pca <- as_tibble(validation_pca[, 1:k]) %>% mutate(y = validation$y)

# Independent components analysis
#
m <- train %>% select(-y)

sources <- 2:100

# find optimal number of sources
rmses <- sapply(sources, function(s) {
  ica <- fastICA(m, s)
  
  # transform train and validation
  train_ica <- as_tibble(ica$S) %>% mutate(y = train$y)
  
  fit_knn_ica <- train(y ~ ., method = "knn", data = train_ica)
  rmse <- min(fit_knn_ica$results$RMSE)
  print(rmse)
  return(rmse)
})

qplot(sources, rmses)
df <- tibble(sources, rmses)
best_source <- sources[which.min(rmses)]
# 12

# calculate predictors with the best source number
ica <- fastICA(m, best_source)

# transform train and validation
train_ica <- as_tibble(ica$S) %>% mutate(y = train$y)
X <- validation %>% select(-y) %>% as.matrix()
S_hat <- X %*% ica$K %*% ica$W
validation_ica <- as_tibble(S_hat) %>% mutate(y = validation$y)

# Variable importance
#

imp <- names(fit_rpart$finalModel$variable.importance)
imp

# transform train and validation. Reduce to the important variables
train_imp <- train %>% select(y, imp)
validation_imp <- validation %>% select(y, imp)

# Clustering -> no use
#
x <- train %>% select(-y) %>% as.matrix()

k <- kmeans(x, centers = 5)

train_cluster <- train %>% mutate(cluster = k$cluster)

train_cluster %>% mutate(cluster = as.factor(cluster)) %>% ggplot(aes(cluster, y)) +
  geom_boxplot()
train_cluster %>% mutate(cluster = as.factor(cluster)) %>% ggplot(aes(cluster, y)) +
  geom_jitter()

train_cluster %>% select(y, imp, cluster) %>%
  mutate(cluster = factor(cluster)) %>%
  gather(key = "feature", value = "value", -y, -cluster) %>%
  ggplot(aes(value, y, color = cluster)) +
    geom_jitter() + 
    facet_wrap( ~ feature, scales = "free") 

train_cluster %>% select(y, imp, cluster) %>%
  gather(key = "feature", value = "value", -y, -cluster) %>%
  mutate(cluster = factor(cluster)) %>%
  group_by(cluster, feature) %>%
  summarise(n = sum(value)) %>%
  ggplot(aes(cluster, feature, size = n)) +
  geom_point() 


##########################
# Kernel Nearest Neighbors
##########################

start <- Sys.time()
fit_knn <- train(y ~ ., method = "knn", data = train_pca)
pred_knn <- predict(fit_knn, validation_pca, method = "raw")
end <- Sys.time()

ggplot(fit_knn, highlight = TRUE)

rmse_results <- bind_rows(rmse_results, tibble(method = "PCA Kernel nearest neighbors", runtime = end - start,
                                               expectedRMSE = min(fit_knn$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_knn)))

start <- Sys.time()
fit_knn_ica <- train(y ~ ., method = "knn", data = train_ica)
pred_knn_ica <- predict(fit_knn_ica, validation_ica, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "ICA Kernel nearest neighbors", runtime = end - start,
                                               expectedRMSE =min(fit_knn_ica$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_knn_ica)))

start <- Sys.time()
fit_knn_imp <- train(y ~ ., method = "kknn", data = train_imp)
pred_knn_imp <- predict(fit_knn_imp, test_imp)
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "IMP Kernel nearest neighbors", runtime = end - start,
                                               expectedRMSE = min(fit_knn_imp$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_knn_imp)))


############
# Regression
############

# linear regression

start <- Sys.time()
fit_lm_pca <- train(y ~ ., method = "lm", data = train_pca)
pred_lm_pca <- predict(fit_lm_pca, validation_pca, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "PCA linear regression", runtime = end - start,
                                               expectedRMSE = fit_lm_pca$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_lm_pca)))

start <- Sys.time()
fit_lm_ica <- train(y ~ ., method = "lm", data = train_ica)
pred_lm_ica <- predict(fit_lm_ica, validation_ica, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "ICA linear regression", runtime = end - start,
                                               expectedRMSE = fit_lm_ica$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_lm_ica)))

start <- Sys.time()
fit_lm_imp <- train(y ~ ., method = "lm", data = train_imp)
pred_lm_imp <- predict(fit_lm_imp, validation_imp, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "IMP linear regression", runtime = end - start,
                                               expectedRMSE = fit_lm_imp$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_lm_imp)))

# logistic regression

start <- Sys.time()
fit_glm_pca <- train(y ~ ., method = "glm", data = train_pca)
pred_glm_pca <- predict(fit_glm_pca, validation_pca, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "PCA logistic regression", runtime = end - start,
                                               expectedRMSE = fit_glm_pca$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_glm_pca)))

start <- Sys.time()
fit_glm_ica <- train(y ~ ., method = "glm", data = train_ica)
pred_glm_ica <- predict(fit_glm_ica, validation_ica, method = "raw")
end <- Sys.time()

rmse_results <- bind_rows(rmse_results, tibble(method = "ICA logistic regression", runtime = end - start,
                                               expectedRMSE = fit_glm_ica$results$RMSE,
                                               RMSE = RMSE(validation$y, pred_glm_ica)))

# Regression trees

start <- Sys.time()
fit_rpart_pca <- train(y ~ ., method = "rpart", data = train_pca)
pred_rpart_pca <- predict(fit_rpart_pca, validation_pca, method = "raw")
end <- Sys.time()

ggplot(fit_rpart_pca)
plot(fit_rpart_pca$finalModel, margin = 0.1)
text(fit_rpart_pca$finalModel)

rmse_results <- bind_rows(rmse_results, tibble(method = "PCA Regression trees", runtime = end - start,
                                               expectedRMSE = min(fit_rpart_pca$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_rpart_pca)))

start <- Sys.time()
fit_rpart_ica <- train(y ~ ., method = "rpart", data = train_ica)
pred_rpart_ica <- predict(fit_rpart_ica, validation_ica, method = "raw")
end <- Sys.time()

ggplot(fit_rpart_ica)
plot(fit_rpart_ica$finalModel, margin = 0.1)
text(fit_rpart_ica$finalModel)

rmse_results <- bind_rows(rmse_results, tibble(method = "ICA Regression trees", runtime = end - start,
                                               expectedRMSE = min(fit_rpart_ica$results$RMSE),
                                               RMSE = RMSE(validation$y, pred_rpart_ica)))


#################
# Result analysis
#################

e <- validation$y - pred_lm_pca

qplot(e)
summary(e)
qqplot(e)


###################################################################################################################

set.seed(1, sample.kind="Rounding")
fit_rf_imp <- train(y ~ ., method = "Rborist", data = train_imp)
pred_rf_imp <- predict(fit_rf_imp, test_imp, method = "raw")

f <- names(data_bin)
f <- f[f %like% "X1_.*"]
data %>% select(ID, X1) %>%
  left_join(data_bin, by = 'ID') %>%
  select(X1, f) %>%
  distinct() %>% knitr::kable()

f <- names(data_bin)
f <- f[f %like% "X1_.*"]
f
