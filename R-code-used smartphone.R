# libraries
library(dplyr)
library(readr)
library(corrplot)
library(ggplot2)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(fastDummies)
library(e1071)

# Data set
data <- read.csv("C:\\Users\\Lenovo\\Downloads\\used_device_data.csv")

# Structure and summary of data 
str(data)
summary(data)


# factoring the categorical columns
data$device_brand <- as.factor(data$device_brand)
data$os <- as.factor(data$os)
data$'X4g' <- as.factor(data$'X4g')
data$'X5g' <- as.factor(data$'X5g')

# Checking 
str(data)
summary(data)

#Handling Na for rear_camera_mp
# In some instance where rear camera pixel is lessar that front camera 'unusal form common scence point of view'
# In common view always rear camera is greater than front camera.
sum(data$rear_camera_mp < data$front_camera_mp, na.rm=T)
sum(data$rear_camera_mp > data$front_camera_mp, na.rm=T)

# Filling NA in rear_camera_mp with group median (by front_camera_mp)
# first we group the front camera, there can be only 3 to 4 values of rear camera,for every front camera 
# we do the median of rear camera and fill NA by this we imputate with the atmost actual value.
as.data.frame(data <- data %>%
  group_by(front_camera_mp) %>%
  mutate(
    rear_camera_mp = ifelse(
      is.na(rear_camera_mp),
      median(rear_camera_mp, na.rm = TRUE),
      rear_camera_mp
    )
  ) %>%
  ungroup())


# filling NA values in Front camera, when rear camera == 12.2
med_front <- median(data$front_camera_mp[data$rear_camera_mp == 12.2], na.rm = TRUE)
data$front_camera_mp[is.na(data$front_camera_mp)] <- med_front


summary(data)

# filling NA in internal_memory
med_internal <- median(data$internal_memory, na.rm = T)
data$internal_memory[is.na(data$internal_memory)] <- med_internal

#filling NA in ram
data[is.na(data$ram),] # finding out missing values

ram_0.06_median <- median(data$ram[data$internal_memory == 0.06],na.rm =T) # median value
data$ram[is.na(data$ram)] <- ram_0.06_median
summary(data)

# filling NA values in battery
as.data.frame(data[is.na(data$battery),])
battery_median <- median(data$battery, na.rm = T) # finding median value

data$battery[is.na(data$battery)] <- battery_median

# filling NA values in weight
as.data.frame(data[is.na(data$weight),])

weight_median <- median(data$weight, na.rm = T)
data$weight[is.na(data$weight)] <- weight_median

summary(data)

#*****************************************************************

# Handling at most Zero values in front_camera

sum(data$front_camera_mp < 1) # totoal less than 1 values
(unique(as.data.frame(data[data$front_camera_mp < 1,c('release_year')]))) # finding out which year

front_median <- median(data$front_camera_mp)
data$front_camera_mp[data$front_camera_mp <1] <- front_median # handling zero values

#*********************************************************************

# Handling at most Zero values in rear_camera
sum(data$rear_camera_mp < 1) # totoal less than 1 values
(unique(as.data.frame(data[data$rear_camera_mp < 1,c('release_year')]))) # finding out which year

rear_median <- median(data$rear_camera_mp)
data$rear_camera_mp[data$rear_camera_mp <1] <- rear_median # handling zero values

summary(data)
colnames(data)

#********************************************************************

summary(data$screen_size)
summary(data$battery)
summary(data$weight)

# ideal phone specification Screen size- 26 cm, battery-5500 mAh and weight 253 grams
as.data.frame(data[data$screen_size > 26 & data$battery > 5500 & data$weight > 253,])


data <- data[!(data$screen_size > 26 & data$battery > 5500 & data$weight > 253), ]
str(data)

# any specification under screen size 11 and battery 1000 are considered as smart watches and removed.
data <- data[!(data$screen_size <11 & data$battery < 1000), ]

# Data Exploration

num_vars <- c("screen_size", "rear_camera_mp", "front_camera_mp", 
              "internal_memory", "ram", "battery", "weight", 
              "release_year", "days_used", 
              "normalized_used_price", "normalized_new_price")

cat_vars <- c("device_brand", "os", "X4g", "X5g")

#  Summary statistics

print(summary(select(data, all_of(num_vars))))

# Frequency count for categorical variable
for (v in cat_vars) {
  print(table(data[[v]]))
}

# Numeric variable plots

for (v in num_vars) {
  # Histogram
  ggplot(data, aes_string(x = v)) +
    geom_histogram(fill = "skyblue", color = "black", bins = 30, na.rm = TRUE) +
    labs(title = paste("Histogram of", v), x = v, y = "Count") -> p1
  print(p1)
  
  # Density
  ggplot(data, aes_string(x = v)) +
    geom_density(fill = "lightgreen", alpha = 0.6, na.rm = TRUE) +
    labs(title = paste("Density Plot of", v), x = v, y = "Density") -> p2
  print(p2)
}
# Categorical variable
for (v in cat_vars) {
  ggplot(data, aes_string(x = v)) +
    geom_bar(fill = "orange", color = "black") +
    labs(title = paste("Bar Plot of", v), x = v, y = "Count") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) -> p
  print(p)
}

# correlation of variables 
num_vars <- data %>% select(where(is.numeric))

# 2. Convert to numeric matrix 
num_matrix <- data.matrix(num_vars)

# 3. Compute correlation matrix
cor_matrix <- cor(num_matrix, use = "complete.obs")

print(cor_matrix)

# heatmap of correlation matrix

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.7, tl.col = "black")


# Predictor analysis and relevancy

TGT <- "normalized_used_price"  
stopifnot(TGT %in% names(data))

# Split numeric vs categorical predictors 
cols <- setdiff(names(data), TGT)
num_cols <- cols[sapply(data[cols], is.numeric)]
cat_cols <- setdiff(cols, num_cols)

# Relevancy of NUMERIC predictors: correlation with target variable
num_corr <- sapply(num_cols, function(v)
  cor(data[[TGT]], data[[v]], use = "pairwise.complete.obs"))

num_corr_tbl <- data.frame(
  predictor = names(num_corr),
  r = as.numeric(num_corr),
  abs_r = abs(num_corr),
  row.names = NULL
)
num_corr_tbl <- num_corr_tbl[order(-num_corr_tbl$abs_r), ]
head(num_corr_tbl, 10)  # top numeric predictors

# Relevancy of CATEGORICAL predictors: one-way ANOVA p-values 
cat_pvals <- sapply(cat_cols, function(v) {
  df <- data[, c(TGT, v)]
  df <- df[complete.cases(df), ]
  if (nrow(df) == 0 || length(unique(df[[v]])) < 2) return(NA_real_)
  summary(aov(df[[TGT]] ~ as.factor(df[[v]]), data = df))[[1]][["Pr(>F)"]][1]
})

cat_pvals_tbl <- data.frame(
  predictor = names(cat_pvals),
  p_value = as.numeric(cat_pvals),
  row.names = NULL
)
cat_pvals_tbl <- cat_pvals_tbl[order(cat_pvals_tbl$p_value), ]
head(cat_pvals_tbl, 10)  # top categorical predictors (small p = more relevant)

#**************************************************************************
 # Data Partition
set.seed(123)
split = sample.split(data$normalized_used_price, SplitRatio = 0.8)
train_set = subset(data, split== TRUE)
test_set = subset(data, split == FALSE)

#*******************************
# Multiple linear Regression
# Building the multiple linear regression model
mlr_model <- lm(normalized_used_price ~ device_brand + os + X4g + X5g + normalized_new_price + battery +
                  rear_camera_mp + front_camera_mp + ram +
                  release_year + days_used,
                data = train_set)

# View model summary
summary(mlr_model)

# Predictions on test set
predictions <- predict(mlr_model, newdata = test_set)

# Comparing actual values with predicted values
results <- data.frame(
  Actual = test_set$normalized_used_price,
  Predicted = predictions,
  Residual = test_set$normalized_used_price - predictions
)
results
# Calculate RMSE
rmse_val <- sqrt(mean((test_set$normalized_used_price - predictions)^2))

# Calculate R-squared
ss_res <- sum((test_set$normalized_used_price - predictions)^2)
ss_tot <- sum((test_set$normalized_used_price - mean(test_set$normalized_used_price))^2)
r2_val <- 1 - (ss_res / ss_tot)

# Print results
print(paste("RMSE:", rmse_val))
print(paste("R-squared:", r2_val))

#************************************************

# Decision Tree Regression
# Fitting a decision tree (regression)
dec_tree <- rpart(
  normalized_used_price ~ ., 
  data = train_set,
  method = "anova", #Analysis of Variance
  control = rpart.control(cp = 0.01, minsplit = 20)
)

# Predicting on test set
dec_pred <- predict(dec_tree, newdata = test_set)

# Evaluate (RMSE and R-squared)
rmse <- sqrt(mean((test_set$normalized_used_price - dec_pred)^2))
r2   <- 1 - sum((test_set$normalized_used_price - dec_pred)^2) /
  sum((test_set$normalized_used_price - mean(test_set$normalized_used_price))^2)

print(paste("RMSE:", rmse))
print(paste("R-squared:", r2))

# finding best cp
best_cp <- dec_tree$cptable[which.min(dec_tree$cptable[,"xerror"]), "CP"]
print(paste("Best cp:", best_cp))

# Prune the tree with best cp
pruned_tree <- prune(dec_tree, cp = best_cp)

# Plot pruned tree
rpart.plot(pruned_tree, type = 2, extra = 101, fallen.leaves = TRUE)

#***************************************************

# Random forest Regression

rf_model <- randomForest(
  normalized_used_price ~ .,   
  data = train_set,
  ntree = 500,                 # number of trees
  mtry = 3,                    # number of variables tried at each split
  importance = TRUE            # to check variable importance
)
print(rf_model)

rf_predict <- predict(rf_model, newdata = test_set)

rf_rmse <- sqrt(mean((rf_predict - test_set$normalized_used_price)^2))
rf_r2   <- 1 - sum((rf_predict - test_set$normalized_used_price)^2) / 
  sum((mean(train_set$normalized_used_price) - test_set$normalized_used_price)^2)

print(paste("RMSE:", rf_rmse))
print(paste("R-squared:", rf_r2))

varImpPlot(rf_model, type=1) # variable importance plot
#**********************************************************************************

# Adding new column for classification model 
# Top 30% as High, rest as Low (above 4.670134 is considered as high)
data$Price_Class <- ifelse(data$normalized_used_price > 
                             quantile(data$normalized_used_price, 0.70, na.rm = TRUE),
                           "High", "Low")

data$Price_Class <- as.factor(data$Price_Class)

#*****************************************************************


# Data partition

set.seed(123)
split_cal = sample.split(data$Price_Class, SplitRatio = 0.8)
train_cal = subset(data, split_cal== TRUE)
test_cal = subset(data, split_cal == FALSE)

#*****************************************************************************
# Removing numerical predictor 

train_cal <- subset(train_cal, select = -c(normalized_used_price))
test_cal  <- subset(test_cal,  select = -c(normalized_used_price))

#******************************************************************

# Classification Model
# Decision Tree Classification (CART)

cart_model <- rpart(
  Price_Class ~ .,       # target is categorical variable (High / Low)
  data = train_cal,
  method = "class",      # classification tree
  control = rpart.control(
    cp = 0.01,           # complexity parameter (pruning)
    minsplit = 20,       # minimum observations to attempt a split
    maxdepth = 10        # maximum depth of the tree
  )
)

print(cart_model)   # if -then rule for classification.
summary(cart_model) 

# Ploting the tree
rpart.plot(cart_model, type = 2, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree Classification (CART)")
# prediting on the test data
pred_cart <- predict(cart_model, newdata = test_cal, type = "class")

# confusion matrix
cm_dec <- table(Actual = test_cal$Price_Class, Predicted = pred_cart)
print(cm_dec)

#********************************************************************
# Random Forest Classification

set.seed(123)
rf_cal <- randomForest(
  Price_Class ~ ., 
  data = train_cal, 
  ntree = 500, 
  importance = TRUE
)

pred_rf_cal <- predict(rf_cal, newdata = test_cal)

# Confusion matrix
cm_rfc <- table(Actual = test_cal$Price_Class, Predicted = pred_rf_cal)
print(cm_rfc)

varImpPlot(rf_cal)

#*************************************************************************
# Naive bayes

nb_model <- naiveBayes(Price_Class ~ ., data = train_cal, laplace = 1)

# Predict on test set
pred_nb <- predict(nb_model, newdata = test_cal)

# Confusion matrix
cm_nb <- table(Actual = test_cal$Price_Class, Predicted = pred_nb)
print(cm_nb)

