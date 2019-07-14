
## Score 0.18490

#https://iamkbpark.com/2018/01/22/deep-learning-with-keras-lime-in-r/#preprocess-data
suppressPackageStartupMessages({ library (readxl)
  library (keras);library (lime);library (tidyquant)
  library (rsample);library (recipes);library (yardstick)
  library (corrr);library(knitr);library (DT);library(tidyverse)
  library(tidyselect); library(keras); library(varhandle);library(ggplot2)
})

train <- read_csv("~/Desktop/GitContent/kaggle-HousePrices/train-2.csv")
test <- read_csv("~/Desktop/GitContent/kaggle-HousePrices/test-2.csv")
set.seed(1337)


#### pre processing
manage_na <- function(datafra)
{
  for(i in 1:ncol(datafra))
  {
    u <- datafra[,i]
    if(is.numeric(u))
    {
      #datafra[is.na(u),i] <- median(datafra[!is.na(u),i])
    } else
    {
      u <- levels(u)[u]
      u[is.na(u)] <- "Not Available" #a NA becomes a new category
      datafra[,i] <- as.factor(u)
    }
  }
  datafra
}


features <- rbind(train[,-c(1, ncol(train))], test[, -1])
#https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda
#features$PoolQC <-  varhandle::unfactor(features$PoolQC)
features$PoolQC[is.na(features$PoolQC)] <- "None"
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
features$PoolQC<-as.integer(plyr::revalue(features$PoolQC, Qualities))

features$PoolQC[2421] <- 2
features$PoolQC[2504] <- 3
features$PoolQC[2600] <- 2

features$Foundation <- as.factor(features$Foundation)

features$MoSold <- as.factor(features$MoSold)
features$GarageYrBlt[2593] <- 2007
features$TotBathrooms <- features$FullBath + (features$HalfBath*0.5) + features$BsmtFullBath + (features$BsmtHalfBath*0.5)

features$TotalSqFeet <- features$GrLivArea + features$TotalBsmtSF
features$Remod <- ifelse(features$YearBuilt==features$YearRemodAdd, 0, 1) #0=No Remodeling, 1=Remodeling
features$Age <- as.numeric(features$YrSold)-features$YearRemodAdd
features$IsNew <- ifelse(features$YrSold==features$YearBuilt, 1, 0)
features$YrSold <- as.factor(features$YrSold)

features$NeighRich[features$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
features$NeighRich[!features$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
features$NeighRich[features$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0

table(features$NeighRich)

#features$TotalPorchSF <- features$OpenPorchSF + features$EnclosedPorch + features$X3SsnPorch + features$ScreenPorch

##Need a feature count variable...

##remove outliers
#cor(features$SalePrice[-c(524, 1299)], features$TotalSqFeet[-c(524, 1299)], use= "pairwise.complete.obs")
features <- features[-c(524, 1299),]
#features <- features %>% select(-SalePrice)


##fill in NAs
library(mice)
library(janitor)##dropping vars (multicollinear)
LogGrLivArea <- log10(features$GrLivArea)
features <- cbind(features, LogGrLivArea)

dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')
features <- features[,!(names(features) %in% dropVars)]

features <- clean_names(features)
#features <- manage_na(features) ##just the categorial vars
micemod <- features %>% mice(method='rf') ##what's left is handed to mice
features<- complete(micemod)
rm(micemod)


##Identifying Num Vars
numericVars <- which(sapply(features, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
cat('There are', length(numericVars), 'numeric variables')

features_numVar <- features[, numericVars]
#cor_numVar <- cor(features_numVar, use="pairwise.complete.obs") #correlations of features numeric variables
#cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))

#select only high corelations
#CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
#cor_numVar <- cor_numVar[CorHigh, CorHigh]
#corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")


#Age <- features$YrSold - features$YearRemodAdd
##remove NA columns count them
na_count <-sapply(features, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
features <- features[ , colSums(is.na(features)) == 0]
anyNA(features)

##Log Transform of Target
LogPrice <- log1p(train$SalePrice)

LogPrice <- LogPrice[-c(524, 1299)]
train_tbl <- cbind(features[1:1458,], LogPrice)
test_tbl <- features[(nrow(train)-1):(nrow(features)),]

#### main processing
# train_tbl <- train %>%
#   select (-Id) 
# test_tbl <- test %>%
#   select (-Id) 


rec <- recipe(LogPrice ~ ., data = train_tbl)

rec2 <- rec %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  #step_log (SalePrice) %>%
  #step_naomit(all_predictors(), -all_outcomes()) %>%
  step_knnimpute(all_predictors(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) 


rec_trained <- prep(rec2, training = train_tbl, retain = TRUE)
summary(rec_trained)
rec_trained_data=juice(rec_trained)
summary(rec_trained_data)

x_train_tbl <- bake(rec_trained, new_data = train_tbl) %>% select(-LogPrice)
x_test_tbl <- bake(rec_trained, new_data = test_tbl)  

y_train_vec <- pull (train_tbl, LogPrice) 
y_train_vec <- log1p(y_train_vec)


# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 40)


############## K-fold prep
build_model <- function() {
  model <- keras_model_sequential() %>%
    # (1) 1st Hidden Layer-------------------------------------------------
  layer_dense (units              = 32, #=> Num Of Nodes
               kernel_initializer = "uniform", 
               activation         = "relu",    
               input_shape        = ncol(x_train_tbl)) %>% 
    layer_dropout (rate = 0.2) %>%  #=> Dropout Below 10%: Prevent overfitting
    # (2) 2nd Hidden Layer-------------------------------------------------
  layer_dense (units              = 32,
               kernel_initializer = "uniform", 
               activation         = "relu") %>% 
    layer_dropout (rate = 0.3) %>%  
    # (2) 3nd Hidden Layer-------------------------------------------------
  layer_dense (units              = 32,
               kernel_initializer = "uniform", 
               activation         = "relu") %>% 
    # (3) Output Layer-----------------------------------------------------
  layer_dense (units              = 1) %>% #=> Common for Binary
    # (4) Compile Model-----------------------------------------------------
  compile (optimizer = 'RMSprop', #=> Most Popular for Optimization Algo.
           loss      = 'mean_squared_logarithmic_error', #=> Binary Classification
           metrics   = c('mae') ) #=> Train/Test Evaluation
}

k <- 4
indices <- sample(1:nrow(x_train_tbl))
folds <- cut(1:length(indices), breaks = k, labels = FALSE)

num_epochs <- 12
all_mae_histories <- NULL
for (i in 1:k) {
  cat ("processing fold number", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- as.matrix(x_train_tbl[val_indices,])
  val_targets <- y_train_vec[val_indices]
  
  partial_train_data <- as.matrix(x_train_tbl[-val_indices,])
  partial_train_targets <- y_train_vec[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(partial_train_data, partial_train_targets,
                           validation_data = list(val_data, val_targets),
                           epochs = num_epochs, batch_size = 4, verbose = 1)
  
  # results <- model %>% evaluate(val_data, val_targets, verbose = 1)
  # all_scores <- c(all_scores, results$mean_absolute_error)
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean))


ggplot(average_mae_history, aes(x = epoch, y= validation_mae)) + geom_line()

meanval <- average_mae_history$validation_mae[nrow(average_mae_history)]
meanval

model <- build_model()

model %>% fit(as.matrix(x_train_tbl), y_train_vec,
              epochs = 10, batch_size=4, verbose=1)
result <- model %>% evaluate(as.matrix(x_train_tbl), y_train_vec)

testPreds <- model %>% predict(as.matrix(x_test_tbl))
testPreds <- testPreds[ , 1]
#testPreds <- expm1(testPreds)



#---------------------------
cat("Making submission file...\n")
version <- "1"

read_csv("/mnt/Data/DataSet/HousePricesKaggle/sample_submission.csv") %>% 
  transmute('Id' = as.integer(Id),
            SalePriceTF1 = expm1(testPreds),
            SalePriceTF2 = expm1(testPreds))  %>%
  rowwise() %>%
  mutate(SalePrice= mean(c(SalePriceTF1, SalePriceTF2))) %>%
  select(Id, SalePrice) %>%
  write_csv(paste0("/mnt/Data/DataSet/HousePricesKaggle/v_",version,"_TF_perf_", round(meanval,5), ".csv"))


