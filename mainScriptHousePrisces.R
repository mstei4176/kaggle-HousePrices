#https://iamkbpark.com/2018/01/22/deep-learning-with-keras-lime-in-r/#preprocess-data
suppressPackageStartupMessages({ library (readxl)
  library (keras);library (lime);library (tidyquant)
  library (rsample);library (recipes);library (yardstick)
  library (corrr);library(knitr);library (DT);library(tidyverse)
  library(tidyselect); library(keras); library(varhandle)
})

train <- read_csv("~/Desktop/GitContent/kaggle-HousePrices/train-2.csv")
test <- read_csv("~/Desktop/GitContent/kaggle-HousePrices/test-2.csv")
set.seed(1337)


train_tbl <- train %>%
  select (-Id) 
test_tbl <- test %>%
  select (-Id) 

rec <- recipe(SalePrice ~ ., data = train_tbl)

rec2 <- rec %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>%
  #step_log (SalePrice) %>%
  step_knnimpute(all_predictors(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes())  %>%
  check_missing(all_predictors(), -all_outcomes()) 


rec_trained <- prep(rec2, training = train_tbl, retain = TRUE)
summary(rec_trained)
rec_trained_data=juice(rec_trained)
summary(rec_trained_data)

x_train_tbl <- bake(rec_trained, new_data = train_tbl) %>% select(-SalePrice)
x_test_tbl <- bake(rec_trained, new_data = test_tbl)  %>% select(-SalePrice)

y_train_vec <- pull (train_tbl, SalePrice) 
y_train_vec <- log1p(y_train_vec)

model_keras <- keras_model_sequential()

model_keras %>% 
  # (1) 1st Hidden Layer-------------------------------------------------
layer_dense (units              = 128, #=> Num Of Nodes
             kernel_initializer = "uniform", 
             activation         = "relu",    
             input_shape        = ncol(x_train_tbl)) %>% 
  layer_dropout (rate = 0.1) %>%  #=> Dropout Below 10%: Prevent overfitting
  # (2) 2nd Hidden Layer-------------------------------------------------
layer_dense (units              = 128,
             kernel_initializer = "uniform", 
             activation         = "relu") %>% 
  layer_dropout (rate = 0.1) %>%  
  # (3) Output Layer-----------------------------------------------------
layer_dense (units              = 1) %>% #=> Common for Binary
  # (4) Compile Model-----------------------------------------------------
compile (optimizer = 'RMSprop', #=> Most Popular for Optimization Algo.
         loss      = 'mean_squared_logarithmic_error', #=> Binary Classification
         metrics   = c('mae') ) #=> Train/Test Evaluation

# Check
model_keras

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 40)

epochs <- 400


system.time ( 
  history <- fit (
    object           = model_keras,             # => Our Model
    x                = as.matrix(x_train_tbl),  # => Matrix
    y                = y_train_vec,             # => Numeric Vector 
    batch_size       = 512,     #=> #OfSamples/gradient update in each epoch
    epochs           = epochs,     #=> Control Training cycles
    validation_split = 0.30,
    verbose = 0,
    callbacks = list(early_stop, print_dot_callback))
  ) #=> Include 30% data for 'Validation' Model

model_keras %>% evaluate(as.matrix(x_train_tbl), y_train_vec)

#c(loss, mae) %<-% (model_keras %>% evaluate(as.matrix (x_train_tbl), y_train_vec, verbose = 0))
#paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))


library(ggplot2)

plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

print(history)

testPreds <- model_keras %>% predict(as.matrix(x_test_tbl))
testPreds <- testPreds[ , 1]
#testPreds <- expm1(testPreds)

#---------------------------
cat("Making submission file...\n")
version <- "12"

#output <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test.csv") 
#my_submission <- data_frame('Id' = as.integer(output$Id), 'SalePrice' = exp(predictiondf$predict))
# library(readr)
# run1 <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test_preds (1).csv")
# run2 <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test_preds (2).csv")
# run3 <- read_csv("/Users/mstein/kaggle-HousePrices/hybrid_solution.csv")
# run4 <- read_csv("/Users/mstein/kaggle-HousePrices/House_price_submission_v57.csv")

#AutoPred <- as.data.frame(expm1(h2o.predict(autoMeta1, test.hex)))

read_csv("/mnt/Data/DataSet/HousePricesKaggle/sample_submission.csv") %>% 
  transmute('Id' = as.integer(Id),
            SalePriceTF1 = expm1(testPreds),
            SalePriceTF2 = expm1(testPreds))  %>%
  rowwise() %>%
  mutate(SalePrice= mean(c(SalePriceTF1, SalePriceTF2))) %>%
  select(Id, SalePrice) %>%
  write_csv(paste0("/mnt/Data/DataSet/HousePricesKaggle/v_",version,"_TF_perf_", round(min(history$metrics$val_mean_absolute_error),5), ".csv"))


