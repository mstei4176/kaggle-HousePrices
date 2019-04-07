##########################################################################
##                               Defaults                               ##
##########################################################################
options(scipen=999)
options(encoding = "UTF-8")
Sys.setenv(TZ='UTC')
##########################################################################
##                               Loading H2O                            ##
##########################################################################
setwd("/mnt/data/DataSet/sandbock")
source("spinup.R")
set.seed(1337)
##########################################################################
##                               Setting WD                             ##
##########################################################################
thisPath <- "/Users/mstein/kaggle-HousePrices"
setwd(thisPath)
library(tidyverse)
library(readr)
##########################################################################
##                               Local Data                             ##
##########################################################################
##https://www.kaggle.com/mariopasquato/fun-with-h2o#L4
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

#Read the train/test file
tr <- read.csv("train.csv")
te <- read.csv("test.csv")

#features <- bind_rows(tr, te)
#Put the features in te and tr together and remove NAs
features <- rbind(tr[,-c(1, ncol(tr))], te[, -1])
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

features$TotalPorchSF <- features$OpenPorchSF + features$EnclosedPorch + features$X3SsnPorch + features$ScreenPorch

##Need a feature count variable...

##remove outliers
#cor(features$SalePrice[-c(524, 1299)], features$TotalSqFeet[-c(524, 1299)], use= "pairwise.complete.obs")
features <- features[-c(524, 1299),]
#features <- features %>% select(-SalePrice)


##fill in NAs
features <- manage_na(features) ##just the categorial vars
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
LogGrLivArea <- log10(features$GrLivArea)
features <- cbind(features, LogGrLivArea)

##dropping vars (multicollinear)
dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')
features <- features[,!(names(features) %in% dropVars)]

## Doublecheck for NAs
anyNA(features)

##log transformation of Price
LogPrice <- log1p(tr$SalePrice)
anyNA(LogPrice)

##preprocessing
numericVarNames <- names(features[,sapply(features, is.numeric)])

DFnumeric <- features[, names(features) %in% numericVarNames]
factorNames <- names(features[,sapply(features, base::is.factor)])
DFfactors <- features[, !(names(features) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

sapply(DFfactors, nlevels)


###skewness
library(psych)
manage_skew <- function(datafra)
{
  for(i in 1:ncol(datafra))
  {
    u <- datafra[,i]
    if(is.numeric(u))
    {
      if (abs(skew(datafra[,i]))>0.8){
        datafra[,i] <- log(datafra[,i] +1)
      } 
    }
  }
  datafra
}
library(caret)
manage_skew(features)

PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)

DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)


##one-hotsi
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

#check if some values are absent in the test set
#ZerocolTest <- which(colSums(DFdummies[(nrow(features[!is.na(tr$SalePrice),])+1):nrow(features),])==0)
#colnames(DFdummies[ZerocolTest])

#DFdummies <- DFdummies[,-ZerocolTest] #removing predictors

#check if some values are absent in the train set
#ZerocolTrain <- which(colSums(DFdummies[1:nrow(features[!is.na(features$SalePrice),]),])==0)
#colnames(DFdummies[ZerocolTrain])

## [1] "MSSubClass1,5 story PUD features"
#DFdummies <- DFdummies[,-ZerocolTrain] #removing predictor

#fewOnes <- which(colSums(DFdummies[1:nrow(features[!is.na(features$SalePrice),]),])<10)
#colnames(DFdummies[fewOnes])

#DFdummies <- DFdummies[,-fewOnes] #removing predictors
dim(DFdummies)

combined <- cbind(DFnorm, DFdummies) #combining features (now numeric) predictors into one dataframe 

LogPrice <- LogPrice[-c(524, 1299)]


#Building  sets with Dummies
dtrain <- cbind(combined[1:1458,], LogPrice)
dtest <- combined[(nrow(tr)-1):(nrow(combined)),]
write.csv(dtrain, "dummy_train.csv", row.names = F)
write.csv(dtest, "dummy_test.csv", row.names = F)

#Building  sets without Dummies
train <- cbind(features[1:1458,], LogPrice)
test <- features[(nrow(tr)-1):(nrow(features)),]
write.csv(train, "nodummy_train.csv", row.names = F)
write.csv(test, "nodummy_test.csv", row.names = F)

#######################################################################################################
#######################################################################################################
#######################################################################################################

# library(h2o)
# h2o_context(sc)


training.hex <- h2o.importFile(path = "/mnt/data/DataSet/HousePricesKaggle/nodummy_train.csv", destination_frame = "prediction.hex")
test.hex <- h2o.importFile(path = "/mnt/data/DataSet/HousePricesKaggle/nodummy_test.csv", destination_frame = "test.hex")

# tr_tbl <- spark_read_csv(sc, "tr","nodummy_train.csv")
# te_tbl <- spark_read_csv(sc, "te","nodummy_test.csv")
# # # train <- read_csv("nodummy_train.csv")
# # # test <- read_csv("nodummy_test.csv")
# # 
# prediction.hex <- as.h2o(te_tbl, destination_frame = "prediction.hex")
# training.hex <- as.h2o(tr_tbl, destination_frame = "training.hex")
# 
# prediction.hex <- as_h2o_frame(sc, tr_tbl)
# test.hex <- as_h2o_frame(sc, te_tbl)


# Convert target to log
#training.train$target <- h2o.log1p(training.train$SalePrice)
#training.test$target <- h2o.log1p(training.test$SalePrice)

# Identify predictors and response
y <- "LogPrice"
x <- setdiff(names(training.hex), c("target","SalePrice"))

training.split <- h2o.splitFrame(data=training.hex, ratios=0.80)
training.train <- training.split[[1]]
training.blend <- training.split[[2]]
nfolds <- 6

amlNEWt1 <- h2o.automl(x = x, y = y,
                       training_frame = training.train,
                       blending_frame = training.blend,
                       balance_classes = FALSE,
                       nfolds = nfolds,
                       include_algos = c("GLM","XGBoost","StackedEnsemble","GBM","DeepLearning","DRF"),
                       keep_cross_validation_predictions = TRUE,
                       max_runtime_secs = 3600,
                       #max_runtime_secs_per_model = 900,
                       stopping_metric = "RMSLE",
                       stopping_rounds = 3,
                       export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                       seed = 123)

# View the AutoML Leaderboard
lb <- amlNEWt1@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)

#######################################################################################################
# Create blended stacked Ensemble
#######################################################################################################
nfolds <- 6

san_xgb1 <- h2o.xgboost(x = x,
                        y = y,
                        training_frame = training.train,
                        validation_frame = training.blend,
                        backend = "auto",
                        model_id = "san_xgb1",
                        #distribution = "gaussian",
                        categorical_encoding = "LabelEncoder", #best
                        tree_method = "exact", #best
                        ntrees = 4300,
                        learn_rate = 0.005,
                        min_rows = 4, #best
                        max_depth = 4, #best
                        normalize_type = "forest", #best
                        grow_policy = "depthwise",
                        booster = "gbtree", #best
                        nfolds = nfolds,
                        stopping_metric = "RMSLE",
                        stopping_tolerance = 0.001,
                        fold_assignment = "Modulo",
                        export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                        keep_cross_validation_predictions = TRUE,
                        seed = 1)
h2o.rmsle(h2o.performance(san_xgb1, training.blend))
# [1] 0.01002807

san_glm1 <- h2o.glm(    x = x,
                        y = y,
                        training_frame = training.train,
                        validation_frame = training.blend,
                        model_id = "san_glm1",
                        family = "gaussian",
                        solver = "L_BFGS", #BEST
                        early_stopping = TRUE,
                        seed = 1,
                        fold_assignment = "Modulo",
                        export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                        keep_cross_validation_predictions = TRUE,
                        nfolds = nfolds)

h2o.rmsle(h2o.performance(san_glm1, training.blend))
#[1] 0.009325758

## Lasso Regression alpha =1
san_glm2 <- h2o.glm(    x = x,
                       y = y,
                       training_frame = training.train,
                       validation_frame = training.blend,
                       model_id = "san_glm2",
                       family = "gaussian",
                       solver = "IRLSM", #BEST
                       alpha = 1,
                       early_stopping = TRUE,
                       lambda_search = TRUE,
                       seed = 1,
                       fold_assignment = "Modulo",
                       export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                       keep_cross_validation_predictions = TRUE,
                       nfolds = nfolds)

h2o.rmsle(h2o.performance(san_glm2, training.blend))
#[1] 0.009780807

## Ridge Regression alpha =0
san_glm3 <- h2o.glm(    x = x,
                        y = y,
                        training_frame = training.train,
                        validation_frame = training.blend,
                        model_id = "san_glm3",
                        family = "gaussian",
                        solver = "IRLSM", #BEST
                        lambda_search = TRUE,
                        alpha = 0,
                        early_stopping = TRUE,
                        seed = 1,
                        fold_assignment = "Modulo",
                        export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                        keep_cross_validation_predictions = TRUE,
                        nfolds = nfolds)

h2o.rmsle(h2o.performance(san_glm3, training.blend))
#[1] 0.009277972

hyper_params <- list(learn_rate = 0.01,
                     ntrees = 1000,
                     stopping_metric = "RMSLE",
                     stopping_tolerance = 0.001)

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                base_models = list(san_glm1, san_glm2, san_glm3),
                                training_frame = training.train,
                                blending_frame = training.blend,
                                model_id = "HP_ensemble",
                                metalearner_algorithm = "gbm",
                                export_checkpoints_dir = "/Users/mstein/kaggle-HousePrices/models",
                                keep_levelone_frame = TRUE,
                                #metalearner_params = hyper_params,
                                metalearner_nfolds = 6, #was 10
                                seed = 1)

h2o.rmsle(h2o.performance(ensemble, training.blend))
# [1] 0.007496884
perf <- h2o.performance(ensemble, newdata = training.blend)

#######################################################################################################
# Combine
#######################################################################################################

# Compare to base learner performance on the test set
perf_xgb1_test <- h2o.performance(san_xgb1, newdata = training.blend)
# perf_xgb2_test <- h2o.performance(san_xgb2, newdata = training.blend)
# perf_nn1_test <- h2o.performance(san_nn1, newdata = training.blend)
# perf_xbm1_test <- h2o.performance(san_xbm1, newdata = training.blend)
perf_glm1_test <- h2o.performance(san_glm1, newdata = training.blend)
perf_glm2_test <- h2o.performance(san_glm2, newdata = training.blend)
perf_glm3_test <- h2o.performance(san_glm3, newdata = training.blend)
# baselearner_best_RMSLE_test <- min(h2o.rmsle(perf_xgb1_test), h2o.rmsle(perf_xgb2_test), h2o.rmsle(perf_nn1_test),
#                                    h2o.rmsle(perf_glm1_test), h2o.rmsle(perf_xbm1_test), h2o.rmsle(perf_glm2_test))
baselearner_best_RMSLE_test <- min(h2o.rmsle(perf_glm1_test), h2o.rmsle(perf_glm2_test), h2o.rmsle(perf_glm3_test), h2o.rmsle(perf_xgb1_test))

ensemble_rmsle_test <- h2o.rmsle(perf)
print(sprintf("Best Base-learner Test RMSLE:  %s", baselearner_best_RMSLE_test))
print(sprintf("Ensemble Test RMSLE:  %s", ensemble_rmsle_test))

# Generate predictions on a test set (if neccessary)
#prediction.hex <- as.h2o(te, destination_frame = "test.hex")

pred <- h2o.predict(ensemble, newdata = test.hex)
predML <- h2o.predict(amlNEWt1, newdata = test.hex)
predictiondf <- as.data.frame(pred)
predictiondfML <- as.data.frame(predML)

version <- "113b"

#---------------------------
cat("Making submission file...\n")
#output <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test.csv") 
#my_submission <- data_frame('Id' = as.integer(output$Id), 'SalePrice' = exp(predictiondf$predict))
library(readr)
run1 <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test_preds (1).csv")
run2 <- read_csv("/mnt/Data/DataSet/HousePricesKaggle/test_preds (2).csv")
run3 <- read_csv("/Users/mstein/kaggle-HousePrices/hybrid_solution.csv")
run4 <- read_csv("/Users/mstein/kaggle-HousePrices/House_price_submission_v57.csv")

#AutoPred <- as.data.frame(expm1(h2o.predict(autoMeta1, test.hex)))

read_csv("/mnt/Data/DataSet/HousePricesKaggle/sample_submission.csv") %>% 
  transmute('Id' = as.integer(Id),
            SalePriceAML = expm1(predictiondfML$predict),
            SalePriceR1 = run1$SalePrice,
            SalePriceR2 = run2$SalePrice,
            SalePriceR3 = run3$SalePrice,
            SalePriceR4 = run4$SalePrice,
            SalePriceEns= expm1(predictiondf$predict))  %>%
  rowwise() %>%
  mutate(SalePrice= mean(c(SalePriceAML, SalePriceR1, SalePriceR2, SalePriceR3, SalePriceR4, SalePriceR4))) %>%
  select(Id, SalePrice) %>%
  write_csv(paste0("/mnt/Data/DataSet/HousePricesKaggle/v_",version,"_base_SVM6_perf_", round(ensemble_rmsle_test,5), ".csv"))

#0.11179 
