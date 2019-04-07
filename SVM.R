#######################################################################################################
# Create SVM Model
#######################################################################################################
library(e1071)

train <- read_csv("/mnt/data/DataSet/HousePricesKaggle/nodummy_train.csv")
test <- read_csv("/mnt/data/DataSet/HousePricesKaggle/nodummy_test.csv")

# train <- as.data.frame(training.hex)
# test <- as.data.frame(test.hex)
test$LogPrice <- NA

merged <- bind_rows(train, test)
merged <- varhandle::unfactor(merged)

##https://stackoverflow.com/questions/44200195/how-to-debug-contrasts-can-be-applied-only-to-factors-with-2-or-more-levels-er/44201384
## get mode of all vars
var_mode <- sapply(merged, mode)

## produce error if complex or raw is found
if (any(var_mode %in% c("complex", "raw"))) stop("complex or raw not allowed!")

## get class of all vars
var_class <- sapply(merged, class)

## produce error if an "AsIs" object has "logical" or "character" mode
if (any(var_mode[var_class == "AsIs"] %in% c("logical", "character"))) {
  stop("matrix variables with 'AsIs' class must be 'numeric'")
}

## identify columns that needs be coerced to factors
ind1 <- which(var_mode %in% c("logical", "character"))

## coerce logical / character to factor with `as.factor`
merged[ind1] <- lapply(merged[ind1], as.factor)

test <- filter(merged, is.na(LogPrice))
test$LogPrice <- NULL

train <- filter(merged, !is.na(LogPrice))

#training.hex = h2o.getFrame(h2o = localH2O, key = "prostate.hex")
svm_model<-svm(LogPrice~., data=train, cost = 3)
svm_pred <- as.data.frame(expm1(predict(svm_model,newdata = test)))
colnames(svm_pred) <- "pred"

svm_model2<-svm(LogPrice~., data=train,
                epsilon = 0.3,
                cost = 3)
svm_pred2 <- as.data.frame(expm1(predict(svm_model2,newdata = test)))
colnames(svm_pred2) <- "pred"