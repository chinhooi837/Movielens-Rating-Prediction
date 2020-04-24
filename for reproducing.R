
#install and load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")


edx<-read_csv('edx.csv',col_names = TRUE)
edx<-edx%>%select(-X1)
edx<-data.frame(edx)

edx1<-preprocess(edx)
edx1<-feature_eng_default(edx1)



#first default
set.seed(1)
library(caret)
set.seed(1)
trainIndex <- createDataPartition(edx1$rating, p = .8, 
                                  list = FALSE, 
                                  times = 1)

#head(trainIndex)
edx1<-data.frame(edx1)
test1Train <- edx1[trainIndex,]
test1Test  <- edx1[-trainIndex,]
y_train_first<-test1Train$rating
X_train_first<-test1Train%>%select(-rating,-timestamp,-title)
y_test_first<-test1Test$rating
X_test_first<-test1Test%>%select(-rating,-timestamp,-title)


#standardization
X_train_transformed_first<-transform(X_train_first)
#standardization
X_test_transformed_first<-transform(X_test_first)


#load model
xgb_first_100_default<- readRDS("xgb_first_100_default.rds")

RMSE_test_first_100_default<-RMSE(xgb_first_100_default,X_test_transformed_first,y_test_first)


edx2<-feature_eng_bias(edx1)

#head(trainIndex)
set.seed(1)
test1Train <- edx2[trainIndex,]
test1Test  <- edx2[-trainIndex,]
y_train<-test1Train$rating
X_train<-test1Train%>%select(-rating,-timestamp,-title)
y_test<-test1Test$rating
X_test<-test1Test%>%select(-rating,-timestamp,-title)

#standardization
X_train_transformed<-transform(X_train)
#standardization
X_test_transformed<-transform(X_test)


#load model
xgb_200_default<- readRDS("xgb_200_default.rds")




#load model
xgb_200_default_fselect<- readRDS("xgb_200_default_fselect.rds")
xgb_400_default_fselect<- readRDS("xgb_400_default_fselect.rds")
#load model
xgb_400_depth_child_fselect<- readRDS("xgb_400_depth_child_fselect.rds")
xgb_400_colsample_subsample_fselect<- readRDS("xgb_400_colsample_subsample_fselect.rds")
#load model
xgb_400_gamma_fselect<- readRDS("xgb_400_gamma_fselect.rds")                                              
xgb_400_gamma_6_fselect<- readRDS("xgb_400_gamma_6_fselect.rds")  

#feature selection
# estimate variable importance
importance <- varImp(xgb_200_default, scale=FALSE)
#select top 10 features
features_selected<- rownames(importance$importance)[1:10]
X_train_transformed_fselect<-X_train_transformed%>%select(features_selected)
X_test_transformed_fselect<-X_test_transformed%>%select(features_selected)

RMSE_test_200_default<-RMSE(xgb_200_default,X_test_transformed,y_test)
RMSE_test_400_default_fselect<-RMSE(xgb_400_default_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_depth_child_fselect<-RMSE(xgb_400_depth_child_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_colsample_subsample_fselect<-RMSE(xgb_400_colsample_subsample_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_gamma_fselect<-RMSE(xgb_400_gamma_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_gamma_6_fselect<-RMSE(xgb_400_gamma_6_fselect,X_test_transformed_fselect,y_test)


validation<-read_csv('validation.csv',col_names = TRUE)
validation<-validation%>%select(-X1)
validation1<-preprocess(validation)
validation1<-feature_eng_default(validation1)
validation1<-feature_eng_bias(validation1)

X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1%>%select(rating)

X_test_val_transformed<-transform(X_test_val)

# preprocessParams_X_test_val <- preProcess(X_test_val, method=c("center", "scale"))
# X_test_val_transformed <- predict(preprocessParams_X_test_val, X_test_val)

X_test_val_transformed_fselect<-X_test_val_transformed%>%select(features_selected)


RMSE_val_200_default<-RMSE_val(xgb_200_default,X_test_val_transformed,y_test_val)
RMSE_val_400_default_fselect<-RMSE_val(xgb_400_default_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_depth_child_subsample_fselect<-RMSE_val(xgb_400_depth_child_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_colsample_subsample_fselect<-RMSE_val(xgb_400_colsample_subsample_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_gamma_fselect<-RMSE_val(xgb_400_gamma_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_gamma_6_fselect<-RMSE_val(xgb_400_gamma_6_fselect,X_test_val_transformed_fselect,y_test_val)