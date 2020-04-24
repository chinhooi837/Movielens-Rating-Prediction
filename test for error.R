validation2<-preprocess(validation)
validation2<-feature_eng_default(validation2)  
validation2<-feature_eng_bias(validation2)  
edx1<-feature_eng_default(edx1)
edx1<-feature_eng_bias(edx1)

edx3<-preprocess(edx)
edx4<-preprocess(edx)

edx5<-feature_eng_default(edx3)
edx6<-feature_eng_default(edx4)

edx7<-feature_eng_bias(edx5)
edx8<-feature_eng_bias(edx6)

edx9<-feature_eng_org(edx3)

validation5<-feature_eng_default(validation3)  
validation6<-feature_eng_default(validation4)  
validation7<-feature_eng_bias(validation5)  
validation8<-feature_eng_bias(validation6)  


validation2$userId<-as.numeric(validation2$userId)
all.equal(edx7,edx9)


# Testing with validation set2
X_test_val2<-validation2%>%select(-rating,-timestamp,-title)
y_test_val2<-validation2%>%select(rating)
X_test_val2_transformed<-transform(X_test_val2)
X_test_val2transformed_fselect<-X_test_val2_transformed%>%select(features_selected)

test_val2_predict_400_colsample_subsample_fselect <- predict(xgb_400_colsample_subsample_fselect, X_test_val2transformed_fselect)
residuals_val2_400_colsample_subsample_fselect <- y_test_val2 - test_val2_predict_400_colsample_subsample_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val2_400_colsample_subsample_fselect <- sqrt(mean(residuals_val2_400_colsample_subsample_fselect$rating^2))

test_val2_predict_400_gamma_3 <- predict(xgb_400_gamma_3, X_test_val2transformed_fselect)
residuals_val2_400_gamma_3 <- y_test_val2 - test_val2_predict_400_gamma_3
#residuals$rating because the column name that stores the values is "rating"
RMSE_val2_400_gamma_3 <- sqrt(mean(residuals_val2_400_gamma_3$rating^2))


# Testing with validation set
X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1%>%select(rating)
X_test_val_transformed<-transform(X_test_val)
RMSE_val_first_100_default<-RMSE_val(xgb_first_100_default,X_test_val_transformed,y_test_val)



validation1<-feature_eng_bias(validation1)  
X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1%>%select(rating)
X_test_val_transformed<-transform(X_test_val)

X_test_val_transformed_fselect<-X_test_val_transformed%>%select(features_selected)

RMSE_val_200_default<-RMSE_val(xgb_200_default,X_test_val_transformed,y_test_val)
RMSE_val_400_default_fselect<-RMSE_val(xgb_400_default_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_depth_child_subsample_fselect<-RMSE_val(xgb_400_depth_child_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_colsample_subsample_fselect<-RMSE_val(xgb_400_colsample_subsample_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_gamma_fselect<-RMSE_val(xgb_400_gamma_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_val_400_gamma_6_fselect<-RMSE_val(xgb_400_gamma_6_fselect,X_test_val_transformed_fselect,y_test_val)



RMSE_test_400_default_fselect<-RMSE(xgb_400_default_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_depth_child_fselect<-RMSE(xgb_400_depth_child_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_colsample_subsample_fselect<-RMSE(xgb_400_colsample_subsample_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_gamma_fselect<-RMSE(xgb_400_gamma_fselect,X_test_transformed_fselect,y_test)
RMSE_test_400_gamma_6_fselect<-RMSE(xgb_400_gamma_6_fselect,X_test_transformed_fselect,y_test)




RMSE_val<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals$rating^2))
  return(RMSE)
}

test_val_predict_400_gamma_3 <- predict(xgb_400_gamma_3, X_test_val_transformed_fselect)
residuals_val_400_gamma_3 <- y_test_val - test_val_predict_400_gamma_3
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_gamma_3 <- sqrt(mean(residuals_val_400_gamma_3$rating^2))

all.equal(validation1,validation2)

#modeling
set.seed(1)
trainIndex <- createDataPartition(edx1$rating, p = .8, 
                                  list = FALSE, 
                                  times = 1)

#head(trainIndex)
edx1<-data.frame(edx1)



test1Train <- edx1[trainIndex,]
test1Test  <- edx1[-trainIndex,]
y_train<-test1Train$rating
X_train<-test1Train%>%select(-rating,-timestamp,-title)
y_test<-test1Test%>%select(rating)
X_test<-test1Test%>%select(-rating,-timestamp,-title)

X_train_transformed<-transform(X_train)
X_test_transformed<-transform(X_test)


importance <- varImp(xgb_200_default, scale=FALSE)
#select top 10 features
features_selected<- rownames(importance$importance)[1:10]
X_train_transformed_fselect<-X_train_transformed%>%select(features_selected)
X_test_transformed_fselect<-X_test_transformed%>%select(features_selected)






#Next, we should tune max_depth, min_child_weight
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = 3,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_gamma_3 <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                     metric="RMSE",
                                     trControl=trctrl,
                                     tuneGrid = tune_grid)


xgb_400_gamma_3

#save model
saveRDS(xgb_400_gamma_3, "xgb_400_new_gamma_3.rds")
#load model
rf_fit<- readRDS("model.rds")


test_predict_400_gamma_3 <- predict(xgb_400_gamma_3, X_test_transformed_fselect)
residuals_400_gamma_3 <- y_test - test_predict_400_gamma_3
# #residuals$rating because the column name that stores the values is "rating"
RMSE_test_400_gamma_3 <- sqrt(mean(residuals_400_gamma_3$rating^2))

test_val_predict_400_gamma_3 <- predict(xgb_400_gamma_3, X_test_val_transformed_fselect)
residuals_val_400_gamma_3 <- y_test_val - test_val_predict_400_gamma_3
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_gamma_3 <- sqrt(mean(residuals_val_400_gamma_3$rating^2))


xgb_400_colsample_subsample_fselect

test_val_predict_400_colsample_subsample_fselect <- predict(xgb_400_colsample_subsample_fselect, X_test_val_transformed_fselect)
residuals_val_400_colsample_subsample_fselect <- y_test_val - test_val_predict_400_colsample_subsample_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_colsample_subsample_fselect <- sqrt(mean(residuals_val_400_colsample_subsample_fselect$rating^2))

#test with 400 gamma 6
test_val_predict_400_gamma_6_fselect <- predict(xgb_400_gamma_6_fselect, X_test_val_transformed_fselect)
residuals_val_400_gamma_6_fselect <- y_test_val - test_val_predict_400_gamma_6_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_gamma_6_fselect <- sqrt(mean(residuals_val_400_gamma_6_fselect$rating^2))