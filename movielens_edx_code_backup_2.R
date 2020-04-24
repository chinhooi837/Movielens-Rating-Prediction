################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")



# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

library(caret)
library(stringr)
library(tidyverse)



ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

library(readr)
# write.csv(edx,'/Users/User/Documents/R/data/edx.csv')
# write.csv(edx1,'edx1.csv')
# write.csv(validation,'/Users/User/Documents/R/data/validation.csv')
# write.csv(validation1,'validation1.csv')


#to reload data
 validation<-read_csv('validation.csv',col_names = TRUE)
# 
# edx1<-read_csv('edx1.csv',col_names = TRUE)
# validation1<-read_csv('validation1.csv',col_names = TRUE)
 validation<-validation%>%select(-X1)
# validation1<-validation1%>%select(-X1)
# 
# edx1<-edx1%>%select(-X1)
# 
 edx<-read_csv('edx.csv',col_names = TRUE)
 edx<-edx%>%select(-X1)
 edx<-data.frame(edx)
#for easier recovery


#Quiz
# nrow(edx%>%filter(rating==3))
# nrow(edx%>%select(movieId)%>%unique())
# nrow(edx%>%select(userId)%>%unique())
# nrow(edx%>%select(rating,genres)%>%unique()%>%filter(grepl("Drama",genres)))
# nrow(edx%>%select(rating,genres)%>%filter(grepl("Thriller",genres)))
# test<-edx%>%group_by(title)%>%summarise(count=n())%>%arrange(desc(count))
# 
# test<-edx %>%select(genres)%>%filter(grepl("Sci-Fi",genres))
# test2<-test%>%str_replace_all("Sci-Fi","SciFi")
#   
# test<-edx%>% separate(genres, c("g1", "g2","g3","g4","g5","g6"),sep="[^|]")



#df1<-edx%>%head(1000)
#preprocess function 
preprocess<-function(df){
  #remove '-' from Sci-Fi, Film-Noir
  df1<-df
  df1$genres<-gsub('-','',df1$genres)
  #replace no genres listed with NA if it exists
  if(nrow(df1%>%select(genres)%>%filter(genres=="(no genres listed)"))!=0){
         df1 <- df1 %>% na_if("(no genres listed)")
         #df1<-df1 %>% replace_with_na_all(condition = ~.genres == "(no genres listed)")
  }
  #seperate genres column by "|“
  df1<-df1%>% separate(genres, c("g1","g2","g3","g4","g5","g6","g7","g8"))
  
  #generate unique values of genres, checked that g1 contains all unique values for genres
  unique<-df1$g1%>%unique()
  l2<-list(unique)
  l2[[1]]<-sort(l2[[1]],decreasing = FALSE)
  
  
  #generate extra columns for all unique values, 1 if genres contain it, 0 if not
  for (i in l2[[1]]){
    df1<-df1%>%mutate(!!i:=ifelse(apply(df1 == !!i, 1, any), 1, 0))
  }
  
  #remove intermediate columns
  df1<-df1%>%select(-c(g1,g2,g3,g4,g5,g6,g7,g8))
  
  return (df1)
}

  
feature_eng_default<-function(df){
  df1<-df
  #remove NA column to prepare for modeling
  if("NA"%in%colnames(df1)){
    df1<-df1%>%select(-"NA")
  }
  #feature engineering
  #impute NA with 0 for processed genres
  #checkNA with edx1%>%select(COLUMN_NAME)%>%filter(is.na(COLUMN_NAME))
  #only genres has NA, impute with 0
  df1[is.na(df1)]<-0
  
  return (df1)
}  

edx1<-preprocess(edx)
# write.csv(edx1,'edx1_default.csv')
# 
 edx1<-read_csv('edx1_default.csv',col_names = TRUE)
edx1<-edx1%>%select(-X1)
# edx2<-edx1
edx1<-feature_eng_default(edx1)
 edx1<-feature_eng_bias(edx1)

validation1<-preprocess(validation)
validation2<-validation1
validation1<-feature_eng_default(validation1)
validation1<-feature_eng_bias(validation1)
























#modeling - first attempt
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
y_test<-test1Test$rating
X_test<-test1Test%>%select(-rating,-timestamp,-title)

#standardization
transform<-function(X){
  preprocessParams_X <- preProcess(X, method=c("center", "scale"))
  X <- predict(preprocessParams_X, X)
  return(X)
}

X_train_transformed_default<-transform(X_train)


#training first model to obtrain a baseline
tune_grid <- expand.grid(nrounds = 100,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_first_100_default <- train(X_train_transformed_default_2,y_train, method = "xgbTree",
                         metric="RMSE",
                         trControl=trctrl,
                         tuneGrid = tune_grid)

xgb_first_100_default

RMSE<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals^2))
  return(RMSE)
}

X_test_transformed<-transform(X_test)
RMSE_test_first_100_default<-RMSE(xgb_first_100_default,X_test_transformed,y_test)
#hence, baseline RMSE is 1.023, need further improvement




#save model
saveRDS(xgb_first_100_default, "xgb_first_100_default.rds")





#Adding new features to accomodate for user bias and movie bias 
#including BaselineRating,UserBias,MovieBias,daySinceFirstUserRating,daySinceFirstMovieRating
feature_eng_bias<-function(df){
  df1<-df
  df1<-df1%>%mutate(timestamp=as_datetime(timestamp,origin = lubridate::origin, tz = "UTC"))
  
  #prepare columns for new features- UserAverage, Movie Average, FirstUserRatingDate,FirstMovieRatingDate
  User<-data.frame(df1%>%
                     group_by(userId)%>%summarise(FirstUserRatingDate=min(timestamp),
                                                  UserAverage=mean(rating))) 
  
  MovieAverage<-data.frame(df1%>%
                             group_by(movieId)%>%summarise(FirstMovieRatingDate=min(timestamp),
                                                           MovieAverage=mean(rating)))
  #join to main data frame and produce the intended features, BaselineRating,UserBias,MovieBias,daySinceFirstUserRating,daySinceFirstMovieRating
  df1<-df1%>%mutate()%>%
    inner_join(User,by="userId")%>%
    inner_join(MovieAverage,by="movieId")%>%
    mutate(BaselineRating=mean(rating),
           UserBias=UserAverage-BaselineRating,
           MovieBias=MovieAverage-BaselineRating,
           daySinceFirstUserRating=as.integer(difftime(timestamp,FirstUserRatingDate,units = "days")),
           daySinceFirstMovieRating=as.integer(difftime(timestamp,FirstMovieRatingDate,units = "days")))%>%
    select(-FirstUserRatingDate,-FirstMovieRatingDate)
  return (df1)
}  



#main, preprocess and do feature engineering for both edx and validation file  
edx1<-feature_eng_bias(edx1)
#write.csv(edx1,'edx1_default.csv')
# edx2<-read_csv('edx1.csv',col_names = TRUE)
# edx2<-edx2%>%select(-X1)





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
y_test<-test1Test$rating
X_test<-test1Test%>%select(-rating,-timestamp,-title)


#standardization
X_train_transformed<-transform(X_train)

# 
# # building baseline models with default parameters, find out the optimal nrounds
# Mat1 = data.matrix(X_train_transformed)
# params <- list(nrounds = 100,
#                max_depth = 6,
#                eta = 0.3,
#                gamma = 0,
#                colsample_bytree = 1,
#                min_child_weight = 1,
#                subsample = 1)
# xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 100, nfold = 5, 
#                  showsd = T, stratified = T, print_every_n = 10, early_stop_round = 10, maximize = F)
# #stops at 100 rounds. train-rmse:0.863215+0.000185	test-rmse:0.864534+0.000451 
# #default 100 rounds doesn't seems to be enough, adding nrounds to 200

#build model at 200 rounds, putting early stopping at 5 instead
Mat1 = data.matrix(X_train_transformed_first)
params <- list(nrounds = 200,
               max_depth = 6,
               eta = 0.3,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)
xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 200, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 5, maximize = F)
#stops at 200 rounds train-rmse:0.860372+0.000164	test-rmse:0.862751+0.000573 
#new rounds [200]	train-rmse:0.860394+0.000178	test-rmse:0.862759+0.000613 



# xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist = list(val=dtest,train=dtrain), 
#                    print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")


#actual train with default model to test with test set
tune_grid <- expand.grid(nrounds = 200,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_200_default <- train(X_train_transformed,y_train, method = "xgbTree",
                metric="RMSE",
                trControl=trctrl,
                tuneGrid = tune_grid)
#tuneLength = 10


# have a look at the model 
xgb_200_default

#test set with edx1 test set
X_test_transformed<-transform(X_test)
RMSE_test_200_default<-RMSE(xgb_200_default,X_test_transformed,y_test)


# 
# test_predict_200_default <- predict(xgb_200_default, X_test_transformed)
# residuals_200_default <- y_test - test_predict_200_default
# # #residuals$rating because the column name that stores the values is "rating"
# RMSE_test_200_default <- sqrt(mean(residuals_200_default^2))
# #RMSE_test_200_default=0.86428, not bad number as compared to cv_test-rmse:0.862751+0.000573 

#save model
saveRDS(xgb_200_default, "xgb_200_default.rds")

#load model
xgb_200_default<- readRDS("xgb_200_default.rds")


#feature selection
# estimate variable importance
importance <- varImp(xgb_200_default, scale=FALSE)
#select top 10 features
features_selected<- rownames(importance$importance)[1:10]
X_train_transformed_fselect<-X_train_transformed%>%select(features_selected)
X_test_transformed_fselect<-X_test_transformed%>%select(features_selected)

#retest with 200 and default
# params <- list(nrounds = 200,
#                max_depth = 6,
#                eta = 0.3,
#                gamma = 0,
#                colsample_bytree = 1,
#                min_child_weight = 1,
#                subsample = 1)
# Mat1<-data.matrix(X_train_transformed_fselect)
# xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 200, nfold = 5, 
#                  showsd = T, stratified = T, print_every_n = 10, early_stop_round = 5, maximize = F)




#actual train with default model with only top 10 featureds extracted out to test with test set
tune_grid <- expand.grid(nrounds = 200,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_200_default_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                         metric="RMSE",
                         trControl=trctrl,
                         tuneGrid = tune_grid)

xgb_200_default_fselect

#test set with edx1 test set

RMSE_test_200_default_fselect<-RMSE(xgb_200_default_fselect,X_test_transformed_fselect)

preprocessParams_X_test <- preProcess(X_test, method=c("center", "scale"))
X_test_transformed <- predict(preprocessParams_X_test, X_test)

test_predict_200_default_fselect <- predict(xgb_200_default_fselect, X_test_transformed_fselect)
residuals_200_default_fselect <- y_test - test_predict_200_default_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_200_default_fselect <- sqrt(mean(residuals_200_default_fselect^2))
#RMSE_test_200_default_fselect=0.86385, not bad number as compared to cv_test-rmse:0.863067+0.000484 


#save model
saveRDS(xgb_200_default_fselect, "xgb_200_default_fselect.rds")
#load model
xgb_200_default_fselect<- readRDS("xgb_200_default_fselect.rds")



#retest with 400 and start to tune other parameters - eta
params <- list(nrounds = 400,
               max_depth = 6,
               eta = 0.1,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)
Mat1<-data.matrix(X_train_transformed_fselect)
xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 400, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 5, maximize = F)
#train-rmse:0.862446+0.000089	test-rmse:0.863981+0.000433 
#worse than baseline model of 0.8638, and much computational heavy for nrounds =400 and eta=0.1
#hence, eta and nrounds should be fixed at 0.3 and 200 respectively


#retest with 400 and default
params <- list(nrounds = 400,
               max_depth = 6,
               eta = 0.3,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)
Mat1<-data.matrix(X_train_transformed_fselect)
xgbcv_eta0.3 <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 400, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 5, maximize = F)
#train-rmse:0.857151+0.000094	test-rmse:0.861184+0.000392 



#actual train with default model with only top 10 featureds extracted out to test with test set
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_default_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                 metric="RMSE",
                                 trControl=trctrl,
                                 tuneGrid = tune_grid)

xgb_400_default_fselect
#load model
xgb_400_default_fselect<- readRDS("xgb_400_default_fselect.rds")


#test set with edx1 test set
test_predict_400_default_fselect <- predict(xgb_400_default_fselect, X_test_transformed_fselect)
residuals_400_default_fselect <- y_test - test_predict_400_default_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_400_default_fselect <- sqrt(mean(residuals_400_default_fselect^2))
#RMSE_test_400_default_fselect=0.8619731, not bad number as compared to cv_test-rmse:0.861184+0.000392
#nrounds =400 and eta = 0.3 has the best performance so far, should not further increase the nrounds to more than 400 due to the long computational time

#save model
saveRDS(xgb_400_default_fselect, "xgb_400_default_fselect.rds")

#Next, we should tune max_depth, min_child_weight
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = c(4,6,8),
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = c(1,3,5),
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_depth_child_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                 metric="RMSE",
                                 trControl=trctrl,
                                 tuneGrid = tune_grid)


xgb_400_depth_child_fselect


#load model
xgb_400_depth_child_fselect<- readRDS("xgb_400_depth_child_fselect.rds")




#test set with edx1 test set
test_predict_400_depth_child_fselect <- predict(xgb_400_depth_child_fselect, X_test_transformed_fselect)
residuals_400_depth_child_fselect <- y_test - test_predict_400_depth_child_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_400_depth_child_fselect <- sqrt(mean(residuals_400_depth_child_fselect^2))
#RMSE_test_400_depth_child_fselect=0.8580418, not bad number as compared to cv-test-rmse:0.8572837
#nrounds = 400, max_depth = 8, eta = 0.3, min_child_weight = 1 has the best performance so far,
#but start to observe difference in performance RMSE diff between test and train, might need to tune gamma soon

#save model
saveRDS(xgb_400_depth_child_fselect, "xgb_400_depth_child_fselect.rds")




#next, fix nrounds=400, eta=0.3, max_depth = 8,min_child_weight = 1 and tune colsample_bytree and subsample
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = c(0.8,1),
                         min_child_weight = 1,
                         subsample = c(0.6,0.8,1))


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_colsample_subsample_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                     metric="RMSE",
                                     trControl=trctrl,
                                     tuneGrid = tune_grid)


xgb_400_colsample_subsample_fselect
#load model
xgb_400_colsample_subsample_fselect<- readRDS("xgb_400_colsample_subsample_fselect.rds")


#test set with edx1 test set
test_predict_400_colsample_subsample_fselect <- predict(xgb_400_colsample_subsample_fselect, X_test_transformed_fselect)
residuals_400_colsample_subsample_fselect <- y_test - test_predict_400_colsample_subsample_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_400_colsample_subsample_fselect <- sqrt(mean(residuals_400_colsample_subsample_fselect^2))
#RMSE_test_400_colsample_subsample_fselect=0.8580418, not bad number as compared to cv-test-rmse:0.8572908
#nrounds = 400, max_depth = 8, eta = 0.3, min_child_weight = 1 has the best performance so far,
#but start to observe difference in performance RMSE diff between test and train, might need to tune gamma soon

#save model
saveRDS(xgb_400_colsample_subsample_fselect, "xgb_400_colsample_subsample_fselect.rds")




#next, fix nrounds=400, eta=0.3, max_depth = 8,min_child_weight = 1, olsample_bytree =1 and subsample=1
#tune gamma
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = c(0,3,6),
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_gamma_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                             metric="RMSE",
                                             trControl=trctrl,
                                             tuneGrid = tune_grid)


xgb_400_gamma_fselect
saveRDS(xgb_400_gamma_fselect, "xgb_400_gamma_fselect.rds")


#load model
xgb_400_gamma_fselect<- readRDS("xgb_400_gamma_fselect.rds")

#next, fix nrounds=400, eta=0.3, max_depth = 8,min_child_weight = 1, olsample_bytree =1 and subsample=1
#tune gamma
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = 6,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)
xgb_400_gamma_6_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                               metric="RMSE",
                               trControl=trctrl,
                               tuneGrid = tune_grid)


xgb_400_gamma_6_fselect



#test set with edx1 test set
test_predict_400_gamma_6_fselect <- predict(xgb_400_gamma_6_fselect, X_test_transformed_fselect)
residuals_400_gamma_6_fselect <- y_test - test_predict_400_gamma_6_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_400_gamma_6_fselect <- sqrt(mean(residuals_400_gamma_6_fselect^2))
#RMSE_test_400_gamma_6_fselect=0.858503, not bad number as compared to cv-test-rmse:0.8578022
#nrounds = 400, max_depth = 8, eta = 0.3, min_child_weight = 1 has the best performance so far,
#but start to observe difference in performance RMSE diff between test and train, might need to tune gamma soon

#save model
saveRDS(xgb_400_gamma_6_fselect, "xgb_400_gamma_6_fselect.rds")


#load model
xgb_400_gamma_6_fselect<- readRDS("xgb_400_gamma_6_fselect.rds")


validation1<-preprocess(validation)
validation1<-feature_eng(validation1)  


# Testing with validation set
X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1%>%select(rating)

#preprocess

X_test_val_transformed<-transform(X_test_val)
RMSE_val_200_default<-RMSE(xgb_200_default,X_test_val_transformed,y_test_val)


RMSE_val_200_default/2


RMSE_val<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals$rating^2))
  return(RMSE)
}



preprocessParams_X_test_val <- preProcess(X_test_val, method=c("center", "scale"))
X_test_val_transformed <- predict(preprocessParams_X_test_val, X_test_val)
X_test_val_transformed_fselect<-X_test_val_transformed%>%select(features_selected)

#test with 400 colsample subsample
test_val_predict_400_colsample_subsample_fselect <- predict(xgb_400_colsample_subsample_fselect, X_test_val_transformed_fselect)
residuals_val_400_colsample_subsample_fselect <- y_test_val - test_val_predict_400_colsample_subsample_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_colsample_subsample_fselect <- sqrt(mean(residuals_val_400_colsample_subsample_fselect$rating^2))

test_predict_200_default_fselect <- predict(xgb_200_default_fselect, X_test_transformed_fselect)
residuals_200_default_fselect <- y_test - test_predict_200_default_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_test_200_default_fselect <- sqrt(mean(residuals_200_default_fselect^2))




#predict

RMSE_val_200_default_fselect<-RMSE(xgb_200_default_fselect,X_test_val_transformed_fselect)

test_val_predict_400_depth_child_fselect <- predict(xgb_400_depth_child_fselect, X_test_val_transformed_fselect)
residuals_val_400_depth_child_fselect <- y_test_val - test_val_predict_400_depth_child_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_depth_child_fselect <- sqrt(mean(residuals_val_400_depth_child_fselect$rating^2))

#test with old model 400 default
test_val_predict_400_default_fselect <- predict(xgb_400_default_fselect, X_test_val_transformed_fselect)
residuals_val_400_default_fselect <- y_test_val - test_val_predict_400_default_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_default_fselect <- sqrt(mean(residuals_val_400_default_fselect$rating^2))


#test with 400 colsample subsample
test_val_predict_400_colsample_subsample_fselect <- predict(xgb_400_colsample_subsample_fselect, X_test_val_transformed_fselect)
residuals_val_400_colsample_subsample_fselect <- y_test_val - test_val_predict_400_colsample_subsample_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_colsample_subsample_fselect <- sqrt(mean(residuals_val_400_colsample_subsample_fselect$rating^2))

#test with 400 gamma 6
test_val_predict_400_gamma_6_fselect <- predict(xgb_400_gamma_6_fselect, X_test_val_transformed_fselect)
residuals_val_400_gamma_6_fselect <- y_test_val - test_val_predict_400_gamma_6_fselect
#residuals$rating because the column name that stores the values is "rating"
RMSE_val_400_gamma_6_fselect <- sqrt(mean(residuals_val_400_gamma_6_fselect$rating^2))

#save model
saveRDS(rf_fit, "model.rds")
#load model
rf_fit<- readRDS("model.rds")








trctrl <- trainControl(method = "cv", number = 5)
rf_fit <- train(X_train_transformed,y_train, method = "xgbTree",
                trControl=trctrl,
                tuneGrid = tune_grid)
#tuneLength = 10


# have a look at the model 
rf_fit

#fine tune max_depth and nrounds








Mat1 = data.matrix(X_train_transformed)

xgb_params_1 = list(
  objective = "reg:squarederror",                                               # binary classification
  eta = 0.1,                                                                  # learning rate
  max.depth = 4,                                                               # max tree depth
  eval_metric = "rmse"                                                          # evaluation/loss metric
)




# fit the model with the arbitrary parameters specified above
xgb_1 = xgboost(data = Mat1,
                label = y_train,
                params = xgb_params_1,
                nrounds = 100,                                                 # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 1,
                early.stop.round = 10                                          # stop if no improvement within 10 trees
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = as.matrix(df_train %>%
                                     select(-SeriousDlqin2yrs)),
                  label = df_train$SeriousDlqin2yrs,
                  nrounds = 100, 
                  nfold = 5,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  showsd = TRUE,                                               # standard deviation of loss across folds
                  stratified = TRUE,                                           # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1, 
                  early.stop.round = 10
)




#actual train
tune_grid <- expand.grid(nrounds = 100,
                         max_depth = c(3,5),
                         eta = 0.1,
                         gamma = 3,
                         colsample_bytree = c(0.5,0.7),
                         min_child_weight = 1,
                         subsample = 0.5)


trctrl <- trainControl(method = "cv", number = 5)
rf_fit <- train(X_train_transformed,y_train, method = "xgbTree",
                trControl=trctrl,
                tuneGrid = tune_grid)
                #tuneLength = 10


# have a look at the model 
rf_fit

# estimate variable importance
importance <- varImp(rf_fit, scale=FALSE)#feature_names = colnames(X_train_transformed),

# Testing with test set from edx
y_test<-test1Test$rating
X_test<-test1Test%>%select(-rating,-timestamp,-title)

preprocessParams_X_test <- preProcess(X_test, method=c("center", "scale"))
X_test_transformed <- predict(preprocessParams_X_test, X_test)

test_predict <- predict(xgb_200_default, X_test_transformed)
residuals <- y_test - test_predict
#residuals$rating because the column name that stores the values is "rating"
RMSE <- sqrt(mean(residuals^2))




check<-cbind(y_test_val,test_val_predict,residuals)




# Testing with validation set
X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1%>%select(rating)

preprocessParams_X_test_val <- preProcess(X_test_val, method=c("center", "scale"))
X_test_val_transformed <- predict(preprocessParams_X_test_val, X_test_val)

test_val_predict <- predict(xgb_200_default, X_test_val_transformed)
residuals_val <- y_test_val - test_val_predict
#residuals$rating because the column name that stores the values is "rating"
RMSE_val <- sqrt(mean(residuals_val$rating^2))

check<-cbind(y_test_val,test_val_predict,residuals)

#save model
saveRDS(rf_fit, "model.rds")
#load model
rf_fit<- readRDS("model.rds")
