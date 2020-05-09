

#install and load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(googledrive)) install.packages("googledrive", repos = "http://cran.us.r-project.org")
if(!require(httpuv)) install.packages("httpuv", repos = "http://cran.us.r-project.org")

################################
# Create edx set, validation set
################################
#We will download the edx data set and validation set using the code provided as follows.
#This is a sample code provided by edx.

# Note: this process could take a couple of minutes


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



#Do note that the validation set is only reserved for final validation. 
#For model training and performance optimization, we will use the test set split from the edx.



#Exploratory Data Analysis

#We will started by answering the following questions using exploratory data analysis.  

#1) What is the rating that were given most frequently? 
#We can study this by looking at the histogram provided by this piece of code.

#It can be seen that 3 and 4 has a much higher frequency as compared to the others, 
#and not many users rate a movie with a half rating (i.e. 0.5, 1.5 and etc).


edx%>%ggplot(aes(x=rating))+
  geom_histogram(binwidth=0.5,colour="blue",
                 aes(y=..density..,fill=..count..))+
  scale_fill_gradient("Count",low="turquoise",high="turquoise4")#+



#2) What is distribution of the rating by user? 
#We can use the following piece of code to produce a histogram of rating by user mean rating. 

# We can see that many users have a mean rating of 3.6-3.7, quite close to the mean of edx which is `r round(mean(edx$rating),2)`.
# It is also obvious that some users are harsher than the others, with average rating below 2.5 whereas some have a mean ratint of 4.5 and above.
# Hence, we should include user bias into our model.


#Mean Rating given by the users
edx %>% group_by(userId)%>%
  summarise(UserAverage=mean(rating)) %>% 
  ggplot(aes(UserAverage)) + 
  geom_histogram(bins=30, color = "black",fill = "turquoise")+
  ggtitle("Mean Rating by User")


# 3) What is distribution of the rating by movie? We can use the following piece of code to produce a histogram of rating by movie mean rating. 
# 
# We can see that many movies have a mean rating of 3.5-3.7.
# It can be shown that some movies are more well liked than the others, have an average rating of 4.5 and above, and the opposite is true.
# Hence, we should include movie bias into our model.



#Mean Rating given by the movies
edx %>% group_by(movieId)%>%
  summarise(MovieAverage=mean(rating)) %>% 
  ggplot(aes(MovieAverage)) + 
  geom_histogram(bins=30, color = "black",fill = "turquoise")+
  ggtitle("Mean Rating by Movie")


# 4) What about the distribution of the total number of rating grouped by movies? 
# The following piece of code produce a histogram of total number of ratings grouped by movies.
# It can be seen that some movies are rated very frequently while some are not so popular. 
#This is understanable as some blockbusters are much popular than the common movies. Thus, we need to be careful and might need to remove some movies that only have 1 rating. 


edx %>% count(userId) %>% 
  ggplot(aes(n))+
  geom_histogram(bins = 30,color = "black",fill = "turquoise")+
  scale_x_log10()+
  xlab("Number of Ratings")+
  ylab("Number of Users")+
  ggtitle("Number of ratings given by users")


## Modelling  

### Modelling techniques  


# To train a good model, we will need to go through the following steps:
#   
# 1. Preprocessing - Preprocess the data to ensure a clean and consistent data frame are input to the model
# 2. Feature Engineering - Adding more useful features and do experiment to validate the performance.
# 3. Feature Selection - selected the top features input to the models and remove the remaining to reduce the noise to the model varImp of caret package in R.
# 4. Hyperparameters tuning - Tune the hyperparameters of the resepctive models selected to ensure that our models are optimized.
# 
# XGBoost is chosen for our modeling for this project as this parellel tree boosting method is able to solve many data science problems in an efficient and accurate way. It main advantages are the 
# 
# * Execution Speed 
# * Model Performance 
# 
# Details can be found in these links: 
#   
#   https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
#   
#   https://xgboost.readthedocs.io/en/latest/
#   
# 


### Baseline Model  


# Before we proceed to train a first baseline model, 
# we would need to go through the necessary process of preprocessing and feature engineering.
# 
# As for the baseline model, we only preprocess the data by separating all the genres into different columns and give them a 1 or 0, similar to the concept of one hot encoding for categorical variables. Do note that this process is quite computational resource intensive, you can use the uploaded preprocessed file to save time. 
# The preprocess function can be done by calling the following function. 
#######IMPORTANT#############
## Note: this preprocess function could take half an hour for our data, hence I suggest we can use the code after the function to download the preprocess edx file called "edx1_default"
## please follow the instruction for drive_auth() to download the file from google drive.

#############################
#preprocess function 
preprocess<-function(df){
  #remove '-' from Sci-Fi, Film-Noir
  df1<-df
  df1$genres<-gsub('-','',df1$genres)
  #replace no genres listed with NA if it exists
  if(nrow(df1%>%select(genres)%>%filter(genres=="(no genres listed)"))!=0){
    df1 <- df1 %>% na_if("(no genres listed)")
    
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

#this is the line to call the function that will take very long that we should skip, suggest we use the output file instead 
edx1<-preprocess(edx)

#setup google account authorization to download file from google drive
#I have created a dummy google account
#When you see "Selection: " , Enter 0 to jump to browser
#Sign in with the account: 
#User Name: edxmovielensych@gmail.com 
#password: welcometoedx
#click "Allow" to allow tidyverse to access this dummy account's info
drive_auth()

#Setup temp file to download google drive file
temp_edx1_default <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("10IyVolVExpG8DUpIJ2_8T1Yti-hXXGyE"), path = temp_edx1_default, overwrite = TRUE)
#unzip the file then read the csv
out_edx1_default <- unzip(temp_edx1_default, exdir = tempdir())
edx1<- read_csv(out_edx1_default,col_names = TRUE)



#feature engineering function for baseline model 
feature_eng_default<-function(df){
  df1<-df
  #remove NA column created during preprocessing to prepare for modeling
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

#process edx1 with the function
edx1<-feature_eng_default(edx1)




#Preparing to train the baseline model
#set seed (1) twice to ensure repeatability as without this it sometimes cannot produce the same result
set.seed(1)
library(caret)
set.seed(1)

#Splitting data to train and test set
trainIndex <- createDataPartition(edx1$rating, p = .8, 
                                  list = FALSE, 
                                  times = 1)
#ensure edx1 is dataframe as it might be tbl.dataframe if we download it from google drive
#remove timestamp and title from train set as these could cause noise to the model
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


#Training of baseline model with all default parameters
#setting tune grid with all default parameters
tune_grid <- expand.grid(nrounds = 100,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)

#setting Cross validation number to 5
trctrl <- trainControl(method = "cv", number = 5)

#######IMPORTANT#############
## Note: this modeling training could take 8 hours for our data, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################

#Actual training step of the model
xgb_first_100_default <- train(X_train_transformed_default,y_train, method = "xgbTree",
                               metric="RMSE",
                               trControl=trctrl,
                               tuneGrid = tune_grid)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_first_100_default <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("1ZBJNtd6Jm3b6rLt5bsCDzSavb8plk5UC"), path = temp_xgb_first_100_default, overwrite = TRUE)
#unzip the file then read the model
out_xgb_first_100_default <- unzip(temp_xgb_first_100_default, exdir = tempdir())
xgb_first_100_default<- readRDS(out_xgb_first_100_default)

#############################################################

#Creating function of RMSE to evaluate the performance of the model 
RMSE<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals^2))
  return(RMSE)
}

#calling RMSE function by using our test set
RMSE_default<-RMSE(xgb_first_100_default,X_test_transformed_first,y_test_first)
#creating a RMSE_results to store all RMSE from different models to compare result
RMSE_results<-data.frame(Method = "Baseline_Model", RMSE = RMSE_default)


#It turns out that the RMSE for baseline model is 1.0234772



### Improved Model 1 - with movie and user bias added as features  
# Next up, We are adding movie and user bias, as well as the number of days since the user first rate a movie as features.

library(lubridate)

#New Feature Engineering function to add in the new features related to movie and user bias,
#as well as the number of days since the user first rate a movie as features.
feature_eng_bias<-function(df){
  df1<-df
  #changing the timestamp format
  df1<-df1%>%mutate(timestamp=as_datetime(timestamp,origin = lubridate::origin, tz = "UTC"))
  
  #prepare columns for new features- UserAverage, Movie Average, 
  #FirstUserRatingDate,FirstMovieRatingDate
  User<-data.frame(df1%>%
                     group_by(userId)%>%summarise(FirstUserRatingDate=min(timestamp),
                                                  UserAverage=mean(rating))) 
  
  MovieAverage<-data.frame(df1%>%
                             group_by(movieId)%>%summarise(FirstMovieRatingDate=min(timestamp),
                                                           MovieAverage=mean(rating)))
  #join to main data frame and produce the intended features, 
  #BaselineRating,UserBias,MovieBias,daySinceFirstUserRating,daySinceFirstMovieRating
  df1<-df1%>%mutate()%>%
    inner_join(User,by="userId")%>%
    inner_join(MovieAverage,by="movieId")%>%
    mutate(BaselineRating=mean(rating),
           UserBias=UserAverage-BaselineRating,
           MovieBias=MovieAverage-BaselineRating,
           daySinceFirstUserRating=as.integer(difftime(timestamp,
                                                       FirstUserRatingDate,
                                                       units = "days")),
           daySinceFirstMovieRating=as.integer(difftime(timestamp,
                                                        FirstMovieRatingDate,
                                                        units = "days")))%>%
    select(-FirstUserRatingDate,-FirstMovieRatingDate)
  return (df1)
}  

#process our data by calling this new feature engineering function 
edx1<-feature_eng_bias(edx1)



#resplit the train and test set for the new data and perform standardization
set.seed(1)
test1Train <- edx1[trainIndex,]
test1Test  <- edx1[-trainIndex,]
y_train<-test1Train$rating
X_train<-test1Train%>%select(-rating,-timestamp,-title)
y_test<-test1Test$rating
X_test<-test1Test%>%select(-rating,-timestamp,-title)

#standardization
X_train_transformed<-transform(X_train)
#standardization
X_test_transformed<-transform(X_test)

# # building baseline models with default parameters, find out the optimal nrounds
Mat1 = data.matrix(X_train_transformed)
params <- list(nrounds = 100,
               max_depth = 6,
               eta = 0.3,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)
xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 100, nfold = 5,
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 10, maximize = F)
# #stops at 100 rounds. 	test-rmse:0.864534+0.000451 
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
#stops at 200 rounds test-rmse:0.862751+0.000573 


#Building the next model with nrounds = 200 as the default nrounds = 100 is not enough after testing with xgb.cv and calculate the RMSE for the test set. 
#actual train with Improved  model 1 with 200 rounds 
tune_grid <- expand.grid(nrounds = 200,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)


#######IMPORTANT#############
## Note: this modeling training could take 10 hours for our data, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################

#Actual training step
xgb_200_default <- train(X_train_transformed,y_train, method = "xgbTree",
                         metric="RMSE",
                         trControl=trctrl,
                         tuneGrid = tune_grid)



######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_200_default <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("15PqOOCtwOsZ83eGBohGl8fHJrj0KSb-6"), path = temp_xgb_200_default, overwrite = TRUE)
#unzip the file then read the model
out_xgb_200_default <- unzip(temp_xgb_200_default, exdir = tempdir())
xgb_200_default<- readRDS(out_xgb_200_default)

#############################################################



RMSE_200_default<-RMSE(xgb_200_default,X_test_transformed,y_test)
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method= "Improved_Model_1_Bias",
                                     RMSE=RMSE_200_default))



#Hence, we can see that the RMSE for Improved Model 1 is 0.8637766. Significant improvement from baselin model
#We will further improve the model by doing features selection in the next section. 




### Feature Selection and Improved Model 2  

#Find out the important features used for our model
importance <- varImp(xgb_200_default, scale=FALSE)
#select top 10 features
features_selected<- rownames(importance$importance)[1:10]
X_train_transformed_fselect<-X_train_transformed%>%select(features_selected)
X_test_transformed_fselect<-X_test_transformed%>%select(features_selected)


# Retrain model again, this time with nrounds = 400 because
# 
# * 200 is still not hitting the early stopping criteria previously
# * Anything above 400 is too computational expensive and will take us very long to train a model 


#actual train with default model with only top 10 featureds extracted out to test with test set
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 6,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)

#######IMPORTANT#############
## Note: this modeling training could take 5 hours for our data, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################

xgb_400_default_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                 metric="RMSE",
                                 trControl=trctrl,
                                 tuneGrid = tune_grid)




######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_400_default_fselect <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("1qgbIUB8Ujeyo8d3I48G7-j0j5jEqo8Sc"), path = temp_xgb_400_default_fselect, overwrite = TRUE)
#unzip the file then read the model
out_xgb_400_default_fselect <- unzip(temp_xgb_400_default_fselect, exdir = tempdir())
xgb_400_default_fselect<- readRDS(out_xgb_400_default_fselect)

#############################################################

#Calculating RMSE for this new model
RMSE_400_default_fselect<-RMSE(xgb_400_default_fselect,
                               X_test_transformed_fselect,y_test)
#Adding this result into the combined RMSE_results
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method= "Improved_Model_2_Feature_selection",
                                     RMSE=RMSE_400_default_fselect))


#It can be seen that doing features selection really does help on the performance, 
#improving the performance to 0.8602110.



### Hyperparameters Tuning & Improved Model 3  

# 
# xgboost models has the following hyperparameters to tune to obtain the best optimized data for our data analysis. 
# 
# + eta 
# + min_child_weight
# + max_depth 
# + colsample_bytree
# + subsample
# + gamma
# 
# 
# #### eta (learning rate)
# 
# As for eta (learning rate), we tried to reduce from the default 0.3 to 0.1 
# but we realized that it was too computational expensive with a worse performance (based on crossed validation as shown in the code below).

#Use cross validation to obtain a test performance with lower eta (0.1)
params <- list(nrounds = 400,
               max_depth = 6,
               eta = 0.1,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)
Mat1<-data.matrix(X_train_transformed_fselect)
xgbcv <- xgb.cv( params = params, data = Mat1,label = y_train, nrounds = 400, nfold = 5, 
                 showsd = T, stratified = T, 
                 print_every_n = 10, early_stop_round = 5, maximize = F)
#test-rmse:0.863981+0.000433 
#worse than eta=0.3, and much computational heavy for nrounds =400 and eta=0.1
#hence, eta and nrounds should be fixed at 0.3 and 400 respectively



#### max_depth and min_child_weight  
# We will now tune max_depth and min_child_weight. 
#These two should be tuned together and first among the others parameters to determine the maximum depth and minimum child weight of the gradient boosting decision trees, 
#basically deciding when splitting the nodes of respective decision tree. 
#The resulting number of max_depth and min_child_weight is 8 and 1 respectively.

tune_grid <- expand.grid(nrounds = 400,
                         max_depth = c(4,6,8),
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = c(1,3,5),
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)




#######IMPORTANT#############
## Note: this modeling training could take 10 hours for our to tune up these 9 possibilities, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################



#Actual training step
xgb_400_depth_child_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                     metric="RMSE",
                                     trControl=trctrl,
                                     tuneGrid = tune_grid)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_400_depth_child_fselect <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("1BOSKSDQpm4end6_h7iv_dTcJbWIttfNs"), path = temp_xgb_400_depth_child_fselect, overwrite = TRUE)
#unzip the file then read the model
out_xgb_400_depth_child_fselect <- unzip(temp_xgb_400_depth_child_fselect, exdir = tempdir())
xgb_400_depth_child_fselect<- readRDS(out_xgb_400_depth_child_fselect)

#############################################################

#Calculating RMSE for this new model
RMSE_400_depth_child_fselect<-RMSE(xgb_400_depth_child_fselect,
                                   X_test_transformed_fselect,y_test)

#The results improves to 0.85019 from 0.86021, showing the importance of parameter tuning



#### colsample_bytree and subsample  



# Next, we will tune colsample_bytree and subsample, keeping the previously tuned parameter constant, and others at default value. 
# Colsample_bytree and subsample are the parameters that determine the number of columns and rows to select during resampling process for each decision trees. 
# The resulting colsample_bytree and subsample are 1 for both. 
# This could be due to the fact that we have done a features selection of top 10 features, 
# hence the resampling at 100% for both columns and rows. 


#nrounds=400, eta=0.3, max_depth = 8,min_child_weight = 1 and tune colsample_bytree and subsample
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = 0,
                         colsample_bytree = c(0.8,1),
                         min_child_weight = 1,
                         subsample = c(0.6,0.8,1))


trctrl <- trainControl(method = "cv", number = 5)

#######IMPORTANT#############
## Note: this modeling training could take 8 hours for our data, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################

#Actual training step
xgb_400_colsample_subsample_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                                             metric="RMSE",
                                             trControl=trctrl,
                                             tuneGrid = tune_grid)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_400_colsample_subsample_fselect <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("1GPTWlMSOjL7ur9M4x4QseyTIJARvfioJ"), path = temp_xgb_400_colsample_subsample_fselect, overwrite = TRUE)
#unzip the file then read the model
out_xgb_400_colsample_subsample_fselect <- unzip(temp_xgb_400_colsample_subsample_fselect, exdir = tempdir())
xgb_400_colsample_subsample_fselect<- readRDS(out_xgb_400_colsample_subsample_fselect)

#############################################################

#We obtain the same RMSE as compare to xgb_400_depth_child_fselect model as the best parameters are the same
RMSE_400_colsample_subsample_fselect<-RMSE(xgb_400_colsample_subsample_fselect,
                                           X_test_transformed_fselect,y_test)





#### gamma  


# Last but not least, we are going to tune gamma, this is the factor that decides the regularization of model. 
# The resulting gamma chosen is 0, which means no regularization is needed for our model. This is understandable as:
#   
# 1. Movie bias, user bias and other bias are mostly taken care of by the modelling process.
# 2. We do not observe any big difference between the training and testing RMSE.



#fix nrounds=400, eta=0.3, max_depth = 8,min_child_weight = 1, olsample_bytree =1 and subsample=1
#tune gamma
tune_grid <- expand.grid(nrounds = 400,
                         max_depth = 8,
                         eta = 0.3,
                         gamma = c(0,3,6),
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)


trctrl <- trainControl(method = "cv", number = 5)


#######IMPORTANT#############
## Note: this modeling training could take 8 hours for the tuning parameters, 
#hence I suggest we can use the code after this training "Actual training step" to download the model
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################

#Actual training step
xgb_400_gamma_fselect <- train(X_train_transformed_fselect,y_train, method = "xgbTree",
                               metric="RMSE",
                               trControl=trctrl,
                               tuneGrid = tune_grid)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_xgb_400_gamma_fselect <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("153GHH3GpXYfBICZsWNqRZXTshwqTEVIv"), path = temp_xgb_400_gamma_fselect, overwrite = TRUE)
#unzip the file then read the model
out_xgb_400_gamma_fselect <- unzip(temp_xgb_400_gamma_fselect, exdir = tempdir())
xgb_400_gamma_fselect<- readRDS(out_xgb_400_gamma_fselect)

#############################################################


RMSE_400_gamma_fselect<-RMSE(xgb_400_gamma_fselect,X_test_transformed_fselect,y_test)
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Method= "Improved_Model_3_HyperParameters_tuning",
                                     RMSE=RMSE_400_gamma_fselect))
RMSE_results%>%head(5)

# The RMSE now maintained at 0.8502 as the tuning parameters are the same.
# We can also see that the RMSE has improved from the baseline model of 1.0235 to 0.8502.



#### Final Model Selection   


# We will be selecting the final model judging by the model performance.
# As we have gone through various cycles of improvement, the RMSE reduces from the baseline 1.0234 to 0.8502.
# Hence the final model we selected is Improved Model 3. 

RMSE_results%>%head(5)


### Model Validation with Validation Set

# We will now validate our final model with the validation set. 

#######IMPORTANT#############
## Note: this preprocess function could take 0.5 hour for our data, 
#hence I suggest we can use the code after this "preprocess and feature engineering"
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#############################
#preprocess and feature engineering
validation1<-preprocess(validation)
validation1<-feature_eng_default(validation1)
validation1<-feature_eng_bias(validation1)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_validation_process <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("10IyVolVExpG8DUpIJ2_8T1Yti-hXXGyE"), path = temp_validation_process, overwrite = TRUE)
#unzip the file then read the csv
out_validation_process <- unzip(temp_validation_process, exdir = tempdir())
validation1<- read_csv(out_validation_process,col_names = TRUE)
validation1<-data.frame(validation1)

#############################################################

#split to X and y and standardize the data
X_test_val<-validation1%>%select(-rating,-timestamp,-title)
y_test_val<-validation1$rating
X_test_val_transformed<-transform(X_test_val)
X_test_val_transformed_fselect<-X_test_val_transformed%>%select(features_selected)

#Calculate the RMSE of the validation set using the final model
RMSE_val_400_gamma_fselect<-RMSE(xgb_400_gamma_fselect,X_test_val_transformed_fselect,y_test_val)
RMSE_comparison<-data.frame(Data_Set = "Test_Set", RMSE = RMSE_400_gamma_fselect)
RMSE_comparison <- bind_rows(RMSE_comparison,
                             data.frame(Data_Set= "Validation_Set",
                                        RMSE=RMSE_val_400_gamma_fselect))


# Do note that validation set is not used for any performance validation until this final step. It
# can be seen that our performance of 0.8446133 has far exceeded the 0.86490 criteria of this project to qualify
# for full marks.

#We also generate the predicted rating on Validation set with the code as follows,

#Generate predicted rating
Predicted_ratings_Validation <- predict(xgb_400_gamma_fselect, X_test_val_transformed_fselect)
Validation_Set_with_PredictedRatings<-cbind(validation2,Predicted_ratings_Validation)


######CODE TO DOWNLOAD FROM GOOGLE DRIVE ###################
## please follow the instruction for drive_auth() previously in edx1 to download the file from google drive.

#Setup temp file to download google drive file
temp_Validation_Set_with_PredictedRatings <- tempfile(fileext = ".zip")
dl <- drive_download(
  as_id("1KPuGN1b0aYDd0_Q13JMHAGfBMQ3MVVpr"), path = temp_Validation_Set_with_PredictedRatings, overwrite = TRUE)
#unzip the file then read the csv
out_Validation_Set_with_PredictedRatings <- unzip(temp_Validation_Set_with_PredictedRatings, exdir = tempdir())
Validation_Set_with_PredictedRatings<- read_csv(out_Validation_Set_with_PredictedRatings,col_names = TRUE)

#############################################################


# # Conclusion  
# We have built a series of useful machine learning algorithm to predict the movie rating.
# It can be seen that by adding user and movie bias, and optimizing our models with features selection and hyperparameters tuning, we can achieve a much better performance as compared to the baseline model. 
# The final model has a RMSE of `r RMSE_val_400_gamma_fselect` which is much lower than the 0.86490 criteria of this project to qualify for full marks. 
# Last but not least, other machine learning models could also improve the results further, but due to time and hardware constraints, we have yet to test it further. 


