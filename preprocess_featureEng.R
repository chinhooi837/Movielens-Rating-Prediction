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



feature_eng_org<-function(df){
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

transform<-function(X){
  preprocessParams_X <- preProcess(X, method=c("center", "scale"))
  X <- predict(preprocessParams_X, X)
  return(X)
}


RMSE<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals^2))
  return(RMSE)
}

RMSE_val<-function(model,X_test_transformed,y_test){
  test_predict <- predict(model, X_test_transformed)
  residuals<- y_test - test_predict
  RMSE<- sqrt(mean(residuals$rating^2))
  return(RMSE)
}
