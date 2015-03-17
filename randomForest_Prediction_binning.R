#Apply Random Forest after binning the windspeed variable.

#import necessary packages
library(randomForest)

#####Feature Engineering function: accepts data frame, returns data frame
featureEngineer <- function(df) {
  
  #convert season, holiday, workingday and weather into factors
  names <- c("season", "holiday", "workingday", "weather")
  df[,names] <- lapply(df[,names], factor)
  
  #Convert datetime into timestamps (split day and hour)
  df$datetime <- as.character(df$datetime)
  df$datetime <- strptime(df$datetime, format="%Y-%m-%d %T", tz="EST") #tz removes timestamps flagged as "NA"
  
  #convert hours to factors in separate feature
  df$hour <- as.integer(substr(df$datetime, 12,13))
  df$hour <- as.factor(df$hour)
  
  #Day of the week
  df$weekday <- as.factor(weekdays(df$datetime))
  df$weekday <- factor(df$weekday, levels = c("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")) #order factors
    
  #something that represents yearly growth:
  #extract year from date and convert to factor
  df$year <- as.integer(substr(df$datetime, 1,4))
  df$year <- as.factor(df$year)
  
  #return full featured data frame
  return(df)
}
# Binning of windspeed variable
train$windspeed =findInterval(train$windspeed,tapply(train$windspeed,cut(train$windspeed,breaks=6),median))

test$windspeed = findInterval(test$windspeed,tapply(test$windspeed,cut(test$windspeed,breaks=6),median))


######MAIN######
#Build features for train and Test set
train <- featureEngineer(train)
test <- featureEngineer(test)

#####RANDOM FOREST STARTS HERE#########
#variables
myNtree = 1000
myMtry = 6
myImportance = TRUE
#set the random seed
set.seed(415)
#fit and predict casual
casualFit <- randomForest(casual ~ hour + year + humidity + temp + atemp + workingday + windspeed, data=train, ntree=myNtree, mtry=myMtry, importance=myImportance)
test$casual <- predict(casualFit, test)
#fit and predict registered
registeredFit <- randomForest(registered ~ hour + year + season + weather + workingday + humidity + atemp + windspeed, data=train, ntree=myNtree, mtry=myMtry, importance=myImportance)
test$registered <- predict(registeredFit, test)
#add both columns into final count, round to whole number
test$count <- round(test$casual + test$registered, 0)

#testplot
plot(train$count)
plot(test$count)

####create output file from dataset test with predictions
submit <- data.frame (datetime = test$datetime, count = test$count)
write.csv(submit, file = "randomForest_Prediction_binning.csv", row.names=FALSE)
