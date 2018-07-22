setwd('C:/Users/Abhinav/Downloads/My Assigments/Housing Prediction/Housing_Analysis/Housing_Analysis')

housingdata <- read.csv('housing.csv')

summary(housingdata)

#let do some data exploration

library(dplyr)
library(ggplot2)
library(ggthemes)


housingdata %>% 
    ggplot(aes(x = housing_median_age, y = median_house_value, color  = population)) + 
    geom_point(alpha = 0.6)+ scale_x_continuous(limits = c(1,52))+ scale_y_continuous(limits = c(14000,550000))+scale_colour_gradient(high='red',low = "blue") + theme_bw()

pairs(housingdata)

library(psych)
pairs.panels(housingdata[3:10], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
             )


#Dropping longitude,latitude, total_bedrooms variable

data_model <- housingdata %>% select(c(3:10))

library(corrplot)

corrplot(cor(data_model[1:7]))

data_model$total_bedrooms[is.na(data_model$total_bedrooms)] <- mean(data_model[['total_bedrooms']],na.rm = T)

summary(data_model)

######CREATING TEST & TRAIN DATA SETS######
## 70% of the sample size
smp_size <- floor(0.7 * nrow(data_model))
## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]

library(MASS)

library(car)


pricemodel <- lm(median_house_value ~.,data = train)

summary(pricemodel)

par(mfrow=c(2,2))
plot(pricemodel)

#checking normality

qqPlot(pricemodel$residuals)

shapiro.test(pricemodel$residuals)

#checking multicollinearity

vif(pricemodel)

# checking hetroscadstiy
library(lmtest)

bptest(pricemodel)

ncvTest(pricemodel)

plot(pricemodel$fitted.values,pricemodel$residuals)


#checking autocorrelation test durbin watson 

durbinWatsonTest(pricemodel)

#Autocorrelation is not significant

#Checking influence variable

infpoints <- influence.measures(pricemodel)

which(apply(infpoints$is.inf,1,any))

#Calculating cooks.distance

train$cooks.distance <- cooks.distance(pricemodel)

coo <- train$cooks.distance > 1

sum(coo)

#Calculating leverage 

train$leverage <- hatvalues(pricemodel)

avglev <- (12 + 1)/nrow(train)


levg <- train$leverage > 3*avglev

sum(levg)

#Calculating covariance ratio 

train$covariance.ratios <- covratio(pricemodel)

cvrup <- 1+ 3*avglev
cvrlo <- 1 - 3*avglev

cvr <- train$covariance.ratios < cvrlo | train$covariance.ratios > cvrup

sum(cvr)

#Calculating the standardized residual

train$standardized.residuals <- rstandard(pricemodel)

#Calculating the studentized residuals

train$studentized.residuals <- rstudent(pricemodel)

#finding out large residuals with absolute value greater than 2

train$large.residual <- train$standardized.residuals > 2| train$standardized.residuals < -2
sum(train$large.residual)

#finding out large residuals with absolute value greater than 2.5

train$large.residual2.5 <- train$standardized.residuals > 2.5| train$standardized.residuals < -2.5
sum(train$large.residual2.5)

#finding out large residuals with absolute value greater than 3

train$large.residual3 <- train$standardized.residuals > 3| train$standardized.residuals < -3
sum(train$large.residual3)

#Checking outlier and influence value

train[train$large.residual3,]

train[cvr,]

train[levg,]


#Using stepwise regression

# Stepwise regression model

null <- lm(median_house_value ~1, data = train)

#Stepwise Regression with forward direction

step1 <- stepAIC(null, scope = list(lower = null, upper =pricemodel),direction = "forward")

step1$anova # display result

#Stepwise Regression with backward direction

step2 <- stepAIC(pricemodel,direction = "backward")

step2$anova #display result

#stepwise Regression with both direction

step3 <- stepAIC(null, scope = list(upper = pricemodel), direction = "both")

step3$anova #display result
summary(step.model)

#Model has hetero sedcatiy
library(caret)

distBCMod <- BoxCoxTrans(train$median_house_value)

print(distBCMod)

train <- cbind(train,new_price = predict(distBCMod,train$median_house_value))

train$median_house_value <- NULL


newmodel <- lm(new_price ~.,data = train)

summary(newmodel)

par(mfrow=c(2,2))
plot(newmodel)

# checking hetroscadstiy
library(lmtest)

bptest(newmodel)

ncvTest(newmodel)

plot(newmodel$fitted.values,newmodel$residuals)

library(gvlma) # global validation of linear model assumptions

gvlma(pricemodel)

#testing the model

test.lm = predict(newmodel, newdata = test)

#Calculating R - square for test data

SS.residual   <- sum((test.lm - log(test$median_house_value))^2)

SS.total<- sum((log(test$median_house_value)-mean(log(test$median_house_value)))^2)

R.square <- 1 - SS.residual/SS.total

plot(exp(test.lm),test$median_house_value,xlab = "predicted",ylab = "Actual")

abline(0,1)

