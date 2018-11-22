
#install.packages(pkgs = "rpart", repos = "http://cran.us.r-project.org")  # Run only once on your machine.
#install.packages(pkgs = "rpart.plot", repos = "http://cran.us.r-project.org")  # Run only once on your machine.
#install.packages(pkgs = "glmnet", repos = "http://cran.us.r-project.org")  # Run only once on your machine.
#install.packages("corrplot")

library(rpart)
library(rpart.plot)
library(glmnet)
library(forecast)
library(corrplot)

elect.df <- read.csv("/Users/mirza/Box Sync/UVA/Obama/Obama.csv")
dim(elect.df)  # How many rows and columns are there?
head(elect.df)
tail(elect.df)

cbind(seq(1, 41), colnames(elect.df))

summary(elect.df)

ImputeData <- function(vec, mn) {
  ifelse(is.na(vec), mn, vec)
}

(data.mean <- sapply(elect.df[ , 10:41], mean, na.rm = TRUE))
(my.mean <- as.data.frame (sapply(elect.df[ , 10:41], mean, na.rm = TRUE)))


for(i in 10:41) {
  elect.df[, i] <- ImputeData(elect.df[ , i], data.mean[i - 9])
}

summary(elect.df)

elect.df$ElectionDate <- as.Date(elect.df$ElectionDate, format="%m/%d/%Y")
elect.df.train <- elect.df[elect.df$ElectionDate < as.Date("2/19/2008", format = "%m/%d/%Y"), ]
elect.df.test <- elect.df[elect.df$ElectionDate >= as.Date("2/19/2008", format = "%m/%d/%Y"), ]

elect.df.train$Obama_margin <- elect.df.train$Obama - elect.df.train$Clinton
elect.df.train$Obama_margin_percent <- elect.df.train$Obama_margin / elect.df.train$TotalVote
elect.df.train$Obama_wins <- ifelse(elect.df.train$Obama_margin > 0, 1, 0)
names(elect.df.train)

(nTrain <- nrow(elect.df.train))

(nSmallTrain <- round(nrow(elect.df.train) * 0.75))
(nValid <- nTrain - nSmallTrain)

set.seed(201)

rowIndicesSmallerTrain <- sample(1:nTrain, size = nSmallTrain, replace = FALSE)

elect.df.smaller.train <- elect.df.train[rowIndicesSmallerTrain, ]
elect.df.validation <- elect.df.train[-rowIndicesSmallerTrain, ]

lm <- lm(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities + AgeBelow35 + SpeakingNonEnglish + Asian  + Hawaiian,  data = elect.df.smaller.train)
summary(lm)


lm.step <- step(lm, direction = "backward")
summary(lm.step)  # Which variables did it drop?

lm.pred <- predict(lm, elect.df.validation)
lm.step.pred <- predict(lm.step, elect.df.validation)

accuracy(lm.pred, elect.df.validation$Obama_margin_percent)
accuracy(lm.step.pred, elect.df.validation$Obama_margin_percent)

rt <- rpart(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities +  AgeBelow35 + SpeakingNonEnglish + Asian  + Hawaiian, data = elect.df.smaller.train)  # Fits a regression tree.
prp(rt, type = 1, extra = 1)  # Use prp from the rpart.plot package to plot the tree.

rt.tuned <- rpart(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities +   AgeBelow35 + SpeakingNonEnglish + Asian  + Hawaiian, data = elect.df.smaller.train, control = rpart.control(cp = 0.005))
prp(rt.tuned, type = 1, extra = 1)

rt.pred <- predict(rt, elect.df.validation)
rt.tuned.pred <- predict(rt.tuned, elect.df.validation)

accuracy(lm.pred, elect.df.validation$Obama_margin_percent)
accuracy(lm.step.pred, elect.df.validation$Obama_margin_percent)
accuracy(rt.pred, elect.df.validation$Obama_margin_percent)
accuracy(rt.tuned.pred, elect.df.validation$Obama_margin_percent)


# MIRZA Alam Mirza

cor(elect.df.smaller.train$Medicare,elect.df.smaller.train$Disabilities)
#cor_matrix <- as.data.frame(cor(elect.df.smaller.train[,10:41])) # To find all pairwise correlations.
cor_matrix <- cor(elect.df.smaller.train[,10:44]) # To find all pairwise correlations.

corrplot(cor_matrix)

data("mtcars")
str(mtcars)
cor(mtcars$disp , mtcars$hp)
my_cor  <- cor(mtcars)
corrplot(my_cor)

lm.with.mc <- lm(Obama_margin_percent ~ Region + Black + HighSchool + Poverty + PopDensity + Medicare + Disabilities, data = elect.df.smaller.train)
summary(lm.with.mc)

lm.without.mc <- lm(Obama_margin_percent ~ Region + Black + HighSchool + Poverty + PopDensity + Disabilities, data = elect.df.smaller.train)
summary(lm.without.mc)

xtrain <- as.matrix(elect.df.smaller.train[, 10:41])
ytrain <- as.vector(elect.df.smaller.train$Obama_margin_percent)
xvalid <- as.matrix(elect.df.validation[, 10:41])
yvalid <- as.vector(elect.df.validation$Obama_margin_percent)

lm.regularized <- glmnet(xtrain, ytrain, family = "gaussian")
plot(lm.regularized, xvar = "lambda", label = TRUE) 

lm.regularized.cv <- cv.glmnet(xtrain, ytrain, nfolds = 5, family = "gaussian")  # Fits the Lasso.
(minLogLambda <- log(lm.regularized.cv$lambda.min))
coef(lm.regularized.cv, s = "lambda.min")  

plot(lm.regularized, xvar = "lambda", label = TRUE)
abline(v = minLogLambda)

lm.regularized.pred <- predict(lm.regularized.cv, newx = xvalid, s = "lambda.min") 

accuracy(lm.pred, yvalid)
accuracy(lm.step.pred, yvalid)
accuracy(rt.pred, yvalid)
accuracy(rt.tuned.pred, yvalid)
accuracy(as.vector(lm.regularized.pred), yvalid)

bm.all.fit <- lm(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities + AgeBelow35 + SpeakingNonEnglish + Asian  + Hawaiian, data = elect.df.train)


bm.all.fit.pred <- predict(bm.all.fit, elect.df.test)

# all4  <- cbind(bm.all.fit.pred, lm.step.pred, rt.pred, lm.regularized.pred)

write.csv(bm.all.fit.pred, "/Users/mirza/Box Sync/UVA//Obama/Out.csv")
write.csv(lm.step.pred, "/Users/mirza/Box Sync/UVA//Obama/Out.csv")
write.csv(rt.tuned.pred, "/Users/mirza/Box Sync/UVA//Obama/Out.csv")
write.csv(bm.all.fit.pred, lm.step.pred, rt.pred, lm.regularized.pred "/Users/mirza/Box Sync/UVA/Obama/all4.csv")

# write.csv(cor_matrix, "/Users/mirza/Box Sync/UVA/Obama/CoMatrix.csv")
