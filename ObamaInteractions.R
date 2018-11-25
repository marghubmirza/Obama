# 
# install.packages(pkgs = "rpart", repos = "http://cran.us.r-project.org")  # Run only once on your machine.
# install.packages(pkgs = "rpart.plot", repos = "http://cran.us.r-project.org")  # Run only once on your machine.
# install.packages(pkgs = "glmnet", repos = "http://cran.us.r-project.org")  # Run only once on your machine.

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

lm <- lm(Obama_margin_percent ~ Region + (Black + HighSchool + Poverty  + Disabilities  + Bachelors)^2 , data = elect.df.smaller.train)
summary(lm)

lm.step <- step(lm, direction = "backward")
summary(lm.step)  # Which variables did it drop?

lm.pred <- predict(lm, elect.df.validation)
lm.step.pred <- predict(lm.step, elect.df.validation)

accuracy(lm.pred, elect.df.validation$Obama_margin_percent)
accuracy(lm.step.pred, elect.df.validation$Obama_margin_percent)

rt <- rpart(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities  + Bachelors, data = elect.df.smaller.train)  # Fits a regression tree.
prp(rt, type = 1, extra = 1)  # Use prp from the rpart.plot package to plot the tree.


rt.tuned <- rpart(Obama_margin_percent ~ Region + Black + HighSchool + Poverty  + Disabilities  + Bachelors , data = elect.df.smaller.train, control = rpart.control(cp = 0.005))
prp(rt.tuned, type = 1, extra = 1)

rt.pred <- predict(rt, elect.df.validation)
rt.tuned.pred <- predict(rt.tuned, elect.df.validation)

accuracy(lm.pred, elect.df.validation$Obama_margin_percent)
accuracy(lm.step.pred, elect.df.validation$Obama_margin_percent)
accuracy(rt.pred, elect.df.validation$Obama_margin_percent)
accuracy(rt.tuned.pred, elect.df.validation$Obama_margin_percent)


xtrain <- as.matrix(elect.df.smaller.train[, 10:41])
ytrain <- as.vector(elect.df.smaller.train$Obama_margin_percent)
ytrainb <- as.vector(elect.df.smaller.train$Obama_wins)
xvalid <- as.matrix(elect.df.validation[, 10:41])
yvalid <- as.vector(elect.df.validation$Obama_margin_percent)
yvalidb <- as.vector(elect.df.validation$Obama_wins)

lm.regularized <- glmnet(xtrain, ytrain, family = "gaussian")
#lm.regularized <- glmnet(xtrain, ytrainb, family = "binomial")
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
accuracy(as.vector(lm.regularized.pred), yvalidb)

#bm.all.fit <- lm(Obama_margin_percent ~ Region + Black + HighSchool + Poverty + PopDensity + AgeBelow35, data = elect.df.train)

# run all models and save for wisdom of crowds
fin.lm.pred <- predict(lm, elect.df.test)
fin.step.pred <- predict(lm.step, elect.df.test)
fin.rt.pred <- predict(rt, elect.df.test)
fin.rt.tuned.pred <- predict(rt.tuned, elect.df.test)
fin.regularized.pred <- predict(lm.regularized.cv, newx = as.matrix(elect.df.test[, 10:41]), s = "lambda.min") 

mat.lm <- as.vector(fin.lm.pred)
mat.step <- as.vector(fin.step.pred)
mat.rt <-  as.vector(fin.rt.pred)
mat.tuned <-  as.vector(fin.rt.tuned.pred)
mat.reg <- as.vector(fin.regularized.pred)
mat.pred <- as.data.frame( cbind(mat.lm, mat.step, mat.rt, mat.tuned,  mat.reg))

write.csv(mat.pred, "/Users/mirza/Box Sync/UVA/Obama/all.csv")


# data("mtcars")
# str(mtcars)
# cor(mtcars$disp , mtcars$hp)
# my_cor  <- cor(mtcars)
# corrplot(my_cor)
#cor_matrix <- as.data.frame(cor(elect.df.smaller.train[,10:41])) # To find all pairwise correlations.
cor_matrix <- cor(elect.df.smaller.train[,10:41]) # To find all pairwise correlations.

corrplot(cor_matrix)



#install.packages("psych")
library(psych)
pairs.panels( elect.df.train[, c( 11, 15,20,22,43)],
              method = "pearson", # correlation method
              hist.col = "#00AFBB",
              density = TRUE,  # show density plots
              ellipses = TRUE # show correlation ellipses
)

# pairs.panels( elect.df.train[, c( 11,15,   20,21, 22,34,39, 43)],
#               method = "pearson", # correlation method
#               hist.col = "#00AFBB",
#               density = TRUE,  # show density plots
#               ellipses = TRUE # show correlation ellipses
# )



