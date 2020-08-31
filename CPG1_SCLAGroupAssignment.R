library(readxl)
library(outliers)
library(scales)
library(plotrix)
library(lattice)
library(GGally)
library(psych) #correlation
library(caTools) #sample.split
library(DMwR) #SMOTE
library(UBL) #SMOTE multiclass
library(ROSE)
library(stringr) #Fixing column names
library(caret)
library(nnet) #Multinominl log reg
library(ROCR) #pr CURVE
library(ineq)
library(e1071) #SVM
library(class) #KNN
library(ipred) #ipred
library(rpart)
library(rpart.plot)
library(mltools)
library(data.table)
library(rattle)
library(gbm)

Inventory<-read_xlsx("09_Inventory.xlsx")
dim(Inventory)

Inventory<-as.data.frame(Inventory)

#Missing value check
sum(is.na(Inventory))

str(Inventory)

#Converting categorical variables into factors
Inventory$`Ship Mode`<-as.factor(Inventory$`Ship Mode`)
Inventory$`Product Container`<-as.factor(Inventory$`Product Container`)
Inventory$`Product Sub-Category`<-as.factor(Inventory$`Product Sub-Category`)


#Dropping irrelevant variables Order date, order ID and Product name
Inventory<-Inventory[-c(1,2,5)]

head(Inventory)
tail(Inventory)

summary(Inventory)

#Check for outliers
boxplot(Inventory$`Order Quantity`,col = "orange",main="Boxplot Order Qty.")
boxplot(Inventory$Sales,col = "orange",main="Boxplot Sales")
summary(boxplot(Inventory$Sales,plot = FALSE)$out)

table(Inventory$`Ship Mode`)

#Outlier treatment
outlier_capping <- function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}

Inventory$Sales=outlier_capping(Inventory$Sales)
boxplot(Inventory$Sales,col = "orange")

#Univariate - Histograms
par(mfrow=c(1,2))
hist(Inventory$`Order Quantity`, main = "Histogram of Order Qty.", xlab = "Order Qty.",col = "Red")
hist(Inventory$Sales, main = "Histogram of Sales", xlab = "Sales",col = "Red")
dev.off()

#Scatterplots
options(scipen = 1000)
plot(Inventory$`Order Quantity`,ylab = "Order Qty.",col="Green",main="Scatterplot Order Qty.")
plot(Inventory$Sales,ylab = "Sales",col="Green",main="Scatterplot Sales")

#Pie chart
dev.off()
par(mfrow=c(1,2))
pie(table(Inventory$`Product Container`))
pie(table(Inventory$`Product Sub-Category`))
pie(table(Inventory$`Ship Mode`))

#Bi-variate analysis
histogram(~Inventory$`Ship Mode`|factor(Inventory$`Product Container`),data = Inventory,
          main="Ship mode wrt Product Container",xlab = "Ship Mode")

histogram(~Inventory$`Ship Mode`|factor(Inventory$`Product Sub-Category`),data = Inventory,
          main="Ship mode wrt Product Sub-Category",xlab = "Ship Mode")

par(mfrow=c(1,3))
dev.off()
boxplot(Inventory$Sales~Inventory$`Ship Mode`, main = "Sales vs Ship Mode",
        xlab = "Ship Mode",ylab = "Sales")
boxplot(Inventory$Sales~Inventory$`Product Container`,main = "Sales vs Prod Container",
        xlab = "Product Container",ylab = "Sales",las=2)
boxplot(Inventory$Sales~Inventory$`Product Sub-Category`,main = "Sales vs Prod Sub Cat",
        xlab="Product Sub Cat", ylab = "Sales",las=2)


boxplot(Inventory$`Order Quantity`~Inventory$`Ship Mode`, main = "Ship Mode vs Order Qty.",
        xlab = "Ship Mode",ylab = "Order Qty.")
boxplot(Inventory$`Order Quantity`~Inventory$`Product Container`, main = "Prod Cont vs Order Qty.",
        xlab = "Product Container",ylab = "Order Qty.")
boxplot(Inventory$`Order Quantity`~Inventory$`Product Sub-Category`, main = "Prod Sub cat vs Order Qty.",
        xlab = "Product Sub-Category",ylab = "Order Qty.")


##Checking multicollinearity
cor.plot(subset(Inventory[,c(1,4)]),numbers = TRUE,xlas=2)
GGally::ggpairs(Inventory[,c(1,4)], mapping = aes(color = Inventory$`Ship Mode`))

#Fixing column names with spaces
names(Inventory)<-str_replace_all(names(Inventory), c(" " = "." , "," = "" , "-" = "" ))
names(Inventory)

Inventory2<-Inventory

#Inventory<-one_hot(as.data.table(Inventory2[,-5,with=FALSE]))
Inventory<-one_hot(as.data.table(Inventory2[,-5]))

names(Inventory)[names(Inventory) == "Product.SubCategory_Chairs & Chairmats"] <- "Product.SubCategory_Chairs and Chairmats"
names(Inventory)[names(Inventory) == "Product.SubCategory_Pens & Art Supplies"] <- "Product.SubCategory_Pens and Art Supplies"
names(Inventory)[names(Inventory) == "Product.SubCategory_Storage & Organization"] <- "Product.SubCategory_Storage and Organization"
names(Inventory)<-str_replace_all(names(Inventory), c(" " = "." , "," = "" , "-" = "" ))

Inventory$Ship.Mode<-Inventory2$Ship.Mode
Inventory$Ship.Mode<-with(Inventory, ifelse(Inventory$Ship.Mode=='Regular Air',1,
                                            ifelse(Inventory$Ship.Mode=='Express Air',2,3)))

cor.plot(subset(Inventory[,1:26]))

setDF(Inventory)
cols<-c(2:25,27)
Inventory[cols]<-lapply(Inventory[cols], factor)

                                       
#########Checking for imbalance data and treating with SMOTE###############
set.seed(1973)
#75:25 ratio data splitting
splitLR = sample.split(Inventory$Ship.Mode, SplitRatio = 0.75)
trainDataLR<-subset(Inventory, splitLR == TRUE)
testDataLR<- subset(Inventory, splitLR == FALSE)
nrow(trainDataLR)
nrow(testDataLR)
prop.table(table(Inventory$Ship.Mode))

prop.table(table(trainDataLR$Ship.Mode))
prop.table(table(testDataLR$Ship.Mode))

#Converting the 3 class problem to two class one for up/downsampling...will remove the 
#additional column later after sampling.
#trainDataLR$Class<-with(trainDataLR, ifelse(trainDataLR$Ship.Mode=='Regular Air',1,0))
trainDataLR$Class<-with(trainDataLR, ifelse(trainDataLR$Ship.Mode==1,1,0))
prop.table(table(trainDataLR$Class))

#combination of over- and under-sampling
#trainDataLR_SMOTE <- ovun.sample(Class ~ ., data = trainDataLR, method = "both", p=0.5,
                                # seed = 1, N=nrow(trainDataLR))$data

trainDataLR_SMOTE <- ovun.sample(Class ~ ., data = trainDataLR, method = "over",p=0.6,seed=1)$data

#trainDataLR_SMOTE <- ovun.sample(Class ~ ., data = trainDataLR, method = "under",p=0.5)$data


#trainDataLR_SMOTE<-ROSE(Class ~ ., data = trainDataLR, seed = 1)$data


prop.table(table(trainDataLR_SMOTE$Class))
prop.table(table(trainDataLR_SMOTE$Ship.Mode))

TwoClassDFCopy<-trainDataLR_SMOTE
TwoClassDFCopyX<-trainDataLR

#Now lets drop the class column to get back to our business
#trainDataLR_SMOTE<-trainDataLR_SMOTE[,-6]
trainDataLR_SMOTE<-trainDataLR_SMOTE[,-28]
trainDataLR<-trainDataLR[,-28]

##trainDataLR_SMOTE<-trainDataLR[,-6]


##MODEL BUILDING
##############################################################################################
#Multinomial Logistic Regression

set.seed(980)
LRModel<-multinom(Ship.Mode~.,data = trainDataLR_SMOTE)
summary(LRModel)

#Predition on train
train.data.predicted = predict(LRModel, newdata=trainDataLR_SMOTE[,-27], "class") 
clf_table<-table(trainDataLR_SMOTE$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##77% #76.76%

#Prediction on test
test.data.predicted = predict(LRModel, newdata=testDataLR[,-27], "class") 
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##88% #88.29

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result

##############################################################################################
##Suppor vector machine
SVMModel <- svm(Ship.Mode~., data=trainDataLR_SMOTE, 
            method="C-classification", kernel="radial",gamma=0.1, cost=10)
summary(SVMModel)
SVMModel$SV

#Predict on train
train.data.predicted = predict(SVMModel, trainDataLR_SMOTE) 
clf_table<-table(trainDataLR_SMOTE$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##77.8%

#Predict on test
test.data.predicted = predict(SVMModel, testDataLR)
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##87.5%

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result
result$byClass

##############################################################################################
##############################################################################################
##Bagging
BagModel<-bagging(trainDataLR$Ship.Mode ~.,
                  data=trainDataLR,
                  control=rpart.control(maxdepth=5, minsplit=15,xval = 5))
summary(BagModel)

#Predict on train
train.data.predicted = predict(BagModel, trainDataLR) 
clf_table<-table(trainDataLR$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##48.9%

#Predict on test
test.data.predicted = predict(BagModel, testDataLR)
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##25.25%

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result


##############################################################################################
##Decision Tree
#define control parameters
r.ctrl <- rpart.control(minsplit = 125, minbucket = 172,cp=0.33,xval = 10)
DTModel <- rpart(Ship.Mode~., data = trainDataLR, method = "class")
print(DTModel)
fancyRpartPlot(DTModel)

#Predict on train
train.data.predicted = predict(DTModel, trainDataLR, type = "class") 
clf_table<-table(trainDataLR$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##78.1%

#Predict on test
test.data.predicted = predict(DTModel, testDataLR, type = "class")
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##88.28%

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result

#####################
#RF
mtry<-tuneRF(trainDataLR_SMOTE[-27],trainDataLR_SMOTE$Ship.Mode,ntreeTry = 300,stepFactor = 1.5,improve = 0.01,
             trace = TRUE,plot = TRUE)
RFModel<-randomForest(Ship.Mode~.,data = trainDataLR_SMOTE, mtry=22, ntree=300,importance=TRUE)
varImpPlot(RFModel)
print(RFModel)

#Predict on train
train.data.predicted = predict(RFModel, trainDataLR_SMOTE, type = "class") 
clf_table<-table(trainDataLR_SMOTE$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##78.1%

#Predict on test
test.data.predicted = predict(RFModel, testDataLR, type = "class")
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##88.28%

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result

#####################RF on unbalanced data
mtry<-tuneRF(trainDataLR[-27],trainDataLR$Ship.Mode,ntreeTry = 300,stepFactor = 1.5,improve = 0.001,
             trace = TRUE,plot = TRUE)
RFModel<-randomForest(Ship.Mode~.,data = trainDataLR, mtry=4, ntree=300,importance=TRUE)
varImpPlot(RFModel)
print(RFModel)

#Predict on train
train.data.predicted = predict(RFModel, trainDataLR, type = "class") 
clf_table<-table(trainDataLR$Ship.Mode,train.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##78.1%

#Predict on test
test.data.predicted = predict(RFModel, testDataLR, type = "class")
clf_table<-table(testDataLR$Ship.Mode,test.data.predicted)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##88.28%

# Compute the confusion matrix and all the statistics
result <- confusionMatrix(test.data.predicted, testDataLR$Ship.Mode, mode="prec_recall")
result




##############################################################################################
# ##KNN
# predKnn9 = knn(trainDataLR_SMOTE[-27], testDataLR[-27], trainDataLR_SMOTE[,27], k = 9)
# tabKnn9 = table(testDataLR[,27], predKnn9)
# tabKnn9
# accKnn9=sum(diag(tabKnn9)/sum(tabKnn9))
# accKnn9
# #Confusion Matrix
# confusionMatrix(predKnn9,testDataLR$Ship.Mode, mode = "prec_recall")
###########################
###Gradient Boosting
GBMModel<-gbm(Ship.Mode ~ ., distribution="multinomial", data=trainDataLR_SMOTE,
                          n.trees=10000, shrinkage=0.001, cv.folds=5,
                          verbose=FALSE, n.cores=NULL)
summary(GBMModel)

#Predict on train
train.data.predicted = predict(GBMModel, trainDataLR_SMOTE[-27])
trainDataLR_SMOTE$pred.Boost <- apply(train.data.predicted, 1, which.max)
clf_table<-table(trainDataLR_SMOTE$Ship.Mode,trainDataLR_SMOTE$pred.Boost)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##76.7%

#Predict on test
test.data.predicted = predict(GBMModel, testDataLR[-27])
testDataLR$pred.Boost <- apply(test.data.predicted, 1, which.max)
clf_table<-table(testDataLR$Ship.Mode,testDataLR$pred.Boost)
clf_table
round((sum(diag(clf_table))/sum(clf_table))*100,2) ##88%

testDataLR$pred.Boost<-as.factor(testDataLR$pred.Boost)
# Compute the confusion matrix and all the statistics
result <- confusionMatrix(testDataLR$pred.Boost, testDataLR$Ship.Mode, mode="prec_recall")
result


