# Install and Load Required Libraries
install.packages("rstudioapi")
library(rstudioapi)
install.packages("ggplot2")
library(ggplot2)
install.packages("lattice")
library(lattice)
install.packages("caret")
library(caret)
install.packages("factoextra")
library(factoextra)
install.packages("multiROC")
library(multiROC)
# Set Working Directory and Load Data
current_path=rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
winequality.white <- read.csv("winequality-white.dat", comment.char="#")
# set random seed for reproducibility
set.seed(55)
# Data Cleaning and Overview
summary(winequality.white)
sum(is.na.data.frame(winequality.white))
sum(duplicated(winequality.white))
winequality.white <- unique.data.frame(winequality.white)
plot(factor(winequality.white$Quality), col = rainbow(7), xlab = "Quality", 
     ylab = "Frequency", main = "Original Quality Distribution")
# Rearranging "Quality" variable
winequality.white$Quality[winequality.white$Quality == "3"] <- 1
winequality.white$Quality[winequality.white$Quality == "4"] <- 1
winequality.white$Quality[winequality.white$Quality == "5"] <- 1
winequality.white$Quality[winequality.white$Quality == "6"] <- 2
winequality.white$Quality[winequality.white$Quality == "7"] <- 3
winequality.white$Quality[winequality.white$Quality == "8"] <- 3
winequality.white$Quality[winequality.white$Quality == "9"] <- 3
# New overview
plot(factor(winequality.white$Quality), col = rainbow(7), xlab = "Quality", 
     ylab = "Frequency", main = "Our Quality Distribution")
winequality.white$Quality <- factor(winequality.white$Quality)
summary(winequality.white$Quality)
# Boxplot
par(mfrow=c(3,4)) 
for(i in 1:11) {
  boxplot(winequality.white[,i], main=names(winequality.white[i])) }
pie(table(winequality.white$Quality), col = rainbow(7), main="Quality")
for(i in 1:11) {
  boxplot(winequality.white[,i] ~ winequality.white$Quality, ylab=NULL, xlab=NULL,
          main=names(winequality.white)[i]) }
pie(table(winequality.white$Quality), col = rainbow(7), main="Quality")

featurePlot(x=winequality.white[,1:11], y=winequality.white$Quality, plot="density", 
            scales=list(x=list(relation="free"), y=list(relation="free")),
            auto.key=list(columns=3))

# Principal Component Analysis


winequality.white[1:11] <- scale(winequality.white[1:11])
prin_comp<- prcomp(winequality.white[1:11])
get_eigenvalue(prin_comp)
data1 <-data.frame(Quality = winequality.white$Quality, prin_comp$x)
data1 <- data.frame(data1[1:6])
featurePlot(x=data1[,2:6], y=data1$Quality, plot="density", auto.key=list(columns=3),
            scales=list(x=list(relation="free"), y=list(relation="free")))
featurePlot(x=data1[1:500,2:6], y=data1$Quality[1:500], plot="pairs", auto.key=list(columns=3))
# Train-Test split
ind = sample(2, nrow(data1), replace = TRUE, prob=c(0.7, 0.3))
trainset = data1[ind == 1,]
testset = data1[ind == 2,]
# Training ML models
control = trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       classProbs = TRUE, summaryFunction = multiClassSummary) 

levels(trainset$Quality) <- c("Q1","Q2","Q3")

rete= train(Quality ~., data = trainset, method="nnet", metric="AUC",
            trace = FALSE, nmax=10000, trControl = control)

svm2= train(Quality ~ ., data = trainset, method = "svmRadial",
            metric="AUC", trControl = control)
# OPTIONAL: training a polynomial kernel SVM
# svm= train(Quality ~ ., data = trainset, method = "svmPoly", degree=3,
#            metric="AUC", trControl = control, nmax=1000)

#  Cross-validation and Comparison of Models
cv.values = resamples(list(net = rete, svm = svm2))
summary(cv.values)
View(svm2$finalModel)
View(rete$finalModel)
dotplot(cv.values, metric = "AUC")
# Testing model prediction
pred.svm2=predict(svm2,testset[, !names(testset) %in% c("Quality")],type = "prob")

pred.rete = predict(rete,testset[, !names(testset) %in% c("Quality")],type = "prob") 

data.test <- data.frame(testset$Quality == "1")
data.test[,2] <- data.frame(testset$Quality == "2")
data.test[,3] <- data.frame(testset$Quality == "3")
data.test <- data.frame(data.test,pred.rete)
data.test <- data.frame(data.test, pred.svm2)
colnames(data.test)<-c("Q1_true","Q2_true","Q3_true","Q1_pred_net","Q2_pred_net","Q3_pred_net","Q1_pred_svm","Q2_pred_svm","Q3_pred_svm")

res <- multi_roc(data.test, force_diag=T) 
n_method <- length(unique(res$Methods))
n_group <- length(unique(res$Groups))
res_df <- data.frame(Specificity= numeric(0), Sensitivity= numeric(0), 
                     Group = character(0), AUC = numeric(0), Method = character(0))

for (i in 1:n_method) {
  for (j in 1:n_group) {
    temp_data_1 <- data.frame(Specificity=res$Specificity[[i]][j],
                              Sensitivity=res$Sensitivity[[i]][j],
                              Group=unique(res$Groups)[j],
                              AUC=res$AUC[[i]][j],
                              Method = unique(res$Methods)[i])
    colnames(temp_data_1) <- c("Specificity", "Sensitivity", "Group", "AUC", "Method")
    res_df <- rbind(res_df, temp_data_1)
  }
}


levels(testset$Quality) <- c("Q1","Q2","Q3")

pred.svm2=predict(svm2,testset[, !names(testset) %in% c("Quality")]) 

pred.rete = predict(rete,testset[, !names(testset) %in% c("Quality")]) 
# Confusion Matrixes
confusionMatrix(pred.rete,testset[,c("Quality")], mode = "prec_recall")

confusionMatrix(pred.svm2,testset[,c("Quality")], mode = "prec_recall")
# Plotting ROC curves
ggplot2::ggplot(res_df, ggplot2::aes(x = 1-Specificity, y=Sensitivity)) + 
  ggplot2::geom_path(ggplot2::aes(color = Group, linetype=Method)) + 
  ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1), 
                        colour='grey', linetype = 'dotdash') + ggplot2::theme_bw() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), 
                 legend.justification=c(1, 0), legend.position=c(.95, .05), 
                 legend.title=ggplot2::element_blank(),
                 legend.background = ggplot2::element_rect(fill=NULL, size=0.5,
                                                           linetype="solid", colour ="black"))
