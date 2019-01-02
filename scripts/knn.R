setwd("~/Desktop/DC/contest/Data")
library(DMwR)
Mydata = read.csv('categorical_data.csv', header = TRUE)
knnOutput <- knnImputation(data=Mydata, k=10)
xcomplete <- Mydata[setdiff(1:nrow(Mydata),which(!complete.cases(Mydata))),]
#if (nrow(xcomplete) < k)
#  stop("Not sufficient complete cases for computing neighbors.")