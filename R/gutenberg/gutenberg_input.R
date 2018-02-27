library(caret)
library(class)
library(dplyr)
library(jsonlite)
library(magrittr)
library(plyr)
library(tm)

# import helper methods for parsing GB data
source("gb_input_util.R")

# set up the folders for this experiment
ai_lit_dir <- dirname(dirname(getwd()))
workspace <- file.path(ai_lit_dir, "workspace")
experiment <- "gb_knn"
exp_dir <- file.path(workspace, experiment)
if (!file.exists(exp_dir)){
  dir.create(exp_dir)
}

# load the input for this experiment
dataset_wkspc <- file.path(workspace, "gb_input")
loadGbDatasets(dataset_wkspc)

# knn needs even frames, so must resample train and test
train <- sample(nrow(merged.corpus.df), ceiling(nrow(merged.corpus.df) * .50))
test <- (1:nrow(merged.corpus.df))[- train]

# evaluationt 1: default kNN
default_eval <- "default_knn"
eval_dir <- file.path(exp_dir, default_eval)
eval_vars <- file.path(eval_dir, paste(default_eval, ".RData", sep=''))

# run or load the kNN standard evaluation
if (file.exists(eval_dir) && file.exists(eval_vars)){
  load(eval_vars, .GlobalEnv)
}else{
  # Create model: training set, test set, training set classifier
  knn.pred <- knn(merged.corpus.df[train, -ncol(merged.corpus.df)], merged.corpus.df[test, -ncol(merged.corpus.df)], merged.corpus.df[train, ncol(merged.corpus.df)])
  
  # Confusion matrix
  conf.mat <- table("Predictions" = knn.pred, Actual = merged.corpus.df[test, ncol(merged.corpus.df)])
  conf.mat
  accuracy <- sum(diag(conf.mat))/length(test) * 100
  caret.conf.mat <- confusionMatrix(conf.mat, mode = "prec_recall")
  p <- sum(caret.conf.mat$byClass[,5]) / 6
  r <- sum(caret.conf.mat$byClass[,6]) / 6
  f1 <- (2 * p * r) / (p + r)
  
  dir.create(eval_dir)
  save(knn.pred, conf.mat, accuracy, caret.conf.mat, p, r, f1, file=eval_vars)
}

accuracy
caret.conf.mat
p
r
f1
