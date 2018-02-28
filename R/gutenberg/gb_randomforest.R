library(caret)
library(randomForest)
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
model_name <- "gb_bow"
model_dir <- file.path(workspace, model_name)
if (!file.exists(model_dir)){
  dir.create(model_dir)
}

# load the input for this experiment
dataset_wkspc <- file.path(workspace, "gb_input")
loadGbDatasets(dataset_wkspc)

# train and test split
train <- sample(nrow(merged.corpus.df), ceiling(nrow(merged.corpus.df) * .70))
test <- (1:nrow(merged.corpus.df))[- train]

# random forest requires factorization
merged.corpus.df[, ncol(merged.corpus.df)] <- factor(merged.corpus.df[, ncol(merged.corpus.df)])

# evaluationt: random forest
experiment_name <- "random_forest"
eval_dir <- file.path(model_dir, experiment_name)
eval_vars <- file.path(eval_dir, paste(experiment_name, ".RData", sep=''))

# run or load the kNN standard evaluation
if (file.exists(eval_dir) && file.exists(eval_vars)){
  load(eval_vars, .GlobalEnv)
}else{
  # Create model: training set, test set, training set classifier
  output.forest <- randomForest(x=merged.corpus.df[train, -ncol(merged.corpus.df)], y=merged.corpus.df[train, ncol(merged.corpus.df)])
  pred.rf <- predict(object = output.forest, merged.corpus.df[test, -ncol(merged.corpus.df)])
  
  # Confusion matrix
  conf.mat <- table("Predictions" = pred.rf, Actual = merged.corpus.df[test, ncol(merged.corpus.df)])
  accuracy <- sum(diag(conf.mat))/length(test) * 100
  caret.conf.mat <- confusionMatrix(conf.mat, mode = "prec_recall")
  p <- sum(caret.conf.mat$byClass[,5]) / 6
  r <- sum(caret.conf.mat$byClass[,6]) / 6
  f1 <- (2 * p * r) / (p + r)
  
  dir.create(eval_dir)
  save(output.forest, pred.rf, accuracy, caret.conf.mat, p, r, f1, file=eval_vars)
}

accuracy
caret.conf.mat
p
r
f1
