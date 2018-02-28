library(caret)
library(dplyr)
library(jsonlite)
library(magrittr)
library(xgboost)
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
numberOfClasses <- 6
nround <- 50
xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = numberOfClasses)

# knn needs even frames, so must resample train and test
train <- sample(nrow(merged.corpus.df), ceiling(nrow(merged.corpus.df) * .70))
test <- (1:nrow(merged.corpus.df))[- train]

# evaluationt 1: naive bayes classifier
experiment_name <- "xgboost"
eval_dir <- file.path(model_dir, experiment_name)
eval_vars <- file.path(eval_dir, paste(experiment_name, ".RData", sep=''))

# run or load the kNN standard evaluation
if (file.exists(eval_dir) && file.exists(eval_vars)){
  load(eval_vars, .GlobalEnv)
}else{
  # need to format the data for the XGBoost alg
  ucl <- unclass(merged.corpus.df[, ncol(merged.corpus.df)])
  ucl <- ucl - 1
  train_matrix <- xgb.DMatrix(data = as.matrix(merged.corpus.df[train, -ncol(merged.corpus.df)]), label = ucl[train])
  test_matrix <- xgb.DMatrix(data = as.matrix(merged.corpus.df[test, -ncol(merged.corpus.df)]), label = ucl[test])
  
  bst <- xgb.train(params = xgb_params, data = train_matrix, nrounds = nround)
  pred.xgboost <- predict(bst, newdata = test_matrix)
  test_prediction <- matrix(pred.xgboost, nrow = numberOfClasses,
                            ncol=length(pred.xgboost)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate(label = ucl[test] + 1,
           max_prob = max.col(., "last"))
  conf.xgb.mat <- confusionMatrix(factor(test_prediction$label),
                                   factor(test_prediction$max_prob),
                                   mode = "everything")
  accuracy <- as.numeric(conf.xgb.mat$overall['Accuracy']) * 100
  p <- sum(conf.xgb.mat$byClass[,5]) / 6
  r <- sum(conf.xgb.mat$byClass[,6]) / 6
  f1 <- (2 * p * r) / (p + r)
  
  dir.create(eval_dir)
  save(bst, pred.xgboost, test_prediction, accuracy, conf.xgb.mat, p, r, f1, file=eval_vars)
}

accuracy
conf.xgb.mat
p
r
f1
