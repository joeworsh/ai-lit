library(jsonlite)
library(tm)
library(magrittr)
library(plyr)
library(dplyr)
library(class)

# import helper methods for parsing GB data
source("gb_input_util.R")

ai_lit_dir <- file.path("C:", "workspaces", "python", "ai_lit")
dataset_wkspc <- file.path(ai_lit_dir, "workspace", "gb_input")
train_json_idx <- file.path(dataset_wkspc, "train_index_list.json")
test_json_idx <- file.path(dataset_wkspc, "test_index_list.json")
sbjs <- loadSubjects(dataset_wkspc)

train_data <- loadData(train_json_idx, sbjs)
train_classes <- train_data$data
train_corpus <- train_data$corpus

test_data <- loadData(test_json_idx, sbjs)
test_classes <- test_data$data
test_corpus <- test_data$corpus

merged_classes <- c(train_classes, test_classes)
merged_corpus <- c(train_corpus, test_corpus)
merged_dtm <- DocumentTermMatrix(merged_corpus)
merged_dtm <- removeSparseTerms(merged_dtm, 0.8)

# Transform dtm to matrix to data frame - df is easier to work with
merged.corpus.df <- as.data.frame(data.matrix(merged_dtm), stringsAsfactors = FALSE)

# Column bind category (known classification)
factorized <- unclass(factor(unlist(merged_classes)))
merged.corpus.df <- cbind(merged.corpus.df, factorized)

# knn needs even frames, so must resample train and test
train <- sample(nrow(merged.corpus.df), ceiling(nrow(merged.corpus.df) * .50))
test <- (1:nrow(merged.corpus.df))[- merged.train.df]

# Create model: training set, test set, training set classifier
knn.pred <- knn(merged.corpus.df[train, -ncol(merged.corpus.df)], merged.corpus.df[test, -ncol(merged.corpus.df)], merged.corpus.df[train, ncol(merged.corpus.df)])

# Confusion matrix
conf.mat <- table("Predictions" = knn.pred, Actual = merged.corpus.df[test, ncol(merged.corpus.df)])
conf.mat
(accuracy <- sum(diag(conf.mat))/length(test) * 100)
