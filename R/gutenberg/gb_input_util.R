library(jsonlite)
library(tm)

# a function for loading and parsing the GB dataset into an R script
# the first time this is run, it will be saved to the dataset workspace
# all subsequent runs will simply load the workspace from disk
train_idx_json <- "train_index_list.json"
test_idx_json <- "test_index_list.json"
gb_wkspc_rdata <- "gutenberg_dataset.RData"
loadGbDatasets <- function(dataset_wkspc){
  wkspc <- file.path(dataset_wkspc, gb_wkspc_rdata)
  if (file.exists(wkspc)){
    load(wkspc, .GlobalEnv)
  }
  else {
    train_json_idx <<- file.path(dataset_wkspc, train_idx_json)
    test_json_idx <<- file.path(dataset_wkspc, test_idx_json)
    sbjs <<- loadSubjects(dataset_wkspc)
    
    train_data <- loadData(train_json_idx, sbjs)
    train_classes <<- train_data$data
    train_corpus <<- train_data$corpus
    
    test_data <- loadData(test_json_idx, sbjs)
    test_classes <<- test_data$data
    test_corpus <<- test_data$corpus
    
    merged_classes <<- c(train_classes, test_classes)
    merged_corpus <<- c(train_corpus, test_corpus)
    merged_dtm <<- DocumentTermMatrix(merged_corpus)
    merged_dtm <<- removeSparseTerms(merged_dtm, 0.8)
    
    # Transform dtm to matrix to data frame - df is easier to work with
    merged.corpus.df <<- as.data.frame(data.matrix(merged_dtm), stringsAsfactors = FALSE)
    
    # Column bind category (known classification)
    factorized <<- unclass(factor(unlist(merged_classes)))
    merged.corpus.df <<- cbind(merged.corpus.df, factorized)
    
    save(train_json_idx, test_json_idx, sbjs, train_classes, train_corpus, test_classes, test_corpus, merged_classes, merged_corpus, merged_dtm, merged.corpus.df, factorized, file=wkspc)
  }
}

# a function for loading the set of subjects from the ai_lit subjects.json file
subject_file <- 'subjects.json'
loadSubjects <- function(dataset_wkspc){
  subject_json <- file.path(dataset_wkspc, subject_file)
  sbjs <- fromJSON(subject_json, flatten = TRUE)
}

# a function for loading and processing a JSON ai_lit gutenberg dataset.
loadData <- function(file, sbjs){
  data <- fromJSON(readLines(file), flatten = TRUE)
  data$atomic_subject <- getAtomicSubjects(data$subjects, sbjs)
  corpus <- VCorpus(VectorSource(data$body))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, function(x) gsub("\r\n", " ", x))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, PlainTextDocument)
  return(list("data" = data$atomic_subject, "corpus" = corpus))
}

# a function to return the atomic subjects of a dataframe which live in the intersection of the two provided subject lists
getAtomicSubjects <- function(gsubs, subjects){
  l <- list()
  i <- 1
  for (doc in gsubs){
    g <- NA
    for (gsub in doc){
      if (gsub %in% subjects){
        g <- gsub
        break
      }
    }
    l[[i]] <- g
    i <- i + 1
  }
  return(l)
}
