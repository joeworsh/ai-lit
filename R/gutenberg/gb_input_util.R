library(jsonlite)
library(tm)

# a function for loading the set of subjects from the ai_lit subjects.json file
subject_file <- 'subjects.json'
loadSubjects <- function(dataset_wkspc){
  subject_json <- file.path(dataset_wkspc, subject_file)
  sbjs <- fromJSON(subject_json, flatten = TRUE)
}

# a function for loading and processing a JSON ai_lit gutenberg dataset.
loadData <- function(file, sbjs){
  data <- fromJSON(readLines(file), flatten = TRUE)
  #data %>% mutate(atomic_subject = getAtomicSubject(subjects, sbjs))
  #data$atomic_subject <- sbjs[match(data$subjects, sbjs)]
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
