library(caret)
library(randomForest)
library(dplyr)
library(jsonlite)
library(magrittr)
library(plyr)
library(tm)

# tools for rendering dataset exploration
library(ggplot2)
library(wordcloud)

# import helper methods for parsing GB data
source("gb_input_util.R")

# set up the folders for this experiment
ai_lit_dir <- dirname(dirname(getwd()))
workspace <- file.path(ai_lit_dir, "workspace")

# load the input for this experiment
dataset_wkspc <- file.path(workspace, "gb_input")
loadGbDatasets(dataset_wkspc)

# split the dataframe into smaller dataframes (one for each subject header)
split_df <- split(merged.corpus.df, merged.corpus.df[,ncol(merged.corpus.df)])

# Compute the frequencies of the tokens in the corpus
freq <- colSums(as.matrix(merged_dtm))
ord <- order(freq)
most_freq_wf <- data.frame(word=names(freq), freq=freq)

# Least Common Terms
freq[head(ord, 15)]

# Most Common Terms
freq[tail(ord, 15)]

# Plot the most frequent words
sorted_freq_wf <- most_freq_wf[with(most_freq_wf, order(-freq)), ]
most_freq_plot <- ggplot(sorted_freq_wf[1:25,], aes(word, freq))
most_freq_plot <- most_freq_plot + geom_bar(stat="identity")
most_freq_plot <- most_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))

# Plot the next most frequent words (no stop words)
next_freq_wf <- most_freq_wf[with(most_freq_wf, order(-freq)), ]
next_freq_plot <- ggplot(next_freq_wf[50:75,], aes(word, freq))
next_freq_plot <- next_freq_plot + geom_bar(stat="identity")
next_freq_plot <- next_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))

# Plot the least frequent words
least_freq_wf <- most_freq_wf[with(most_freq_wf, order(freq)), ]
least_freq_plot <- ggplot(least_freq_wf[1:25,], aes(word, freq))
least_freq_plot <- least_freq_plot + geom_bar(stat="identity")
least_freq_plot <- least_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))

# Build a wordcloud for the most frequent words
#dark2 <- brewer.pal(6, "Dark2")
#wordcloud(names(freq), freq, max.words=50, rot.per=0.2, colors=dark2) 

# extract the most and least common terms for the individual subject categories
most_freq_sbj_plots <- vector("list", 6)
next_freq_sbj_plots <- vector("list", 6)
least_freq_sbj_plots <- vector("list", 6)
for (sbj_factor in 1:length(split_df)) {
  sbj_split_df = split_df[[sbj_factor]][, -ncol(merged.corpus.df)]
  
  freq <- colSums(as.matrix(sbj_split_df))
  ord <- order(freq)
  most_freq_wf <- data.frame(word=names(freq), freq=freq)
  
  # Plot the most frequent words
  sorted_freq_wf <- most_freq_wf[with(most_freq_wf, order(-freq)), ]
  most_freq_plot <- ggplot(sorted_freq_wf[1:25,], aes(word, freq))
  most_freq_plot <- most_freq_plot + geom_bar(stat="identity")
  most_freq_plot <- most_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))
  most_freq_sbj_plots[[sbj_factor]] <- most_freq_plot
  
  # Plot the next most frequent words (no stop words)
  next_freq_wf <- most_freq_wf[with(most_freq_wf, order(-freq)), ]
  next_freq_plot <- ggplot(next_freq_wf[100:125,], aes(word, freq))
  next_freq_plot <- next_freq_plot + geom_bar(stat="identity")
  next_freq_plot <- next_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))
  next_freq_sbj_plots[[sbj_factor]] <- next_freq_plot
  
  # Plot the least frequent words
  least_freq_wf <- most_freq_wf[with(most_freq_wf, order(freq)), ]
  least_freq_plot <- ggplot(least_freq_wf[1:25,], aes(word, freq))
  least_freq_plot <- least_freq_plot + geom_bar(stat="identity")
  least_freq_plot <- least_freq_plot + theme(axis.text.x=element_text(angle=45, hjust=1))
  least_freq_sbj_plots[[sbj_factor]] <- least_freq_plot
}

# Weight the data with tf-idf and find the most informative terms
tfidf = weightTfIdf(merged_dtm)
weights <- colSums(as.matrix(tfidf))
weights_ord <- order(weights)
weighted_wf <- data.frame(word=names(weights), weight=weights)

# Highest weighted Terms
weights[tail(weights_ord, 15)]

# Plot the highest weighted words
sorted_weight_wf <- weighted_wf[with(weighted_wf, order(-weight)), ]
weighted_plot <- ggplot(sorted_weight_wf[1:25,], aes(word, weight))
weighted_plot <- weighted_plot + geom_bar(stat="identity")
weighted_plot <- weighted_plot + theme(axis.text.x=element_text(angle=45, hjust=1))

# Build a wordcloud for the highest weighted words
#dark2 <- brewer.pal(6, "Dark2")
#wordcloud(names(weights), weights, max.words=50, rot.per=0.2, colors=dark2) 

genre_names <- vector("list", 6)
genre_names[[1]] <- "Adventure"
genre_names[[2]] <- "Detective"
genre_names[[3]] <- "Historical"
genre_names[[4]] <- "Love"
genre_names[[5]] <- "SciFi"
genre_names[[6]] <- "Western"

# combine all the genres into individual BOWs to find a cross-genre weighting scheme
combined_genres <- vector("list", 6)
for (sbj_factor in 1:length(split_df)) {
  sbj_split_df = split_df[[sbj_factor]][, -ncol(merged.corpus.df)]
  #sbj_merged_df <- data.frame(colSums(sbj_split_df))
  #sbj_merged_df <- colSums(sbj_split_df)
  sbj_merged_df <- data.frame(t(colSums(sbj_split_df)))
  colnames(sbj_merged_df) <- colnames(sbj_split_df)
  combined_genres[[sbj_factor]] <- sbj_merged_df
}
combined_genres_df <- data.frame(matrix(unlist(combined_genres), nrow=6, byrow=T))
colnames(combined_genres_df) <- colnames(sbj_split_df)
combined_genres_tfidf <- as.DocumentTermMatrix(combined_genres_df, weighting = weightTf)

library(stats)
library(ggbiplot)
rownames(combined_genres_df) <- genre_names
pca <- prcomp(combined_genres_df[,-ncol(combined_genres_df)], center = TRUE, scale = TRUE)
#ggbiplot(pca, ellipse=TRUE, obs.scale = 1, var.scale = 1, var.axes = FALSE, groups = genre_names)
factor_groups <- factor(unlist(genre_names))
ggbiplot(pca, obs.scale = 1, var.scale = 1, var.axes = FALSE,
         #labels = rownames(combined_genres_df),
         ellipse = TRUE, circle = TRUE) +
         scale_color_discrete(name = '') +
         theme(legend.direction = 'horizontal', legend.position = 'top')
pca$rotation[tail(order(abs(pca$rotation[,1])), 50), 1]

weights <- colSums(as.matrix(combined_genres_tfidf))
weights_ord <- order(weights)
weighted_wf <- data.frame(word=names(weights), weight=weights)
sorted_weight_wf <- weighted_wf[with(weighted_wf, order(-weight)), ]
weighted_plot <- ggplot(sorted_weight_wf[1:25,], aes(word, weight))
weighted_plot <- weighted_plot + geom_bar(stat="identity")
weighted_plot <- weighted_plot + theme(axis.text.x=element_text(angle=45, hjust=1))

library(tsne)
colors = rainbow(6)
names(colors) = genre_names
ecb = function(x,y){ plot(x,t='n'); text(x,labels=merged.corpus.df[,ncol(merged.corpus.df)], col=colors[merged.corpus.df[,ncol(merged.corpus.df)]]) }
tsne_corpus = tsne(merged.corpus.df[, -ncol(merged.corpus.df)], epoch_callback = ecb, perplexity=50)
