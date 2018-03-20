# QUESTION & ANSWER

# IEMS 308 
# Renee Probetts 
# This is the second part in the Text Mining Assignment 

#install and attach packages 
install.packages('tm')
install.packages(c("NLP", "openNLP", "magrittr","RWeka","qdap","openNLPdata"))
install.packages("openNLPmodels.en", dependencies = TRUE,repos = "http://datacube.wu.ac.at/")
install.packages("rJava", type = "source")
install.packages("stringr", dependencies = TRUE)
install.packages("quanteda")
install.packages("dplyr")
install.packages("solrium")
install.packages("tidytext")
install.packages("corpus")
library('tm')
library('dplyr')
library("tidyr")
library("quanteda")
library(NLP)
library(openNLP)
library(stringr)
library(openNLPmodels.en)
library(tidytext)
library(corpus)

# Read the question from the txt file 
user.question <-readline(prompt = "What would you like to know: " )
who.question <- grepl("[Ww]ho[m]?",user.question)
which.question <- grepl("[Ww]hich",user.question)
what.question <- grepl("[Ww]hat",user.question) 

# download the text files and create the corpus - used for answer the questions  
pathname2013 <- file.path("D:", "Profiles", "rsp714", "Documents","TextAnalytics", "2013")
dir(pathname2013) # make sure these are the correct text files
pathname2014 <- file.path("D:", "Profiles", "rsp714", "Documents","TextAnalytics", "2014")
dir(pathname2014) # make sure these are the correct text files 
tmcorpus <- Corpus(DirSource(c(pathname2013,pathname2014)))
qcorpus <- corpus(tmcorpus)
tmcorpus <- tm_map(tmcorpus, content_transformer(tolower))
tmcorpus <- tm_map(tmcorpus,removeWords,stopwords("english"))

# determine keywords if it is a "Who" question - looking for CEO 
if (who.question == TRUE) {
  # extract keywords 
  question <- as.String(user.question)
  word_ann <- Maxent_Word_Token_Annotator(language = "en")
  org_ann <- Maxent_Entity_Annotator(kind = "organization")
  question_annotated<-annotate(question,list(word_ann,org_ann))
  CEO_needed <- lapply(question_annotated,entities,kind = "organization")
  keywords <-c("CEO",CEO_needed)}


# determine keywords if it is a "Which" question - looking for companies, month, year, bankrupt 
if (which.question == TRUE){
  #extract keywords 
  question <- lapply(user.question,as.String)
  question <- tm_map(question,removeWords,stopwords('english'))
  word_ann <- Maxent_Word_Token_Annotator(language = "en")
  date_ann <- Maxent_Entity_Annotator(kind="date")
  pos_ann <- Maxent_POS_Tag_Annotator()
  pipeline_used <- list(word_ann,date_ann,pos_ann)
  question_annotated <- question %>% 
    lapply(annotate_entities,pipeline_used)
  date_needed <- lapply(question_annotated,entities,kind = "date")
  #remove key nouns and adjectives 
  keynouns <- strsplit(unlist(question_annotated[1]),'/NN')
  keynouns <- tail(strsplit(unlist(keynouns[1])," ")[[1]],1)
  keyadjs <- strsplit(unlist(question_annotated[1]),'/JJ')
  keyadjs <- tail(strsplit(unlist(keyadjs[1])," ")[[1]],1)
  
  keywords <- c(keynouns,keyadjs,date_needed) #figure out how to remove certain parts of speech 
}

# determine keywords if it is a "What" question - looking for things that affect GDP - requires follow up question 
if (what.question == TRUE){
  #extract keywords 
  question <- lapply(user.question,as.String)
  question <- tm_map(question, removeWords,stopwords('english'))
  word_ann <- Maxent_Word_Token_Annotator()
  pos_ann <- Maxent_POS_Tag_Annotator()
  pipeline_used <- list(word_ann,pos_ann)
  question_annotated <- question %>% 
    lapply(annotate_entities,pipeline_used)
  keyverbs <- strsplit(unlist(question_annotated[1]),'/VB')
  keyverbs <- tail(strsplit(unlist(keyverbs[1])," ")[[1]],1)
  keynouns <- strsplit(unlist(question_annotated[1]),'/NN')
  keynouns <- tail(strsplit(unlist(keynouns[1])," ")[[1]],1)
  keywords <- c(keyverbs,keynouns)
}

# search for the documents that have the keywords in them 
keyword_documents <- kwic(qcorpus,keywords[1],valuetype = "fixed")
for (i in 2: length(keywords)){
  keyword_documents <- kwic(qcorpus,keywords[i],valuetype = "fixed")
  newkeyword_documents <-data.frame(keyword_documents)
  
  # only use the documents that contain all keywords 
  keyword_documents <- intersect(keyword_documents,newkeyword_documents)
}

# put list of documents into a dataframe 
final.list.docs <- data.frame(keyword_documents)


# loop through the documents and assign a score based on the frequency of keywords 
score <- data.frame(matrix(ncol = 1,nrow= nrow(final.list.docs)))
index <- data.frame(matrix(ncol = 1,nrow= nrow(final.list.docs))) 
weighted.score <-data.frame(matrix(ncol = 1,nrow= nrow(final.list.docs)))

for (i in 1:nrow(final.list.docs)){
  score[i,] = 0
  text <- final.list.docs[i,]
  index[i,] <- as.character(text[1])
  dtm <- DocumentTermMatrix(tmcorpus[names(tmcorpus)==index[i,]])
  freq <- colSums(as.matrix(dtm))
  wf <- data.frame(word = names(freq),freq=freq)
  # loop through the keywords and assign a score 
  for (j in 1:length(keywords)){
  keyterms <- wf[wf$word == keywords[j],]
  score[i,] = score[i,] + keyterms$freq
  }
  
  # calculate the weighted score based on the weighted document term matrix 
  weighted.score[i,] = 0
  weighted_dtm <- weightTfIdf(wtf,normalize = TRUE)
  freq <- colSums(as.matrix(weighted_dtm))
  wf <- data.frame(word = names(freq),freq=freq)
  # loop through the keywords and assign a score 
  for (j in 1:length(keywords)){
    keyterms <- wf[wf$word == keywords[j],]
    weighted.score[i,] = weighted.score[i,] + keyterms$freq
  }
}

# create a data frame of the documents and their scores 
index_score <- data.frame(index,score,weighted.score)
colnames(index_score)<-c("document","score","weightedscore")
index_score <- index_score[order(-index_score$weightedscore),]

# select the top 5% of documents to search in for the answer
fivepercent <- 0.5*nrow(index_score)
top.doc <- index_score[1:fivepercent,]
colnames(top.doc)<-c("document","score","weightedscore")

# create a corpus with these selected documents and segment into sentences  
selectcorpus <- tmcorpus[names(tmcorpus)==top.doc$document]
word_ann <- Maxent_Word_Token_Annotator()
sent_ann <-Maxent_Sent_Token_Annotator()
select_annotations <- annotate(selectcorpus,list(word_ann,sent_ann))
sentencecorpus <- corpus(select_annotations)

# select the sentences that have all keywords
keyword_sentences <- kwic(sentencecorpus,keywords[1],valuetype = "fixed")
for (i in 2:length(keywords)){
  keyword_sentences <- kwic(sentencecorpus,keywords[i],valuetype = "fixed")
  newkeyword_sent <-data.frame(keyword_documents)
  keyword_sentences <- intersect(keyword_sentences,newkeyword_sent)
}

list.sentences <- data.frame(keyword_sentences)

score <- data.frame(matrix(ncol = 1,nrow= nrow(list.sentences)))
index <- data.frame(matrix(ncol = 1,nrow= nrow(list.sentences)))

# score the sentences using the term frequency, similarly to the documents 
for (i in 1:length(list.sentences)){
  score[i,] = 0
  text <- final.list.docs[i,]
  index[i,] <- as.character(text[1])
  dtm <- DocumentTermMatrix(sentencecorpus[names(sentencecorpus)==index[i,]])
  freq <- colSums(as.matrix(dtm))
  wf <- data.frame(word = names(freq),freq=freq)
  # loop through the keywords and assign a score 
  for (j in 1:length(keywords)){
    keyterms <- wf[wf$word == keywords[j],]
    score[i,] = score[i,] + keyterms$freq
  }
}

sent_score <- data.frame(index,score)
colnames(sent_score)<-c('sentence','score')
sent_score <- sent_score[order(-sent_score$sentence),]

# the best sentence has the highest score 
best.sent<- sentencecorpus[[1,1]]

# using NER, return the proper entity-word of the highest scoring sentence
if (who.question == TRUE){
  per_ann <- Maxent_Entity_Annotator(kind ="person")
  best.sent<- annotate(best.sent,per_ann)
  answer <- lapply(question_annotated,entities,kind = "person")
  print(answer)
}

if (which.question == TRUE){
  org_ann <- Maxent_Entity_Annotator(kind = "organization")
  best.sent <- annotate(best.sent,pos_ann)
  answer <- lapply(question_annotated,entities, kind = "organization")
  print(answer)
}

# if it is a what question, there will be a follow up question allowed 
if (what.question == TRUE){
  best.sent <- annotate(best.sent,pos_ann)
  answer <- strsplit(unlist(question_annotated[1]),'/NN')
  answer <- tail(strsplit(unlist(answer[1])," ")[[1]],1)
  print(answer)
  followup <-readline(prompt = "What more would you like to know: " )
}

# classify  the follow-up question and search the top documents for the word
followup <- annotate(followup,pos_ann)
property <- strsplit(unlist(question_annotated[1]),'/NN')
property <- tail(strsplit(unlist(property[1])," ")[[1]],1)

# search the top sentences for the answer to the follow up question 
potential_answer <- str_extract_all(sentencecorpus,regex(property,multiline = TRUE))

# extract the percent value using regular expressions 
answer <- grep("[:digit:]%|[:digit:]percent|[a-z]+[ ]?percent",potential_answer)
print(answer)


