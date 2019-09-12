#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging, codecs,os,re


# In[ ]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()


# In[ ]:


import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
from pprint import pprint


# In[ ]:


PATH = './corpus_1900_1910/4/' #filename (.txt)
stopfilename = 'stop.txt'  # file with stop-words
filename = 'out.txt'


# In[ ]:


file_names = os.listdir(PATH)[:15]
texts = []
for name in file_names:
    if name.endswith(".txt"):
        with codecs.open(PATH + "/" + name, encoding = 'utf-8') as f:
            #print(name)
            text = f.read()
            lst = re.findall(r'\w+', text)
            words = []
            for word in lst:
                lemma = morph.parse(word)[0]  # parsing
                words.append(lemma.normal_form) #add a lemma to words
            
            texts.append(words)
            
dictionary = corpora.Dictionary(texts)
for i in dictionary.keys():
    print (dictionary[i])


# In[ ]:


#### Clining the dictionary  #### 
# deleting of stop-words 

stopfile = open('stop.txt', 'r') # file with stop-words with coding utf8
stopfile = codecs.open(stopfilename, 'r', encoding = 'utf8')
stopwords = stopfile.read()
stoplist = set(stopwords.split())
stopfile.close()


# In[ ]:


import logging, codecs

# logging.basicConfig(format='%
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities


# In[ ]:


stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
             if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids) # удаляем слова
print(dictionary)


# In[ ]:


once_ids = [tokenid for tokenid,docfreq in dictionary.dfs.items() if docfreq < 5] 
dictionary.filter_tokens(once_ids)
print(dictionary)


# In[ ]:


dictionary.compactify() # remove gaps in id sequence after words that were removed


# In[ ]:


# store the dictionary, for future reference
dictionary.save_as_text(filename + '_dict.txt')


# In[ ]:


##### LDA ###############
corpus = []
for document in file_names:
    if name.endswith(".txt"):
        with codecs.open(PATH + "/" + document, encoding = 'utf-8')as m:
            corpus.append(dictionary.doc2bow(m.read().lower().split()))
corpora.MmCorpus.serialize(filename + '_corpus.mm', corpus)
      
#print(corpus)


# In[ ]:


# load id->word mapping (the dictionary)
id2word = corpora.Dictionary.load_from_text(filename + '_dict.txt')
mm = corpora.MmCorpus(filename + '_corpus.mm') # Matrix Market format.
lda = models.ldamodel.LdaModel(corpus=mm, 
                               id2word=id2word, 
                               num_topics=10, # quantity of topics from the corpus 
                               update_every=1,#quantity of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning
                               chunksize=100, #quantity of documents for using in every chunk
                               passes=10 # quantity of iterations through the corpus .
                             # alpha='auto',
                             # per_word_topics=True
                              )


# In[ ]:


# print words from (number) random topics 
topics = lda.print_topics(10)
for topic in topics:
    print (topic[0],':',topic[1],"\n")


# In[ ]:


pprint(lda.print_topics()) ### perplexity for (number) topics ###
doc_lda = lda[corpus]
print('\nPerplexity: ', lda.log_perplexity(corpus))  # the result shows how good the model is. The lower the better.


# In[ ]:


# c_v Measure
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# UMass
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
vis


# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = lda
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


#lda = models.ldamodel.LdaModel(corpus=mm, 
 #                              id2word=id2word, # Mapping from word IDs to words. to determine the vocabulary size, as well as for debugging and topic printing.
  #                             num_topics=10, #The number of requested latent topics to be extracted from the training corpus.
   #                            update_every=1,#Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning
    #                           chunksize=100, #Number of documents to be used in each training chunk
     #                          passes=10 #Number of passes through the corpus during training.
                             # alpha='auto',
                             # per_word_topics=True
        #                      )


# In[ ]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=40, step=4)

limit=40; start=2; step=4;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[ ]:


# Select the model and print the topics
optimal_model = model_list[5]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[ ]:


#Find the most representative document for each topic

def format_topics_sentences(ldamodel=lda, corpus=corpus, texts=texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(35)

