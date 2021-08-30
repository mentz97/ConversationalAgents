import string
import re 
import nltk
import spacy
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stopwords=nltk.corpus.stopwords.words('english')
wordnet_lemmatizer=WordNetLemmatizer()
porter_stemmer=PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def tok(text):
  token=re.split('\s', text)
  return token

def remove_stopwords(text): 
  out=[i for i in text if i not in stopwords]
  return out

def stemming(text):
  stem=[porter_stemmer.stem(word) for word in text]
  return stem

def lemm(text):
  lem_t=[wordnet_lemmatizer.lemmatize(sent) for sent in text]
  return lem_t

def tostring(text):
  s="".join([i+" " for i in text])
  s=s[0:len(s)-1]
  s=s.replace('cls', 'CLS')
  s=s.replace('sep', 'SEP')
  s=s.replace('pad', 'PAD')
  return s

def infinitive(sentence: str) -> str:
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            sentence = sentence.replace(token.text, " " + token.lemma_, 1) if token.shape_.startswith(
                "'") else sentence.replace(token.text, token.lemma_, 1)

    return sentence
    
def preprocess(sentence: str) -> str:
  ts=infinitive(sentence)
  #ts=tok(sent)
  #ts=remove_stopwords(sent)
  #ts=stemming(sent)
  #ts=lemm(sent) 
  #ts=tostring(sent)
  return ts

