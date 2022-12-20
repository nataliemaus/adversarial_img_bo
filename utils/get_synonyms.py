import nltk 
nltk.download('omw-1.4')
from nltk.corpus import wordnet

def get_synonyms(word):
    syns = wordnet.synsets(word)
    synonyms = []
    for syn in syns:
        for lemma in syn.lemmas():
            synonyms = synonyms + lemma.name().split("_") 
    return synonyms
