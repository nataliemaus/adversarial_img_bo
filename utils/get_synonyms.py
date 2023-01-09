import nltk 
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet

def get_synonyms(word):
    syns = wordnet.synsets(word)
    synonyms = []
    # dog = wordnet.synsets('dog')[0]
    for syn in syns:
        for lemma in syn.lemmas():
            synonyms = synonyms + lemma.name().split("_") 
    return synonyms

if __name__ == "__main__":
    import pdb 
    pdb.set_trace() 
    get_synonyms(word='spaniel') 
    word='spaniel'
    word = dog = wordnet.synsets(word)[0]
    params = word.hypernyms()[0]
