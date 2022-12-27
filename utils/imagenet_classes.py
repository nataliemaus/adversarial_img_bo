import pandas as pd 
from utils.get_synonyms import get_synonyms

def load_imagenet():
    path = "../data/imagenet_classes.txt"
    df = pd.read_csv(path)
    classes = df.values.squeeze() 
    imagenet_class_to_ix = {}  
    ix_to_imagenet_class = {}
    for ix, clas in enumerate(classes):
        if len(clas.split()) == 1:
            if not ('-' in clas):
                if not ("dog" in clas): # a picture of a dog does not work if dog is the class
                    if len(get_synonyms(clas)) > 1:
                        imagenet_class_to_ix[clas] = ix + 1
                        ix_to_imagenet_class[ix + 1] = clas 

    return imagenet_class_to_ix, ix_to_imagenet_class
