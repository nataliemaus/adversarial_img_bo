import pandas as pd 
from utils.get_synonyms import get_synonyms

def load_imagenet():
    path = "../data/imagenet_classes.txt"
    df = pd.read_csv(path)
    classes = df.values.squeeze() 
    imagenet_dict = {}
    for ix, clas in enumerate(classes):
        if len(clas.split()) == 1:
            if not ('-' in clas):
                if not ("dog" in clas): # a picture of a dog does not work if dog is the class
                    if len(get_synonyms(clas)) > 1:
                        imagenet_dict[clas] = ix
    return imagenet_dict 