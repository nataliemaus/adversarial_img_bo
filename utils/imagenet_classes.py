import pandas as pd 
from utils.get_synonyms import get_synonyms

def load_imagenet():
    path = "../data/imagenet_classes.txt"
    df = pd.read_csv(path, header=None)
    classes = df.values.squeeze() 
    imagenet_class_to_ix = {}  
    ix_to_imagenet_class = {}
    for ix, clas in enumerate(classes):
        imagenet_class_to_ix[clas] = ix 
        ix_to_imagenet_class[ix] = clas 

    # class_ix = 817
    import pdb 
    pdb.set_trace() 
    return imagenet_class_to_ix, ix_to_imagenet_class


def load_valid_imagenet_classes():
    path = "../data/imagenet_classes.txt"
    df = pd.read_csv(path, header=None)
    classes = df.values.squeeze() 
    valid_classes = []
    for clas in classes:
        if len(clas.split()) == 1:
            if not ('-' in clas):
                if not ("dog" in clas): # a picture of a dog does not work if dog is the class
                    if len(get_synonyms(clas)) > 1:
                        valid_classes.append(clas)

    return valid_classes
