import numpy as np
import sys 
sys.path.append("../")
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from utils.adversarial_objective import AdversarialsObjective

objective = AdversarialsObjective(
    allow_related_prompts=True
)
all_embeddings = objective.all_token_embeddings.detach().cpu().numpy()
# torch.Size([49408, 768]) = Ntokens x 768 

pca = PCA(n_components=all_embeddings.shape[-1] )
pca.fit(all_embeddings)
var_explained_arr = pca.explained_variance_ratio_ # (768,)

nums_comps = []
var_explained = []
for n_components in range(2, all_embeddings.shape[-1] ):
    nums_comps.append(n_components)
    var_explained.append(var_explained_arr[0:n_components].sum().item())

nums_comps = np.array(nums_comps)
var_explained = np.array(var_explained )

plt.title("Word Embeddings, Number of Components vs. Variance Explained")
plt.plot(nums_comps, var_explained)
plt.xlabel("Number of Components in PCA")
plt.ylabel("Variance Explained") 
plt.savefig("../data/pca.png")

import pdb 
pdb.set_trace() 
