

from transformers import GPT2Tokenizer, OPTModel, pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

distilBert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
distilBert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


#device = 0#1 if torch.cuda.is_available() else 0
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
generator = pipeline("text-generation", model="facebook/opt-125m")
model = OPTModel.from_pretrained("facebook/opt-125m")

# XXX 
#generator = generator.to(torch_device)
model = model.to(torch_device)

word_embedder = model.get_input_embeddings()
vocab = tokenizer.get_vocab()


class text_pipeline():

    def __init__(self, num_gen_seq, max_gen_length, related_tokens = []):
        self.num_gen_seq = num_gen_seq
        self.max_gen_length = max_gen_length
       
        # Currently not used
        self.related_tokens = related_tokens            # List of related tokens to the prompt
        non_related_values = [vocab[key] for key in vocab.keys() if key not in self.related_tokens]
        self.all_token_embeddings = word_embedder(torch.tensor(non_related_values).to(torch_device)) 
        self.all_token_embeddings_norm = self.all_token_embeddings / self.all_token_embeddings.norm(dim=-1, keepdim=True)

    '''
        Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
        Iterates through each token dim and projects it to the closest token
        args:
            word_embedding: (batch_size, max_num_tokens, 768) word embedding
        returns:
            proj_tokens: (batch_size, max_num_tokens) projected tokens
    '''
    def proj_word_embedding(self, word_embedding, metric = 'cosine_sim'):
        # Get word embedding of all possible tokens as torch tensor

        proj_tokens = []
        # Iterate through batch_size
        
        for i in range(word_embedding.shape[0]):
            
            if metric == 'cosine_sim':
                sims = torch.matmul(self.all_token_embeddings_norm, word_embedding[i,:,:].T)/torch.norm(word_embedding[i,:,:], dim = 1)
                sims = torch.nan_to_num(sims, nan = -100)
                top5 = torch.topk(sims, 5, axis = 0, largest=True)
                # print("top 5")
                # for i in range(n_tokens):
                #     decoded = [tokenizer.decode(top5.indices[ind,i]) for ind in range(5)]
                #     print(f"token {i+1}/{n_tokens}: {decoded} with sims {top5.values[:,i].detach().cpu().numpy()}")
                closest_tokens = torch.argmax(sims, axis = 0)
            elif metric == "sq_euclidean":
                # Euclidean Norm
                dists =  torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
                closest_tokens = torch.argmin(dists, axis = 0)
            cur_proj_tokens = tokenizer.decode(closest_tokens)
            proj_tokens.append(cur_proj_tokens)

        return proj_tokens

    def prompt_to_text(self, prompt):
        gen_texts = generator(prompt, max_length=self.max_gen_length, 
            num_return_sequences=self.num_gen_seq, 
            num_beams = self.num_gen_seq)
        gen_texts = [[cur_dict['generated_text'] for cur_dict in cur_gen] for cur_gen in gen_texts]
        return gen_texts
        
    def text_to_loss(self, text, loss_type = 'log_prob_pos'):
        num_prompts = len(text)
        flattened_text = [item for sublist in text for item in sublist]
        inputs = distilBert_tokenizer(flattened_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = distilBert_model(**inputs).logits
        probs = torch.softmax(logits, dim = 1)
        
        if loss_type == 'log_prob_pos':
            loss = torch.log(probs[:,1])
        elif loss_type == 'log_prob_neg':
            loss = torch.log(probs[:,0])
        else:
            raise ValueError(f"loss_type must be one of ['log_prob_pos', 'log_prob_neg'] but was {loss_type}")
        loss = loss.reshape(num_prompts, -1)
        return loss 
        
    def pipe(self, input_type, input_value, output_types):

        valid_input_types = ['raw_word_embedding' ,'prompt']
        valid_output_types = ['prompt', 'generated_text', 'loss']

        # Check that types are valid
        if input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types} but was {input_type}")
        for cur_output_type in output_types:
            if cur_output_type not in valid_output_types:
                raise ValueError(f"output_type must be one of {valid_output_types} but was {cur_output_type}")
        # Check that output is downstream
        pipeline_order = ["raw_word_embedding", "prompt", "generated_text","loss"]
        pipeline_maps = {"raw_word_embedding": self.proj_word_embedding,
                        "prompt": self.prompt_to_text,
                        "generated_text": self.text_to_loss,
                        }

        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{cur_output_type} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index,cur_end_index)

        # Check that shapes are valid
        if input_type == "raw_word_embedding":
            if len(input_value.shape) != 3 or input_value.shape[2] != 768:
                raise ValueError(f"Word embeddings are the incorrect size, \
                    should be (batch_size, num_tokens, 768) but were {input_value.shape}")

        cur_pipe_val = input_value
        output_dict = {}
        for i in range(start_index, max_end_index):
            cur_type = pipeline_order[i]
            mapping = pipeline_maps[cur_type]
            
            cur_pipe_val = mapping(cur_pipe_val)
            next_type = pipeline_order[i+1]
            if next_type in output_types:
                output_dict[next_type] =  cur_pipe_val
        return output_dict



# Examples:
pl = text_pipeline(num_gen_seq = 3, max_gen_length = 20)
out = pl.pipe(input_type = "prompt", 
    input_value = ["I went to the movies and", "I had an ice cream and"],
    output_types = ["generated_text","loss"])
print(out["generated_text"])
print(out["loss"])

embed = (torch.randn(4, 5, 768)).to(torch_device)
out = pl.pipe(input_type = "raw_word_embedding", 
    input_value = embed,
    output_types = ["generated_text","loss"])
print(out["generated_text"])
print(out["loss"])