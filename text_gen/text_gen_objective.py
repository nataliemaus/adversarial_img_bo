from transformers import GPT2Tokenizer, OPTModel, pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import sys 
sys.path.append("../")
from utils.objective import Objective 
from utils.autoencoder import AE 

class AdversarialsTextGenObjective(Objective):
    def __init__(
        self,
        num_calls=0,
        n_tokens=1,
        minimize=True,
        batch_size=10,
        visualize=False,
        compress_search_space=False,
        single_number_per_token=False,
        prepend_to_text="",
        num_gen_seq=5,
        max_gen_length=10,
        dist_metric="sq_euclidean",
        lb=None,
        ub=None,
        **kwargs,
    ):
        super().__init__(
            num_calls=num_calls,
            task_id='adversarial4',
            dim=n_tokens*768,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 

        assert dist_metric in ['cosine_sim', "sq_euclidean"]
        self.single_number_per_token = single_number_per_token
        self.prepend_to_text = prepend_to_text
        self.N_extra_prepend_tokens = len(self.prepend_to_text.split() )
        self.dist_metric = dist_metric # metric='cosine_sim'
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.distilBert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.distilBert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        self.generator = pipeline("text-generation", model="facebook/opt-125m")
        self.model = OPTModel.from_pretrained("facebook/opt-125m")
        self.model = self.model.to(self.torch_device)
        self.word_embedder = self.model.get_input_embeddings()
        self.vocab = self.tokenizer.get_vocab()
        self.num_gen_seq = num_gen_seq
        self.max_gen_length = max_gen_length + n_tokens + self.N_extra_prepend_tokens 
       
        # Currently not used
        self.all_token_idxs = list(self.vocab.values())
        self.all_token_embeddings = self.word_embedder(torch.tensor(self.all_token_idxs).to(self.torch_device)) 
        self.all_token_embeddings_norm = self.all_token_embeddings / self.all_token_embeddings.norm(dim=-1, keepdim=True)
        # XXX 
        # XXX 

        self.compress_search_space = compress_search_space
        self.visualize = visualize # flag to print individual tokens
        # self.token = "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" 
        self.n_tokens = n_tokens
        self.minmize = minimize 
        self.batch_size = batch_size

        if self.compress_search_space:
            assert not self.single_number_per_token
            # latent_dim_dict = {} 
            # latent_dim_dict[24] = ["devout-firebrand-15", 5]
            n_layers = 5
            load_ckpt_wandb_name = "devout-firebrand-15"
            self.ae = AE(
                input_shape=768,
                n_layers=n_layers,
            ) 
            path_to_state_dict = f"../ae_models/{load_ckpt_wandb_name}.pkl" 
            state_dict = torch.load(path_to_state_dict) # load state dict 
            self.ae.load_state_dict(state_dict, strict=True) 
            self.ae = self.ae.cuda() 
            self.compressed_embeddings = self.ae.encoder(self.all_token_embeddings.float()).to(torch.float16)
            # torch.Size([49408, 768]) --> torch.Size([49407, 24])
            self.search_space_dim = self.compressed_embeddings.shape[-1] # dim per token 
        elif self.single_number_per_token:
            self.search_space_dim = 1 # dim per token 
            self.compressed_embeddings = (torch.tensor(self.all_token_idxs).float()/len(self.vocab)).unsqueeze(-1)
            self.compressed_embeddings = self.compressed_embeddings.to(self.torch_device)
        else:
            self.search_space_dim = 768 
        self.dim = self.n_tokens*self.search_space_dim

    '''
        Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
        Iterates through each token dim and projects it to the closest token
        args:
            word_embedding: (batch_size, max_num_tokens, 768) word embedding
        returns:
            proj_tokens: (batch_size, max_num_tokens) projected tokens
    '''
    def proj_word_embedding(self, word_embedding):
        # Get word embedding of all possible tokens as torch tensor

        proj_tokens = []
        # Iterate through batch_size
        
        for i in range(word_embedding.shape[0]):
            
            if self.dist_metric == 'cosine_sim':
                sims = torch.matmul(self.all_token_embeddings_norm, word_embedding[i,:,:].T)/torch.norm(word_embedding[i,:,:], dim = 1)
                sims = torch.nan_to_num(sims, nan = -100)
                # top5 = torch.topk(sims, 5, axis = 0, largest=True)
                # print("top 5")
                # for i in range(n_tokens):
                #     decoded = [tokenizer.decode(top5.indices[ind,i]) for ind in range(5)]
                #     print(f"token {i+1}/{n_tokens}: {decoded} with sims {top5.values[:,i].detach().cpu().numpy()}")
                closest_tokens = torch.argmax(sims, axis = 0)
            elif self.dist_metric == "sq_euclidean":
                # Euclidean Norm
                dists =  torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
                closest_tokens = torch.argmin(dists, axis = 0)
            cur_proj_tokens = self.tokenizer.decode(closest_tokens)
            proj_tokens.append(cur_proj_tokens)

        return proj_tokens

    def prompt_to_text(self, prompts):
        gen_texts = self.generator( prompts, max_length=self.max_gen_length, num_return_sequences=self.num_gen_seq, num_beams=self.num_gen_seq)
        gen_texts = [[cur_dict['generated_text'] for cur_dict in cur_gen] for cur_gen in gen_texts]
        return gen_texts
        
    def text_to_loss(self, text, loss_type = 'log_prob_pos'):
        num_prompts = len(text)
        flattened_text = [item for sublist in text for item in sublist]
        inputs = self.distilBert_tokenizer(flattened_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.distilBert_model(**inputs).logits
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
                        "prompt": self.prompt_to_text, # prompt to generated text 
                        "generated_text": self.text_to_loss, # text to generated loss 
                        }

        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{cur_output_type} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index,cur_end_index)

        # # Check that shapes are valid
        # if input_type == "raw_word_embedding":
        #     if len(input_value.shape) != 3 or input_value.shape[2] != 768:
        #         raise ValueError(f"Word embeddings are the incorrect size, \
        #             should be (batch_size, num_tokens, 768) but were {input_value.shape}")

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

    def query_oracle(self, x ):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda() 
        x = x.reshape(-1, self.n_tokens, self.search_space_dim) 
        out_dict = self.pipe(
            input_type="raw_word_embedding", 
            input_value=x, 
            output_types=['prompt','generated_text','loss'] 
        ) 
        y = out_dict['loss'].mean(-1 ) 
        if self.minmize: 
            y = y*-1 
        return out_dict['prompt'], y, out_dict["generated_text"]

    def proj_word_embedding(self, word_embedding):
        '''
            Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
            Iterates through each token dim and projects it to the closest token
            args:
                word_embedding: (batch_size, max_num_tokens, 768) word embedding
            returns:
                proj_tokens: (batch_size, max_num_tokens) projected tokens
        '''
        # Get word embeddings of all possible tokens as torch tensor
        proj_tokens = []
        # Iterate through batch_size
        for i in range(word_embedding.shape[0]):
            # Euclidean Norm
            if self.compress_search_space or self.single_number_per_token:
                dists =  torch.norm(self.compressed_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
            else:
                dists =  torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
            closest_tokens = torch.argmin(dists, axis = 0)
            closest_tokens = torch.tensor([self.all_token_idxs[token] for token in closest_tokens]).to(self.torch_device)
            closest_vocab = self.tokenizer.decode(closest_tokens)
            if self.prepend_to_text: 
                closest_vocab = closest_vocab + " " + self.prepend_to_text
            # cur_proj_tokens = [closest_vocab]
            proj_tokens.append(closest_vocab)  # cur_proj_tokens) 
            if self.visualize: # visualizing 
                tokenized = ""
                for ix, token in enumerate(closest_tokens):
                    if ix > 0:
                        tokenized += ","
                    word = self.tokenizer.decode(token)
                    tokenized += word 
                    print(f"token{ix + 1}:", word)
                print("TOKENIZED:", tokenized)
                import pdb 
                pdb.set_trace() 

        return proj_tokens


if __name__ == "__main__":
    obj = AdversarialsTextGenObjective() 
    x = torch.randn(10, 1*768, dtype=torch.float16)*0.01
    x, y, gen_text  = obj(x)  # torch.Size([10]) == bsz 
