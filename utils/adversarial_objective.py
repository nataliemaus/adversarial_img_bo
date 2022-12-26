import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from torchvision import transforms
from .objective import Objective
import sys 
sys.path.append("../")
from .get_synonyms import get_synonyms # 
from .related_vocab import RELATED_VOCAB_DICT
from .imagenet_classes import load_imagenet
from utils.autoencoder import AE 

class AdversarialsObjective(Objective):
    def __init__(
        self,
        num_calls=0,
        n_tokens=1,
        minimize=True,
        batch_size=10,
        use_fixed_latents=False,
        project_back=True, # project back embedding to closest real tokens
        avg_over_N_latents=8,
        exclude_all_related_prompts=False,
        exclude_some_related_prompts=True,
        visualize=False,
        compress_search_space=False,
        prepend_to_text="",
        optimal_class="cat",
        seed=1,
        remove_synonyms=False,
        single_number_per_token=False,
        **kwargs,
    ):
        super().__init__(
            num_calls=num_calls,
            task_id='adversarial4',
            dim=n_tokens*768,
            lb=None,
            ub=None,
            **kwargs,
        ) 
        # CONSTANTS
        # if prepend_to_text is not "", we will prepend the adversairla prompt to the text
        # ie prepend_to_text = "a picture of a dog"
        self.single_number_per_token = single_number_per_token
        self.prepend_to_text = prepend_to_text 
        self.remove_synonyms = remove_synonyms
        # if self.prepend_to_text:
        #     self.prepend_to_text = self.prepend_to_text + " <|endoftext|>" 
        self.N_extra_prepend_tokens = len(self.prepend_to_text.split() )

        self.compress_search_space = compress_search_space
        self.optimal_class = optimal_class
        if self.prepend_to_text:
            assert project_back
        self.exclude_all_related_prompts=exclude_all_related_prompts
        self.exclude_some_related_prompts=exclude_some_related_prompts
        self.visualize = visualize # flag to print individual tokens
        self.avg_over_N_latents = avg_over_N_latents # for use when use_fixed_latents==False,
        self.project_back = project_back
        self.token = "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" 
        self.n_tokens = n_tokens
        self.minmize = minimize 
        self.batch_size = batch_size
        self.height = 512                        # default height of Stable Diffusion
        self.width = 512                         # default width of Stable Diffusion
        self.num_inference_steps = 25            # Number of denoising steps, this value is decreased to speed up generation
        self.guidance_scale = 7.5                # Scale for classifier-free guidance
        self.generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
        self.max_num_tokens = n_tokens + self.N_extra_prepend_tokens # maximum number of tokens in input, at most 75 and at least 1
        self.dtype = torch.float16 
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=self.dtype, revision="fp16", use_auth_token=self.token)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=self.dtype)
        self.text_model = self.text_encoder.text_model

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=self.dtype, revision="fp16", use_auth_token=self.token)

        self.vae = self.vae.to(self.torch_device)
        self.text_encoder = self.text_encoder.to(self.torch_device)
        self.text_model = self.text_model.to(self.torch_device)
        self.unet = self.unet.to(self.torch_device) 

        # Scheduler for noise in image
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)
        self.scheduler.set_timesteps(self.num_inference_steps)

        # 4. The resnet18 model for classifying cat vs dog
        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet18.eval()
        self.resnet18.to(self.torch_device)

        # For reading the imagenet classes:
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
            
        self.preprocess_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.word_embedder = self.text_encoder.get_input_embeddings()
        self.uncond_input = self.tokenizer(
                    [""], padding="max_length", max_length=self.max_num_tokens+2, return_tensors="pt")
        with torch.no_grad():
            self.uncond_embed = self.word_embedder(self.uncond_input.input_ids.to(self.torch_device))

        if use_fixed_latents:
            self.fixed_latents = torch.randn(
                (1, 4, self.height // 8, self.width // 8),
                generator=self.generator, dtype=self.dtype
            ).to(self.torch_device) 

            self.fixed_latents = self.fixed_latents.repeat(self.batch_size, 1, 1, 1)
        else:
            self.fixed_latents = None 
        
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {self.vocab[k]:k for k in self.vocab.keys() }

        if self.exclude_all_related_prompts or self.exclude_some_related_prompts:
            # if only exlucde some (optimal class name, optimal class name +s, 
            #   anything containg optimal class name)
            if self.remove_synonyms:
                related_vocab = get_synonyms(self.optimal_class) + [self.optimal_class]
                related_vocab_s = [word + "s" for word in related_vocab]
                self.related_vocab = related_vocab + related_vocab_s 
            else:
                self.related_vocab = [self.optimal_class, self.optimal_class+"s"]
            if self.exclude_all_related_prompts: # cat, car or violin only prepped 
                self.related_vocab = RELATED_VOCAB_DICT[self.optimal_class]
            self.all_token_idxs = self.get_non_related_values() 
        else:
            self.all_token_idxs = list(self.vocab.values())
            self.related_vocab = []
        self.all_token_embeddings = self.word_embedder(torch.tensor(self.all_token_idxs).to(self.torch_device)) 

        if self.compress_search_space:
            assert not self.single_number_per_token
            # v2 (5 layers --> poor preformance! )
            n_layers = 4
            load_ckpt_wandb_name = "laced-snow-14"
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
            self.search_space_dim = self.compressed_embeddings.shape[-1] 
        elif self.single_number_per_token:
            self.search_space_dim = 1 # dim per token 
            self.compressed_embeddings = (torch.tensor(self.all_token_idxs).float()/len(self.vocab)).unsqueeze(-1)
            self.compressed_embeddings = self.compressed_embeddings.to(self.torch_device)
        else:
            self.search_space_dim = 768 # dim per token 
        self.dim = self.n_tokens*self.search_space_dim

    def get_non_related_values(self):
        tmp = [] 
        for word in self.related_vocab:
            tmp.append(word)
            tmp.append(word+'</w>') 
        self.related_vocab = tmp
        non_related_values = []
        for key in self.vocab.keys():
            if not ((key in self.related_vocab) or (self.optimal_class in key)):
                non_related_values.append(self.vocab[key])
        return non_related_values

    def prompt_to_token(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=self.max_num_tokens+2,
                            truncation=True, return_tensors="pt").input_ids.to(self.torch_device)
        return tokens


    def tokens_to_word_embed(self, tokens):
        with torch.no_grad():
            word_embed = self.word_embedder(tokens)
        return word_embed

    '''
        Preprocesses word embeddings
        word_embeddings can have max_num_tokens or max_num_tokens + 2, either with or without the 
        start-of-sentence and end-of-sentence tokens
        Will manually concatenate the correct SOS and EOS word embeddings if missing
        
        In the setting where word_embed is fed from tokens_to_word_embed, the middle dimension will be
        max_num_tokens + 2
        
        For manual optimization, suffices to only use dimension max_num_tokens to avoid redundancy
        
        args: 
            word_embed: dtype pytorch tensor shape (batch_size, max_num_tokens, 768) 
                        or (batch_size, max_num_tokens+2, 768) 
        returns:
            proc_word_embed: dtype pytorch tensor shape (batch_size, max_num_tokens+2, 768)
    '''
    def preprocess_word_embed(self, word_embed):
            
        # The first token dim is the start of text token and 
        # the last token dim is the end of text token
        # if word_embed is manually generated and missing these, we manually add it
        if word_embed.shape[1:] == (self.max_num_tokens, 768):
            batch_size = word_embed.shape[0]
            rep_uncond_embed = self.uncond_embed.repeat(batch_size, 1, 1)
            word_embed = torch.cat(
                [rep_uncond_embed[:,0:1,:],word_embed,rep_uncond_embed[:,-1:,:]],
                dim = 1
            )
        
        return word_embed


    '''
        Modified from https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/clip/modeling_clip.py#L611
        args: 
            proc_word_embed: dtype pytorch tensor shape (2, batch_size, max_num_tokens+2, 768)
        returns:
            CLIP_embed: dtype pytorch tensor shape (2, batch_size, max_num_tokens+2, 768)
    '''
    def preprocessed_to_CLIP(self, proc_word_embed): 

        # Hidden state from word embedding
        hidden_states = self.text_model.embeddings(inputs_embeds = proc_word_embed)
        
        attention_mask = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        with torch.no_grad():
            encoder_outputs = self.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        CLIP_embed = last_hidden_state
        return CLIP_embed

    '''
        Generates images from CLIP embeddings using stable diffusion
        args:
            clip_embed: dtype pytorch tensor shape (batch_size, max_num_tokens + 2, 768)
        returns:
            images: array of PIL images
        
    '''
    def CLIP_embed_to_image(self, clip_embed, fixed_latents = None):


        batch_size = clip_embed.shape[0]
        rep_uncond_embed = self.uncond_embed.repeat(batch_size, 1, 1)

        # Concat unconditional and text embeddings, used for classifier-free guidance    
        clip_embed = torch.cat([rep_uncond_embed, clip_embed])
        if fixed_latents is not None:
            assert fixed_latents.shape == (batch_size, self.unet.in_channels, self.height // 8, self.width // 8)
            latents = fixed_latents
        else:
        # Generate initial random noise
            latents = torch.randn(
            (batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
            generator=self.generator, dtype = self.dtype
            ).to(self.torch_device)
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

        scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * scheduler.init_noise_sigma
        
        # Diffusion process
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=clip_embed).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        # Use vae to decode latents into image
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    '''
        Uses resnet18 as a cat classifier
        Our loss is the negative log probability of the class of cats
        Loss will be small when stable diffusion generates images of cats
    '''
    def image_to_loss(self, imgs):
        input_tensors = []
        for img in imgs:
            input_tensors.append(self.preprocess_img(img))
        input_tensors = torch.stack(input_tensors)
        input_batch = input_tensors.to(self.torch_device)
        with torch.no_grad():
            output = self.resnet18(input_batch)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output, dim=1)
        if self.optimal_class == "cat":
            class_ix0 = 281
            class_ix1 = 282
        elif self.optimal_class == "violin":
            class_ix0 = 889
            class_ix1 = 890 
        elif self.optimal_class == "car": 
            # 705, 706 = passenger car (but could be train passenger car too... idk, stick w/ sports car!)
            # 817, 818 = sports car !!! 
            class_ix0 = 817
            class_ix1 = 818 
        else:
            ix = load_imagenet()[self.optimal_class]
            class_ix0 = ix + 1 
            class_ix1 = ix + 2 
        total_probs = torch.max(probabilities[:,class_ix0:class_ix1], dim = 1).values # classes 281:282 are HOUSE cat classes
        # total_cat_probs = torch.max(probabilities[:,281:286], dim = 1).values # classes 281:286 are cat classes
        #total_dog_probs = torch.sum(probabilities[:,151:268], dim = 1) # classes 151:268 are dog classes
        #p_dog = total_dog_probs / (total_cat_probs + total_dog_probs)
        loss = - torch.log(total_probs)
        return loss



    '''
        Pipeline order
        prompt -> tokens -> word_embedding -> processed_word_embedding -> 
        CLIP_embedding -> image -> loss

        Function that accepts intermediary values in the pipeline and outputs downstream values

        input_type options: ['prompt', 'word_embedding', 'CLIP_embedding', 'image']
        output_type options: ['tokens', 'word_embedding', 'processed_word_embedding',
                                'CLIP_embedding', 'image', 'loss']
    '''
    def pipeline(self, input_type, input_value, output_types, fixed_latents = None):

        valid_input_types = ['prompt', 'word_embedding', 'CLIP_embedding', 'image']
        valid_output_types = ['tokens', 'word_embedding', 'processed_word_embedding', 'CLIP_embedding', 'image', 'loss']
        if input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types} but was {input_type}")
        for cur_output_type in output_types:
            if cur_output_type not in valid_output_types:
                raise ValueError(f"output_type must be one of {valid_output_types} but was {cur_output_type}")
        # Check that output is downstream
        pipeline_order = ["prompt", "tokens", "word_embedding", "processed_word_embedding", 
                        "CLIP_embedding", "image","loss"]
        pipeline_maps = {"prompt": self.prompt_to_token,
                        "tokens": self.tokens_to_word_embed,
                        "word_embedding": self.preprocess_word_embed, 
                        "processed_word_embedding": self.preprocessed_to_CLIP, 
                        "CLIP_embedding": self.CLIP_embed_to_image, 
                        "image": self.image_to_loss}

        start_index = pipeline_order.index(input_type)
        max_end_index = start_index
        for cur_output_type in output_types:
            cur_end_index = pipeline_order.index(cur_output_type)
            if start_index >= cur_end_index:
                raise ValueError(f"{output_types} is not downstream of {input_type}.")
            else:
                max_end_index = max(max_end_index,cur_end_index)
        # Check that shapes are valid
        if input_type == "word_embedding":
            if input_value.shape[1:] != (self.max_num_tokens, 768):
                raise ValueError(f"Word embeddings are the incorrect size, \
                    should be (batch_size, {self.max_num_tokens}, 768) but were {input_value.shape}")
        elif input_type == "CLIP_embedding": 
            if input_value.shape[1:] != (self.max_num_tokens+2, 768):
                raise ValueError(f"CLIP embeddings are the incorrect size, \
                    should be (batch_size, {self.max_num_tokens+2}, 768) but were {input_value.shape}")


        cur_pipe_val = input_value
        output_dict = {}
        for i in range(start_index, max_end_index):
            cur_type = pipeline_order[i]
            mapping = pipeline_maps[cur_type]
            if cur_type == "CLIP_embedding":
                
                cur_pipe_val = mapping(cur_pipe_val, fixed_latents = fixed_latents)
            else:
                cur_pipe_val = mapping(cur_pipe_val)
            next_type = pipeline_order[i+1]
            if next_type in output_types:
                output_dict[next_type] =  cur_pipe_val
        return output_dict

    def query_oracle(self, x, return_img=False):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda() 
        x = x.reshape(-1, self.n_tokens, self.search_space_dim) 
        out_types = ["loss"]
        if return_img:
            out_types = ["image", "loss"]
        
        input_type = "word_embedding"
        if self.project_back:
            x = self.proj_word_embedding(x)
            input_type = "prompt"
            x = [x1[0] for x1 in x] 

        if self.fixed_latents is None: # not using fixed latents... 
            ys = [] 
            imgs_per_latent = [] ## N latents x bsz : [ [latent 0, bsz imgs ], [latent 1, bsz imgs], ..., [latent N bsz imgs]]
            for _ in range(self.avg_over_N_latents):
                out_dict = self.pipeline(
                    input_type=input_type,
                    input_value=x, 
                    output_types=out_types,
                    fixed_latents=self.fixed_latents
                )
                y = out_dict['loss']
                ys.append(y.unsqueeze(0)) 
                if return_img:
                    imgs = out_dict["image"] 
                    imgs_per_latent.append(imgs) 
            ys = torch.cat(ys) # torch.Size([N_latents, bsz]) 
            y = ys.mean(0) # (bsz,) 
            if return_img:
                imgs = [] # bsz x N latents: [ [latent 0, latent 1, ...], [latent 0, latent 1, ...],, ..., [latent 0, latent 1, ...],]
                for i in range(self.batch_size):
                    imgs_from_xi = [] 
                    for ims_per_latent in imgs_per_latent:
                        imgs_from_xi.append(ims_per_latent[i])
                    imgs.append(imgs_from_xi)
        else:
            out_dict = self.pipeline(
                input_type=input_type,
                input_value=x, 
                output_types=out_types,
                fixed_latents=self.fixed_latents
            )
            y = out_dict['loss']
            if return_img:
                imgs = out_dict["image"] 
        if self.minmize: 
            y = y*-1 
        if return_img:
            return imgs, x, y # return x becaausee w/ proj back it is the closeest! prompt
        return x, y 
    
    def get_init_word_embeddings(self, prompts):
        # Word embedding initialization at "cow"
        # promts = list of words, ie ["cow", "horse", "cat"]
        if self.single_number_per_token:
            all_tokens = []
            for prompt in prompts:
                tokens = [self.vocab[word] for word in prompt.split()]
                tokens = torch.tensor(tokens).float().unsqueeze(0)
                all_tokens.append(tokens)
            all_tokens = torch.cat(all_tokens).cuda() 
            word_embeddings = all_tokens/len(self.vocab)
        else:
            word_embeddings =self.pipeline(
                input_type="prompt", 
                input_value=prompts, 
                output_types = ["word_embedding"]
            )["word_embedding"][:,1:-1,:] 
            if self.N_extra_prepend_tokens > 0:
                word_embeddings = word_embeddings[:, 0:-self.N_extra_prepend_tokens, :]

            if self.compress_search_space:
                tmp = []
                for x in word_embeddings:
                    tmp.append(self.ae.encoder(x.float()).unsqueeze(0))
                word_embeddings = torch.cat(tmp, 0).to(torch.float16)

        return word_embeddings


    def proj_word_embedding(self, word_embedding):
        '''
            Given a word embedding, project it to the closest word embedding of actual tokens using cosine similarity
            Iterates through each token dim and projects it to the closest token
            args:
                word_embedding: (batch_size, max_num_tokens, 768) word embedding
            returns:
                proj_tokens: (batch_size, max_num_tokens) projected tokens
        '''
        # Get word embedding of all possible tokens as torch tensor
        proj_tokens = []
        # Iterate through batch_size
        for i in range(word_embedding.shape[0]):
            # Euclidean Norm
            if self.compress_search_space or self.single_number_per_token: 
                # single num per token:
                #   compressed_emb = (49408, 1) --> torch.Size([49408, 1, 1])
                #   embedding =  bsz, tokens, 768 
                # compressed w/ ae:
                #   compressed_emb = torch.Size([49408, 768]) --> torch.Size([49408, 1, 768])
                #   embedding =  bsz, n_tokens, 768 
                dists = torch.norm(self.compressed_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
            else:
                dists = torch.norm(self.all_token_embeddings.unsqueeze(1) - word_embedding[i,:,:], dim = 2)
            closest_tokens = torch.argmin(dists, axis = 0)
            closest_tokens = torch.tensor([self.all_token_idxs[token] for token in closest_tokens]).to(self.torch_device)
            closest_vocab = self.tokenizer.decode(closest_tokens)
            if self.prepend_to_text: 
                closest_vocab = closest_vocab + " " + self.prepend_to_text + " <|endoftext|>" 
            cur_proj_tokens = [closest_vocab]
            proj_tokens.append(cur_proj_tokens) 
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
    obj = AdversarialsObjective() 
    x = torch.randn(10, 1*768, dtype=torch.float16) 
    y = obj(x)  # torch.Size([10]) == bsz 
    

