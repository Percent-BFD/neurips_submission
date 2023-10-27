from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.generation import GenerationConfig
import re
import torch
from peft import PeftModel


class BaseModel:

    def __init__(self,model_path,tokenizer_path,device):
        self.model_path=model_path
        self.tokenizer_path=tokenizer_path
        self.device=device

    def load_model(self,path,device):
        pass

    def load_tokenizer(self,path):
        pass

    def generate(self,input):
        pass

class HFllama(BaseModel):

    def __init__(self, model_path, tokenizer_path, device, cache_dir=None,if_bf16=True,lora_weights=None,load_8bit=False,load_4bit=False):
        super().__init__(model_path, tokenizer_path, device)
        self.cache_dir=cache_dir
        if load_8bit and load_4bit:
            raise Exception('load 8bit and 4bit cannot be true at the same time')
        else:
            self.load_4bit = load_4bit
            self.load_8bit = load_8bit
        if if_bf16:
            self.dtype=torch.bfloat16
        else:
            self.dtype=torch.float32
        print('Loading model from',self.model_path,'on',self.device)
        self.model=self.load_model(self.model_path,self.device)
        self.model.eval()
        if lora_weights != None:
            print('Loading lora weights from',lora_weights)
            self.model=self.load_lora(lora_weights)
        print('Loading tokenizer from',tokenizer_path)
        self.tokenizer=self.load_tokenizer(self.tokenizer_path,self.cache_dir)
        print('Got it!')   

    def load_model(self, path, device):
        config=AutoConfig.from_pretrained(path, trust_remote_code=True,cache_dir=self.cache_dir)
        if device!='auto':
            pattern = r'cuda:(\d+)'
            match = re.match(pattern, device)
            device_id = match.group(1)
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype,
                load_in_4bit=self.load_4bit,
                load_in_8bit=self.load_8bit,
                device_map={'':int(device_id)}
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype,
                load_in_4bit=self.load_4bit,
                load_in_8bit=self.load_8bit,
                device_map='auto',
            )
    
    def load_lora(self,lora_path):
        model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            torch_dtype=self.dtype,
        )
        return model
    
    def load_tokenizer(self,path,cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True,cache_dir=cache_dir,use_fast=False)
        if tokenizer.pad_token == None:
            self.model.config.pad_token_id = 0
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.padding_side='left'
        return tokenizer
    
    def generate(self, input, max_new_tokens=64, temperature=0.95,top_p=0.96,top_k=1):
        inputs = self.tokenizer.batch_encode_plus(input, return_tensors='pt',padding=True)
        inputs_ids = inputs['input_ids'].to(self.model.device)
        preds = self.model.generate(input_ids=inputs_ids,do_sample=True, max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=top_k)
        input_length=inputs_ids.shape[1]
        output_length=preds.shape[1]
        preds = torch.index_select(preds, dim=1, index=torch.arange(input_length, output_length).to(self.model.device))
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)

class Bloom(BaseModel):

    def __init__(self, model_path, tokenizer_path, device, cache_dir=None,if_bf16=True,lora_weights=None):
        super().__init__(model_path, tokenizer_path, device)
        self.cache_dir=cache_dir
        self.cache_dir = cache_dir
        if if_bf16:
            self.dtype=torch.bfloat16
        else:
            self.dtype=torch.float32
        print('Loading model from',self.model_path,'on',self.device)
        self.model=self.load_model(self.model_path,self.cache_dir,self.device)
        self.model.eval()
        if lora_weights != None:
            print('Loading lora weights from',lora_weights)
            self.model=self.load_lora(lora_weights)
        print('Loading tokenizer from',tokenizer_path)
        self.tokenizer=self.load_tokenizer(self.tokenizer_path,self.cache_dir)
        print('Got it!')   

    def load_model(self, path, cache_dir, device):
        config=AutoConfig.from_pretrained(path, trust_remote_code=True,cache_dir=cache_dir)
        if device!='auto':
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
            ).to(device)
        else:
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map='auto',
                torch_dtype=self.dtype,
            )
        
    def load_lora(self,lora_path):
            model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=self.dtype,
            )
            return model

    def load_tokenizer(self,path,cache_dir):
        return AutoTokenizer.from_pretrained(path,trust_remote_code=True,cache_dir=cache_dir)

    def generate(self, input, max_new_tokens=64, temperature=0.95,top_p=0.96,top_k=1):
        inputs = self.tokenizer.batch_encode_plus(input, return_tensors='pt',padding=True)
        inputs_ids = inputs['input_ids'].to(self.device)
        preds = self.model.generate(input_ids = inputs_ids, max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=top_k)
        input_length=inputs_ids.shape[1]
        output_length=preds.shape[1]
        preds = torch.index_select(preds, dim=1, index=torch.arange(input_length, output_length).to(self.device))
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    
class Qwen(BaseModel):

    def __init__(self, model_path, tokenizer_path, device, cache_dir=None,if_bf16=True,lora_weights=None):
        super().__init__(model_path, tokenizer_path, device)
        self.cache_dir=cache_dir
        self.cache_dir = cache_dir
        if if_bf16:
            self.dtype=torch.bfloat16
        else:
            self.dtype=torch.float32
        print('Loading model from',self.model_path,'on',self.device)
        self.model=self.load_model(self.model_path,self.cache_dir,self.device)
        self.model.eval()
        if lora_weights != None:
            print('Loading lora weights from',lora_weights)
            self.model=self.load_lora(lora_weights)
        print('Loading tokenizer from',tokenizer_path)
        self.tokenizer=self.load_tokenizer(self.tokenizer_path,self.cache_dir)
        print('Got it!')   

    def load_model(self, path, cache_dir, device):
        config=AutoConfig.from_pretrained(path, trust_remote_code=True,cache_dir=cache_dir)
        if device!='auto':
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
            ).to(device)
        else:
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map='auto',
                torch_dtype=self.dtype,
            )
        
    def load_lora(self,lora_path):
            model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=self.dtype,
            )
            return model

    def load_tokenizer(self,path,cache_dir):
        tokenizer =  AutoTokenizer.from_pretrained(path,trust_remote_code=True,cache_dir=cache_dir)
        if tokenizer.pad_token == None:
            self.model.config.pad_token_id = 151643
            self.model.config.eos_token_id = 151643
            tokenizer.pad_token_id = 151643
            tokenizer.eos_token_id = 151643
        return tokenizer
    
    # def input_template(self,input):
    #     prefix="<|im_start|>system\nYou are a helpful assistant."
    #     prompt=f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    #     inputs = prefix + prompt
    #     return inputs

    def generate(self, input, max_new_tokens=64, temperature=0.95,top_p=0.96,top_k=1):
        # input = self.input_template(input)
        inputs = self.tokenizer.batch_encode_plus(input, return_tensors='pt',padding=True)
        inputs_ids = inputs['input_ids'].to(self.device)
        preds = self.model.generate(input_ids = inputs_ids, max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=top_k)
        input_length=inputs_ids.shape[1]
        output_length=preds.shape[1]
        preds = torch.index_select(preds, dim=1, index=torch.arange(input_length, output_length).to(self.device))
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)
