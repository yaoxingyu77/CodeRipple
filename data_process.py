import os
import numpy as np
import json
from typing import Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"


ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)
huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}
# human eval data
def read_data(dataset_type, task, generative_model):
    if dataset_type == "human_eval":
        base_dir = "./HumanEval" 
        print(base_dir)
        with open(f'./HumanEval/{task}/{task}_human.json', 'r') as f:
            human_data = json.load(f)
        if task == 'Code':
            human_data = [s[0] + s[1] for s in human_data]
        
        # Load LLM-generated data.
        with open(f'{base_dir}/{task}/{task}_{generative_model}.json', 'r') as f:
            ai_data = json.load(f)


    return human_data, ai_data


class llama_entropy(object):
    def __init__(self,
                 model_name_or_path: str = "codellama/CodeLlama-7b-hf",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512
                 ) -> None:
        # assert_tokenizer_consistency(model_name_or_path)
        self.observer_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   output_hidden_states=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
      
        self.observer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed


    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length", 
            truncation=True, 
            max_length=self.max_token_observed,
            padding_side="right", 
            return_token_type_ids=False).to(self.observer_model.device)
        
        return encodings
    
    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        output = self.observer_model(**encodings.to(DEVICE_1))
        logits = output.logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return logits
   
    def perplexity(self,encoding: transformers.BatchEncoding,
                logits: torch.Tensor,
                median: bool = False,
                temperature: float = 1.0):
        shifted_logits = logits[..., :-1, :].contiguous() / temperature 
        shifted_labels = encoding.input_ids[..., 1:].contiguous() 
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous() 
        
      
        if median:
            ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                    masked_fill(~shifted_attention_mask.bool(), float("nan"))) 
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        else:
            
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                shifted_attention_mask)
            
            ppl = ppl.to("cpu").float().numpy()

        return ppl

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch) 
        logits = self._get_logits(encodings)
        ppl = self.perplexity( encodings, logits)
        token_len = (encodings.attention_mask).sum(dim=1).cpu().numpy()
        return ppl.tolist()
        

