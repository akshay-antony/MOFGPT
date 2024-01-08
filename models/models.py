from cProfile import label
import sys
sys.path.append("../")
from torch.cuda.amp import autocast


class LLMModel():
    def __init__(self,
                 network,
                 model_name,
                 config) -> None:
        self.network = network
        self.model_name = model_name
        self.config = config

    def forward(self,
                token_ids,
                mask_ids,
                target_token_ids,
                is_fp16):
        if self.model_name == "llama2":
            if is_fp16:
                with autocast():
                    outputs = self.network(input_ids=token_ids,
                                           attention_mask=mask_ids,
                                           labels=target_token_ids,
                                           use_cache=self.config["use_cache"],
                                           return_dict=self.config["return_dict"],
                                           output_attentions=self.config["output_attentions"],
                                           output_hidden_states=self.config["output_hidden_states"],)
            else:
                outputs = self.network(input_ids=token_ids,
                                       attention_mask=mask_ids,
                                       labels=target_token_ids,
                                       use_cache=self.config["use_cache"],
                                       return_dict=self.config["return_dict"],
                                       output_attentions=self.config["output_attentions"],
                                       output_hidden_states=self.config["output_hidden_states"],)
                
        elif self.model_name == "gpt2":
            if is_fp16:
                with autocast():
                    outputs = self.network(input_ids=token_ids,
                                           attention_mask=mask_ids,
                                           labels=token_ids,
                                           use_cache=self.config["use_cache"],
                                           return_dict=self.config["return_dict"],
                                           output_attentions=self.config["output_attentions"],
                                           output_hidden_states=self.config["output_hidden_states"],)
            else:
                outputs = self.network(input_ids=token_ids,
                                       attention_mask=mask_ids,
                                       use_cache=self.config["use_cache"],
                                       return_dict=self.config["return_dict"],
                                       output_attentions=self.config["output_attentions"],
                                       output_hidden_states=self.config["output_hidden_states"],)
        return outputs
    
if __name__ == "__main__":
    pass