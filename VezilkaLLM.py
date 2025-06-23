import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Please ensure a CUDA-compatible GPU is present.")

model_id = "finki-ukim/VezilkaLLM"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda",
    torch_dtype=torch.float16  # Use float16 for stability
)

from transformers.generation import LogitsProcessor
class CleanLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, logits):
        logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.tensor(-1e9, device=logits.device), logits)
        logits = torch.clamp(logits, min=-1e9, max=1e9)
        return logits

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    logits_processor=[CleanLogitsProcessor()]
)

prompt = "Зборот „пријателство“ значи "

outputs = generator(
    prompt,
    max_new_tokens=500,  
    do_sample=False,     
    num_beams=1,         
    return_dict_in_generate=True  
)

generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(generated_text)