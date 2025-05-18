import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig

def load_model(model_id):
    """
    Loads model from Hugging Face Hub.
    """
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    # Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 quantization_config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=False, # use full memory 
                                                 attn_implementation=attn_implementation) # which attention version to use

    if not use_quantization_config: 
        llm_model.to("cuda")





if __name__ == "__main__":
    
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))

    if gpu_memory_gb < 5.1:
        print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb < 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
        use_quantization_config = False 
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb > 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False 
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")

    
    # Load the model
    load_model(model_id)
    