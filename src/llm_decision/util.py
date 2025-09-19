
import torch
import torch.nn.functional as F
from transformers import AutoConfig,AutoModelForCausalLM, AutoTokenizer,logging

# Config & device
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")
# -----------------------------
# Load model/tokenizer
# -----------------------------
def load_model_and_tokenizer(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 family has no pad token -> map pad to EOS
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.set_verbosity_info()
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        attn_implementation="eager",   
        dtype=torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer
