from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import PreTrainedTokenizer
from PIL import Image
import torch
from typing import Tuple, List
from utils import visual_attacker

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).half().to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).half().to(images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

# From ~/.cache/huggingface/modules/transformers_modules/Qwen/Qwen-VL-Chat/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/qwen_generation_utils.py
def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

# ========================================
#             Model Initialization
# ========================================

def model_init(args, device=0):
    print(">>> Initializing Models")

    print("model = ", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=f"cuda:{device}", trust_remote_code=True, fp16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = model.transformer.visual.image_transform
    
    print("[Initialization Finished]\n")
    
    return tokenizer, model, image_processor

# ========================================
#             Prompt Preparation
# ========================================

def prompt_init(args, tokenizer, model, inp):
    inp = "Picture 1: <img>https://place/holder.jpeg</img>\n" + inp
    
    raw_text, context_tokens = make_context(
            tokenizer,
            inp,
            history=[],
            system="You are a helpful assistant.",
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format,
        )

    
    return torch.tensor([context_tokens])

# # ========================================
# #             Image Preparation
# # ========================================

def image_init(image_path, image_processor, model):
    
    image = load_image(image_path)
    image_size = image.size
    image_tensor = image_processor(image)
    
    return image_size, image_tensor

# # ========================================
# #             Injected Prompt
# # ========================================
def injected_init(injected_prompt, tokenizer, model):
    # eos_token = "<|endoftext|>"
    # injected_ids = tokenizer.encode(injected_prompt+eos_token)
    injected_ids = tokenizer.encode(injected_prompt)
    return torch.tensor([injected_ids])

def attack(args, model, inputs):
    my_attacker = visual_attacker.Attacker(args, model, [], device=model.device, is_rtp=False)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
        image_tensor_final = my_attacker.attack_method(
            inputs=inputs,
            num_iter=args.n_iters,
            lr=args.alpha,
        )
    return image_tensor_final


# # ========================================
# #             Evaluation
# # ========================================

def output_gen(args, tokenizer, image_processor, model, image_path, question):

        query = tokenizer.from_list_format([
            {'image': image_path}, # Either a local path or an url
            {'text': question},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        return repr(response)
