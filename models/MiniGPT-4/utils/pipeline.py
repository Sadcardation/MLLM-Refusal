from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import CONV_VISION_LLama2
import torch

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

# ========================================
#             Model Initialization
# ========================================

def model_init(args, device=0):
    print(">>> Initializing Models")

    print("model = ", args.model_path)
    args.options = None
    if args.model_path == "minigpt4":
        args.cfg_path = args.model_position + "/eval_configs/minigpt4_llama2_eval.yaml"
    cfg = Config(args)
    
    model_config = cfg.model_cfg
    model_config.device_8bit = device
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(device))
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    image_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    tokenizer = model.llama_tokenizer
    
    return tokenizer, model, image_processor

# ========================================
#             Prompt Preparation
# ========================================

def prompt_init(args, tokenizer, model, inp):
    conv = CONV_VISION_LLama2.copy()
    conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
            and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], inp])
    else:
        conv.append_message(conv.roles[0], inp)
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return input_ids

# # ========================================
# #             Image Preparation
# # ========================================

def image_init(image_path, image_processor, model):
    image = load_image(image_path)
    image_size = image.size
    image_tensor = image_processor(image).unsqueeze(0).to(model.device)
    
    return image_size, image_tensor

# # ========================================
# #             Injected Prompt
# # ========================================
def injected_init(injected_prompt, tokenizer, model):
    injected_ids = tokenizer(injected_prompt+tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return injected_ids

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

def output_gen(args, tokenizer, image_processor, model, image_path, query):

    image_attacked_size, image_attacked_tensor = image_init(image_path, image_processor, model)
    input_ids = prompt_init(args, tokenizer, model, query)
    input_prompt = tokenizer.decode(input_ids, skip_special_tokens=False).lstrip()
    image_emb, _ = model.encode_img(image_attacked_tensor)
    inputs_embs = model.get_context_emb(input_prompt, [image_emb])
    
    with torch.inference_mode() and model.maybe_autocast():
        output_token= model.llama_model.generate(
            inputs_embeds=inputs_embs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )[0]
    outputs = tokenizer.decode(output_token, skip_special_tokens=True)
    return repr(outputs)
