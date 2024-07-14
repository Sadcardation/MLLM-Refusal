import torch
from llava.model.builder import load_pretrained_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
from models.LLaVA.utils import visual_attacker

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
    # disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    device = torch.device("cuda:{}".format(device))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=device,
    )
    model.eval()
    print("[Initialization Finished]\n")
    
    return tokenizer, model, image_processor

# ========================================
#             Prompt Preparation
# ========================================

def prompt_init(args, tokenizer, model, inp):
    conv_mode = "llava_v1"

    conv = conv_templates[conv_mode].copy()

    if model.config.mm_use_im_start_end:
        inp = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        )
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
        
    return input_ids

# # ========================================
# #             Image Preparation
# # ========================================

def image_init(image_path, image_processor, model):
    image = load_image(image_path)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    return image_size, image_tensor

# # ========================================
# #             Injected Prompt
# # ========================================
def injected_init(injected_prompt, tokenizer, model):
    injected_ids = (
        tokenizer_image_token(
            injected_prompt+tokenizer.eos_token, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)[:, 1:]
    )
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
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_attacked_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            # streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    outputs = outputs.replace("<s> ", '')
    outputs = outputs.replace("</s>", '')
    return repr(outputs)
