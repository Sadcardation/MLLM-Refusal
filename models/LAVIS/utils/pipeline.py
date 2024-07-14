from PIL import Image
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.models import load_preprocess
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
    model_cls = registry.get_model_class(args.model_path)

    # load model
    model = model_cls.from_pretrained(model_type=args.model_base).to('cuda:{}'.format(device))
    cfg = OmegaConf.load(model_cls.default_config_path(args.model_base))
    vis_processors, _ = load_preprocess(cfg.preprocess)
    
    image_processor = vis_processors["eval"]
    tokenizer = model.llm_tokenizer
    
    return tokenizer, model, image_processor

# ========================================
#             Prompt Preparation
# ========================================

def prompt_init(args, tokenizer, model, inp):
    input_ids = tokenizer(inp, return_tensors="pt", add_special_tokens=False).input_ids[0]
    # print(f'decode: {tokenizer.decode(input_ids, skip_special_tokens=False).lstrip()}')
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
    
    injected_ids = tokenizer(injected_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
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

    _, image_tensor = image_init(image_path, image_processor, model)
    
    samples = {
        "image": image_tensor,
        "prompt": query,
    }
    
    output = model.generate(
        samples=samples,
        use_nucleus_sampling=True,
    )

    return repr(output[0])
