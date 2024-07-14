import argparse
from utils.data_handler import ImageData, ImagePrompt, choose_question
from utils.function import save_args, append_image_to_csv, filter_list, load_args
import os
from torchvision.utils import save_image
import importlib


def launch(args, data_part):
    
    import sys
    parent_dir = os.path.abspath(os.path.join(args.model_position, os.pardir, os.pardir))
    sys.path.append(parent_dir)
    sys.path.append(parent_dir + "/" + args.model_position)
    pipeline = importlib.import_module(args.model_position.replace("/", ".") + ".utils.pipeline")

    csv_file_path = f'{args.model_position}/{args.log_dir}/data/log.csv'
    tokenizer, model, image_processor = pipeline.model_init(args)
    for image in data_part:
        questions_list = choose_question(args, image, optimize=True)
        question_token = [pipeline.prompt_init(args, tokenizer, model, question) for question in questions_list]
        image_size, image_tensor = pipeline.image_init(image.position, image_processor, model)
        injected_ids = pipeline.injected_init(image.injected_prompt, tokenizer, model)
        
        whole = ImagePrompt(image.id, image_tensor, image_size, question_token, injected_ids)
        
        image_tensor_final = pipeline.attack(args, model, whole)
        
        attacked_image_path = f'{args.model_position}/{args.log_dir}/images/{args.database_name}_{image.id}.bmp'
        save_image(image_tensor_final, attacked_image_path)
        pre_eval_question = choose_question(args, image, optimize=False)
        image.target_answer = pipeline.output_gen(args, tokenizer, image_processor, model, attacked_image_path, pre_eval_question)
        append_image_to_csv(image, csv_file_path)

def main(args):
    if not args.checkpoint:
        os.mkdir(args.model_position + f'/{args.log_dir}')
        os.mkdir(args.model_position + f'/{args.log_dir}/images')
        os.mkdir(args.model_position + f'/{args.log_dir}/meta')
        os.mkdir(args.model_position + f'/{args.log_dir}/data')
        os.mkdir(args.model_position + f'/{args.log_dir}/plots')
        image_data = ImageData(args.file_path, args.images_path, args.database_name)
        sampled_images = image_data.save_data(args.model_position + f'/{args.log_dir}/meta/sample.csv')
        save_args(args, args.model_position + f'/{args.log_dir}/meta/args.json')
    else:
        checkpoint_path = args.model_position + f'/{args.log_dir}/'
        image_data = ImageData(checkpoint_path + 'meta/sample.csv', args.images_path, args.database_name, args.prompt_policy, load_sample=True)
        sampled_images = image_data.process_sample(filter_list(checkpoint_path + 'images/'))
    print(len(sampled_images))
    launch(args, sampled_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Related
    parser.add_argument("--model", type=str, default="llava")
    # Attack Related
    parser.add_argument("--freq_split", type=float, default=1, help="frequency splits")
    parser.add_argument("--freq_mask", type=str, default="static", help="grad mask update")
    parser.add_argument("--model_position", type=str, default="cls", help="model position")
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100,
        help="specify the number of iterations for attack.",
    )
    parser.add_argument(
        "--eps", type=int, default=32, help="epsilon of the attack budget"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="step_size of the attack"
    )
    parser.add_argument("--l2norm", action="store_true")
    parser.add_argument("--optimizer", type=str, default="FGSM", help="optimizer")
    parser.add_argument("--attack_mode", type=str, default="normal", help="attack mode")
    parser.add_argument("--num_shadow_qs", type=int, default=50, help="Number of general shadow questions")
    parser.add_argument("--q_batch", type=int, default=5, help="batch size for shadow questions")
    # Job Related
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--log_dir", type=str, default="benchmarking_output", help="log directory")
    # Dataset Related
    parser.add_argument("--database_name", type=str, default="VHTest", help="Name of the dataset")
    parser.add_argument("--file_path", type=str, default="/data", help="Path to the dataset file")
    parser.add_argument("--images_path", type=str, default="/images", help="Path to root of images")
    parser.add_argument("--case", type=int, required=True, help="Case number")
    args = parser.parse_args()
    
    model_args = load_args(f"scripts/configs/{args.model}.json")
    args.__dict__.update(vars(model_args))
    
    assert not args.checkpoint or (args.checkpoint and os.path.exists(args.model_position + f'/{args.log_dir}/meta/args.json')), "Checkpoint does not exist"
        
    main(args)