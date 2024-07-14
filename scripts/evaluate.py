import os
import pandas as pd
import argparse
import re
from query import gpt4_evaluation
from utils.function import load_args
import sys
import importlib
from tqdm import tqdm
import ast
from utils.questions import UNRELATED_QUESTIONS

api_keys = ["API_KEY"]

def deny_evaluation(row, candidate_index):
    target = row[f'Attacked_Answer_{candidate_index}'].strip('\'"') if candidate_index != -1 else row[f'Attacked_Answer']
    target = re.sub('^[^a-zA-Z]*', '', target)
    score = gpt4_evaluation(target,api_keys)
    return 1 if score == 1 or target == '' else 0

def output_candidate_gen(row, args, model_args, directory, tokenizer, model, image_processor, query):
    
    attacked_image_path = os.path.join(directory, 'images', f"{args.dataset}_{row['Image']}.bmp")
    target_question = row['Target_Question']
    
    if args.case in [1, 3, 5]:
        question = ast.literal_eval(target_question)
    else:
        question = target_question
        
    output = query(model_args, tokenizer, image_processor, model, attacked_image_path, question)
    output = output.replace("<s> ", "")
    print(repr(output))
        
    return output
    
def evaluate_csv_files_dos(args, directory):
    # Get a list of CSV files in the specified directory
    csv_data_directory = os.path.join(directory, 'data')
    output_csv_file = os.path.join(directory, f'combined-{args.model}-{args.case}.csv')
    score_txt_file = os.path.join(directory, f'score-{args.model}-{args.case}.txt')
    csv_files = [os.path.join(csv_data_directory, file) for file in os.listdir(csv_data_directory) if file.endswith('.csv')]

    # Read and combine all CSV files into a single DataFrame
    combined_data = pd.concat([pd.read_csv(file, sep='\t') for file in csv_files], ignore_index=True)
    
    if args.need_query:
        if args.case in [2, 4, 6]:
            combined_data['Target_Question'] = combined_data.apply(lambda row: UNRELATED_QUESTIONS[row['Image']], axis=1)
        
        model_args = load_args(f"scripts/configs/{args.model}.json")
        
        parent_dir = os.path.abspath(os.path.join(model_args.model_position, os.pardir, os.pardir))
        sys.path.append(parent_dir)
        sys.path.append(parent_dir + "/" + model_args.model_position)
        pipeline = importlib.import_module(model_args.model_position.replace("/", ".") + ".utils.pipeline")
        tokenizer, model, image_processor = pipeline.model_init(model_args)

        tqdm.pandas(desc="Generating Candidates")
        for i in range(args.candidates):
            combined_data[f'Attacked_Answer_{i}'] = combined_data.progress_apply(output_candidate_gen, axis=1, args=(args, model_args, directory, tokenizer, model, image_processor, pipeline.output_gen))
            
    target_asr = 0
    if args.need_query:
        for i in range(args.candidates):
            combined_data[f'TARGET_ATTACK_{i}'] = combined_data.apply(deny_evaluation, axis=1, args=(i,))
            candidate_asr = combined_data[f'TARGET_ATTACK_{i}'].sum() / len(combined_data)
            target_asr += candidate_asr
            print(f"Candidate {i} Target ASR: {combined_data[f'TARGET_ATTACK_{i}'].sum()} / {len(combined_data)} = {candidate_asr:.6f}\n")
            with open(score_txt_file, 'a') as f:
                f.write(f"Candidate {i} Target ASR: {combined_data[f'TARGET_ATTACK_{i}'].sum()} / {len(combined_data)} = {candidate_asr:.6f}\n")
    else:
        args.candidates = 1
        combined_data[f'TARGET_ATTACK'] = combined_data.apply(deny_evaluation, axis=1, args=(-1,))
        target_asr = combined_data[f'TARGET_ATTACK'].sum() / len(combined_data)
    target_asr /= args.candidates
    print(f"Total Target ASR: {target_asr:.6f}")
    with open(score_txt_file, 'a') as f:
        f.write(f"Total Target ASR: {target_asr:.6f}\n")
    # Save the combined data to the specified output file
    combined_data.to_csv(output_csv_file, sep='\t', index=False)
    print(f"Combined CSV file saved as {output_csv_file}")
    return target_asr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default='models/LLaVA')
    parser.add_argument("--result_index", type=str, default='')
    parser.add_argument("--candidates", type=int, default=1, help="Number of candidates")
    parser.add_argument("--need_query", action="store_true", default=False)
    parser.add_argument("--case", type=int, default=3)
    parser.add_argument("--model", type=str, default='llava', help='Eval on which model')
    parser.add_argument("--dataset", type=str, default='VQA')
    
    # case 1: shadow question is exactly user question from dataset, user question is image related (same with shadow question)
    # case 2: shadow question is exactly user question from dataset, user question is image irrelevant (sampled from commonsense_qa)
    # case 3: shadow questions are similar user questions generated from gpt4, user question is image related question from dataset
    # case 4: shadow questions are similar user questions genreated from gpt4, user question is image irrelevant (sampled from commonsense_qa)
    # case 5: shadow questions are general user questions generated from gpt4, user question is image related question from dataset
    # case 6: shadow questions are general user questions genreated from gpt4, user question is image irrelevant (sampled from commonsense_qa)
    
    args = parser.parse_args()
    welcome = f"================= Eval On Model {args.model} For Case {args.case} ================="
    
    directory_path = os.path.join(args.results_path, args.result_index)
    score_txt_file = os.path.join(directory_path, f'score-{args.model}-{args.case}.txt')
    with open(score_txt_file, 'a') as f:
        f.write(welcome + "\n")
    target_asr = evaluate_csv_files_dos(args, directory_path)
    with open(score_txt_file, 'a') as f:
        f.write("=" * len(welcome) + "\n")