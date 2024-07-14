import pandas as pd
import os
import random
import json
from utils.refusal import DENY_LIST
from utils.questions import GENERAL_QUESTIONS, UNRELATED_QUESTIONS, UNRELATED_SIMILAR_QUESTIONS

random.seed(1)

def format_question(string):
    string = string.lower()
    string = string[0].upper() + string[1:]
    if not string.endswith('?'):
        string += '?'
    return string

def choose_question(args, image, optimize):
    match args.case:
        case 1:
            return [image.target_question] if optimize else image.target_question
        case 2:
            return [UNRELATED_QUESTIONS[image.id]] if optimize else UNRELATED_QUESTIONS[image.id]
        case 3:
            dataset_path = os.path.dirname(args.file_path)
            with open(os.path.join(dataset_path, "similar_questions.json"), 'r') as file:
                similar_questions = json.load(file)
            return similar_questions[image.id] if optimize else image.target_question
        case 4:
            return UNRELATED_SIMILAR_QUESTIONS[image.id] if optimize else UNRELATED_QUESTIONS[image.id]
        case 5:
            return random.sample(GENERAL_QUESTIONS, args.num_shadow_qs) if optimize else image.target_question
        case 6:
            return random.sample(GENERAL_QUESTIONS, args.num_shadow_qs) if optimize else UNRELATED_QUESTIONS[image.id]

class Image:
    def __init__(
        self,
        id,
        position,
        target_question,
        reference_answer,
        injected_prompt=None
    ):
        self.id = id
        self.position = position
        self.target_question = format_question(target_question)
        self.reference_answer = reference_answer
        # Attacked
        self.injected_prompt = injected_prompt if injected_prompt else random.choice(DENY_LIST)
        print(repr(self.injected_prompt))
        self.target_answer = None


class ImageData:
    def __init__(
        self,
        file_path,
        images_path,
        database_name="VQA",
        load_sample=False,
    ):
        self.database_name = database_name
        self.data_path = file_path
        self.images_path = images_path
        self.data = (
            self.load_data(file_path)
            if not load_sample
            else self.load_sample(file_path)
        )
        self.images = (
            self.process_data() if not load_sample else self.process_sample([])
        )

    def load_data(self, file_path):
        return pd.read_excel(file_path)

    def process_data(self):
        images = []
        index = 0  # key
        for _, row in self.data.iterrows():
            image_path = row["Image"]
            target_question = str(row["Target_Question"]).replace('\n', ' ')
            reference_answer = str(row["Reference_Answer"]).replace('\n', ' ')

            images.append(
                Image(
                    id=index,
                    position=os.path.join(self.images_path, image_path),
                    target_question=target_question,
                    reference_answer=reference_answer,
                )
            )

            index += 1

        return images

    def load_sample(self, file_path):
        return pd.read_csv(file_path, sep="\t")

    def process_sample(self, filter_list=[]):
        images = []

        for _, row in self.data.iterrows():
            index = row["Index"]
            if index in filter_list:
                continue
            image_path = row["Image"]
            target_question = row["Target_Question"]
            reference_answer = str(row["Reference_Answer"])
            injected_prompt = str(row["Injected_Prompt"])

            images.append(
                Image(
                    id=index,
                    position=image_path,
                    target_question=target_question,
                    reference_answer=reference_answer,
                    injected_prompt=injected_prompt,
                )
            )
        print(
            f"Loaded {len(images)} images, {len(filter_list)} already completed and filtered"
        )

        return images

    def save_data(self, file_path):
        data_list = []
        for image in self.images:
            data_list.append(
                {
                    "Index": image.id,
                    "Image": image.position,
                    "Target_Question": image.target_question,
                    "Reference_Answer": image.reference_answer,
                    "Injected_Prompt": image.injected_prompt,
                }
            )
        sample_data = pd.DataFrame(data_list)
        sample_data.to_csv(file_path, index=False, sep="\t")
        return self.images

class ImagePrompt:
    def __init__(self, image_id, image_tensor, image_size, question_ids, injected_ids):
        self.id = image_id
        self.image_tensor = image_tensor
        self.image_size = image_size
        self.question_ids = question_ids
        self.injected_ids = injected_ids
