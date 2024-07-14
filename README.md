# MLLM-Refusal

## Instructions for reimplementing MLLM-Refusal
### 1. Install the required packages
```bash
git clone https://github.com/Sadcardation/MLLM-Refusal.git
cd MLLM-Refusal
conda env create -f environment.yml
conda activate mllm_refusal
```

### 2. Prepare the datasets
Check the datasets from the following links:
- [**CelebA**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): [Download Link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=drive_link) (Validation)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html): [Download Link](https://cs.stanford.edu/people/dorarad/gqa/download.html) (Test Balanced)
- [TextVQA](https://textvqa.org/): [Download Link](https://textvqa.org/dataset/) (Test)
- [VQAv2](https://visualqa.org/): [Download Link](https://visualqa.org/download.html) (Validation)

Download the datasets and place them in the `datasets` directory. The directory structure should look like this:

```
MLLM-Refusal
└── datasets
    ├── CelebA
    │   ├── Images
    │   │   ├── 166872.jpg
    │   │   └── ...
    │   ├── sampled_data_100.xlsx
    │   └── similar_questions.json
    ├── GQA
    │   ├── Images
    │   │   ├── n179334.jpg
    │   │   └── ...
    │   ├── sampled_data_100.xlsx
    │   └── similar_questions.json
    ├── TextVQA
    │   ├── Images
    │   │   ├── 6a45a745afb68f73.jpg
    │   │   └── ...
    │   ├── sampled_data_100.xlsx
    │   └── similar_questions.json
    └── VQAv2
        ├── Images
        │   └── mscoco
        │       └── val2014
        │           ├── COCO_val2014_000000000042.jpg
        │           └── ...
        ├── sampled_data_100.xlsx
        └── similar_questions.json   
```
`sampled_data_100.xlsx` contains the 100 sampled image-question for each dataset. `similar_questions.json` contains the similar questions for each questions in the sampled data.

### 3. Prepare the MLLMs
Clone the MLLM repositories and place them in the `models` directory, and follow the install instructions for each MLLM. Include corresponding `utils` directory in each MLLM's directory. 
- [**LLaVA-1.5**](https://github.com/haotian-liu/LLaVA)

    Additional instructions:

    1. Add 
        ```
        config.mm_vision_tower = "openai/clip-vit-large-patch14"
        ```
        below [here](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/language_model/llava_llama.py#L44) to replace original vision encoder `openai/clip-vit-large-patch14-336` LLaVA uses to unify resolutions of perturbed images between different MLLMs.

- [**MiniGPT-4**](https://github.com/Vision-CAIR/MiniGPT-4)
- [**InstructBLIP**](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)
- [**Qwen-VL-Chat**](https://github.com/QwenLM/Qwen-VL)

    Additional instructions:

    1. Add
        ```python
        if kwargs:
            kwargs['visual']['image_size'] = 224
        ```
        below [here](https://huggingface.co/Qwen/Qwen-VL-Chat/blob/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/configuration_qwen.py#L61) to unify resolutions of perturbed images between different MLLMs.

    2. Add
        ```python
        image_emb = None,
        ```
        as addtional [argument for forward function of QWenModel](https://huggingface.co/Qwen/Qwen-VL-Chat/blob/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/modeling_qwen.py#L538-L553), and replace [this line of code](https://huggingface.co/Qwen/Qwen-VL-Chat/blob/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/modeling_qwen.py#L565) with
        ```python
        images = image_emb if image_emb is not None else self.visual.encode(images)
        ```
        so that image embeddings can directly be passed to the forward function.

### 4. Run the experiments
To produced images with refusal perturbation on 100 sampled images for VQAv2 dataset on LLaVA-1.5 with three different types of shadow questions under default settings, run the following command:
```bash
./attack.sh
```
The results will be saved under LLaVA-1.5's directory.

### 5. Evaluate the results
To evaluate the results, run the following command:
```bash
./evaluate.sh
```
with corresponding MLLM's directory and the name of the result directory. Refusal Rates will be printed on the terminal and saved in the each result directory.