# PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2212.02802-b31b1b.svg)](https://arxiv.org/abs/2402.12842)
![Method](./figures/main_figure.png)

> **PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning**<br>
> Gyeongman Kim, Doohyuk Jang, Eunho Yang <br>

>**Abstract**: <br>
> Recent advancements in large language models (LLMs) have raised concerns about inference costs, increasing the need for research into model compression. While knowledge distillation (KD) is a prominent method for this, research on KD for generative language models like LLMs is relatively sparse, and the approach of distilling student-friendly knowledge, which has shown promising performance in KD for classification models, remains unexplored in generative language models. To explore this approach, we propose PromptKD, a simple yet effective method that utilizes prompt tuning - for the first time in KD - to enable generative language models to transfer student-friendly knowledge. Unlike previous works in classification that require fine-tuning the entire teacher model for extracting student-friendly knowledge, PromptKD achieves similar effects by adding a small number of prompt tokens and tuning only the prompt with student guidance. Extensive experiments on instruction-following datasets using the GPT-2 model family show that PromptKD achieves state-of-the-art performance while adding only 0.0007% of the teacher's parameters as prompts. Further analysis suggests that distilling student-friendly knowledge alleviates exposure bias effectively throughout the entire training process, leading to performance enhancements.


## Requirements

See `install.sh`
```
conda create -n promptkd python=3.8
conda activate promptkd
bash install.sh
```

### Pretrained models

In order to use this project you need to download pretrained models and datasets. 
(Many of them are from MiniLLM github repository)

Use the `download_requirements.sh` script
```
bash download_requirements.sh
```
This script downloads 
- Model(GPT-2, OPT, Llama) checkpoints for reproducibility in `checkpoints` folder. 
    - Here, we provide PromptKD checkpoints via huggingface hub ([Link](https://huggingface.co/collections/gmkim/promptkd-66728dc78171db46e7fb7bcd))
- Data for training and evaluation in `data` and `processed_data` folder.


## Evaluation

Reproduce the performance of the model checkpoint used in the paper.

```
bash scripts/{model}/eval/run_eval.sh . checkpoints/{model}/train/{method}/{model}-{size}
```
You can choose `{model}` in `{gpt2, opt, llama}`, and `{method}` in `{sft, kd, seqkd, minillm, promptkd}`.

For GPT-2, you can choose `{size}` in `{base, medium, large}`.  
For OPT, you can choose `{size}` in `{1.3B, 2.7B, 6.7B}`.  
For Llama, put `7B` in `{size}`.


## Training

Training scripts are as below.
```
bash scripts/{model}/{method}/{method}_{student_size}.sh # For training
bash scripts/{model}/eval/run_eval.sh . results/{model}/train/{method}/{model}-{student_size}-{teacher_size}/{your_exp_folder}/best_rougeL # For evaluation
```
You can choose `{model}` in `{gpt2, opt, llama}`, and `{method}` in `{sft, kd, seqkd, minillm, promptkd}`.

Determine `{student_size}` and `{teacher_size}` by checking the `scripts` folder.  
Also, put your experiment output folder in `{your_exp_folder}` by checking the `results/{model}/train` folder.

We use 8x NVIDIA GeForce RTX 3090 GPUs for GPT-2, and 4x NVIDIA A100 80 GB (PCIe) GPUs for OPT and Llama model.



### Change Model Parallel Size
You can increase/decrease the tensor parallel sizes with
```
python3 tools/convert_mp.py \
    --input_path checkpoints/llama/train/sft/llama-13B \
    --source_mp_size 4 \
    --target_mp_size 1 \
    --model_type llama # choose from opt and llama
```



## Credits
MiniLLM implementation:  
https://github.com/microsoft/LMOps/tree/main/minillm  
License (MIT) https://github.com/microsoft/LMOps/blob/main/LICENSE  
