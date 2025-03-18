# LEAVS

by [Ricardo Bigolin Lanfredi](https://github.com/ricbl).

This repository contains code and data for the [LEAVS: An LLM-based Labeler for Abdominal CT Supervision](https://arxiv.org/abs/2503.13330) paper.

## Using the LEAVS labeler

To learn about all the arguments to run the labeler, run `python src/one_load_model.py --help`.

Run it with at least 155GB VRAM. The default input arguments runs the prompt with the LEAVS method. Example:

```
python src/one_load_model.py --result_root=./test_run_results/ --single_file ./report_text.txt
```

The vllm library sometimes has trouble halting with Ctrl+C or exceptions. You might have to use Ctrl+Z, then run `ps` and use `kill -9 <process_id>` for all python processes.

Prompts employed can be seen in file cli.py in lines 410 (finding uncertainty assessment for present finding types), 423 (finding uncertainty assessment for absent finding types), 569 (finding type strings), 577 (finding type additional descriptions), 603 (part of sentence filtration prompt), 738 (urgency prompt), 746 (finding type assessment), 769 (first step of sentence filtration), 771 (second step of sentence filtration).

The outputs of the script will be stored in a csv file. 

For rows where "type_annotation" is "labels":
- -3 means uncertainty because of ambiguity of language or mention of broad anatomical area.
- -2 means not mentioned
- -1 means uncertainty because the radiologist was not sure
- 0 mean absence mentioned
- 1 means presence mentioned


## Datasets

The AMOS-MM dataset can be found at https://era-ai-biomed.github.io/amos/dataset.html#download

The amos_test_annotations.csv file contains the human annotations for the AMOS-MM test set of 200 samples that we employed.

The parsing_results_llm_test_amos.csv file contains the labeling done by LEAVS for the same set of 200 samples.

The parsing_results_llm_other_amos.csv file contains the labeling done by LEAVS for the rest of the AMOS-MM dataset.

## Requirements

It was tested with

- python                    3.10.14
- torch                     2.5.1
- pytorch-cuda              12.1
- torchvision               0.20.1
- torchaudio                2.3.1
- vllm                      0.6.4.post1
- transformers              4.47.0
- xformers                  0.0.28.post3
- vllm-flash-attn           2.5.9
- tokenizers                0.21.0
- pandas                    2.2.2
- numpy                     1.26.4
- huggingface-hub           0.24.2
- joblib                    1.4.2
- levenshtein               0.25.1
- nltk                      3.9.1
- retry                     0.9.2
- tk (for annotation tool in annotator_python_script.py)

## Citation

Cite the [LEAVS: An LLM-based Labeler for Abdominal CT Supervision](https://arxiv.org/abs/2503.13330) paper if you employ the code from this repository or the annotation data from this repository. Cite the [Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation](https://arxiv.org/abs/2206.08023) paper if you use the annotations from this repository.