# File sturcture


### 0. Model classes
- `generator.py`: wrapper arround any LLM used to generate text from a prompt
- `detector.py`: wrapper around any BERT type (bi-directional encoder) used to classify text as human or LLM generated

### 1. Generating dataset of fake LLM responses and human reponses
- `generate_fake_true_dataset.py`: takes an LLM and generate responses using prompts from a given dataset

### 2. Training an LLM to detect LLM generated responses
- `train_detector.py`: finetune a pretrained detector on the dataset generated using `generate_fake_true_dataset.py` 
