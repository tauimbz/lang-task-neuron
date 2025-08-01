# Amplification of Language-Specific Neurons
This repository is an implementation of our paper  [Unveiling the Influence of Amplifying Language-Specific Neurons](https://arxiv.org/abs/2507.22581)

## Dataset
The reconstructed MLAMA dataset to calculate LSS is available at [inayarhmns/MLAMA-dod-185](https://huggingface.co/datasets/inayarhmns/MLAMA-dod-185)

---
### Environment setup
```
pip install -r requirements.txt
```
This repository provides connecting with kaggle account to store the datas. If using kaggle, move the kaggle.json file inside this directory and run:
```
python setup_kaggle.py
```
### 1. Get the neuron activations
To get the neurons activations, run `get_activations.py`. For command examples, refer to [commands_example/get_act/](commands_example/get_act/).

### 2. Identify Language-Specific and Language-Activated Neurons

To get language-specific neurons, run `lape.py`. The example commands are available at  [commands_example/lape/xx](commands_example/lape/xx).

To get language-activated neurons, run `raw_act_neurons.py`. The example commands are available at  [commands_example/raw_act_neurons](commands_example/raw_act_neurons).

### 3. Compute LSS Score
To compute LSS score using the reconstructed MLAMA dataset, refer to `intervention_dod.py`. The example commands are in [commands_example/intervention](commands_example/intervention). 

### Task Evaluation Intervention
To intervene on tasks evaluation and language modeling performance, run `intervention_tasks.py`. Here are some example commands:
- Perplexity: [commands_example/int_flores/ppl](commands_example/int_flores/ppl)
- Translation: [commands_example/int_flores/bleu/](commands_example/int_flores/bleu/) 
- XWinograd: [commands_example/int_xwinograd/xx/lape/](commands_example/int_xwinograd/xx/lape/)
- XCOPA: [commands_example/int_xcopa/xx/lape/](commands_example/int_xcopa/xx/lape/)
- Include-lite: [commands_example/int_include/xx/lape/](commands_example/int_include/xx/lape/)


## Citation
```
@misc{rahmanisa2025unveilinginfluenceamplifyinglanguagespecific,
      title={Unveiling the Influence of Amplifying Language-Specific Neurons}, 
      author={Inaya Rahmanisa and Lyzander Marciano Andrylie and Krisna Mahardika Ihsani and Alfan Farizki Wicaksono and Haryo Akbarianto Wibowo and Alham Fikri Aji},
      year={2025},
      eprint={2507.22581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.22581}, 
}
```
