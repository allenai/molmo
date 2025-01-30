<div align="center">
  <img src="assets/Molmo-logo.svg" alt="Molmo Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>Molmo: Multimodal Open Language Model</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/mm_olmo/blob/release/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://molmo.allenai.org/blog">
    <img alt="Blog Post" src="https://img.shields.io/badge/Molmo-blog-F0529C">
  </a>
  <a href="https://arxiv.org/pdf/2409.17146">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2409.17146-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b">
    <img alt="PixMo (Datasets)" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-PixMo (Datasets)-yellow">
  </a>
</p>

Molmo is a repository for training and using Ai2's state-of-the-art multimodal open language models.

Here is a **[video demo](https://www.youtube.com/watch?v=spBxYa3eAlA)** of Molmo's capabilities. Try Molmo using our public **[demo](https://molmo.allenai.org)** showcasing the Molmo-7B-D model.

This codebase is based on the [OLMo codebase](https://github.com/allenai/OLMo) with the addition
of vision encoding and integrating generative evaluations.

## Release Notes

- [2024/12/05] **🔥 Molmo**: code for modeling, training and evaluation has been released. You can find detailed technical report [here](https://arxiv.org/pdf/2409.17146).

- [2024/11/27] **🔥 [PixMo](https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b)**, our new collection of datasets for pre-training and fine-tuning VLMs, has been released. PixMo consists of:

  - [**PixMo-Cap**](https://huggingface.co/datasets/allenai/pixmo-cap) (pre-training, fine-tuning): highly detailed dense caption dataset (roughly 200 words on average)
  - [**PixMo-AskModelAnything**](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) (fine-tuning): instruction-tuning data containing human-authored image-question-answer triplets
  - [**PixMo-CapQA**](https://huggingface.co/datasets/allenai/pixmo-cap-qa) (fine-tuning): synthetic instruction-tuning data, using a LLM to build QA pairs from dense captions of images
  - [**PixMo-Points**](https://huggingface.co/datasets/allenai/pixmo-points) (fine-tuning): images paired with referring expressions and annotated points, supporting grounding and counting
  - [**PixMo-Point-Explanations**](https://huggingface.co/datasets/allenai/pixmo-point-explanations) (fine-tuning): instruction-tuning data with explanations containing in-line points referring to parts of the image
  - [**PixMo-Docs**](https://huggingface.co/datasets/allenai/pixmo-docs) (fine-tuning): synthetic image-question-answer triplets about various kinds of computer-generated charts, tables, diagrams and documents. Code available [here](https://github.com/allenai/pixmo-docs).
  - [**PixMo-Clocks**](https://huggingface.co/datasets/allenai/pixmo-clocks) (fine-tuning): virtual watch faces and time annotations
  - [**PixMo-Count**](https://huggingface.co/datasets/allenai/pixmo-count) (fine-tuning): diverse images with counting QA pairs

  All datasets were constructed **without the use of VLMs**.

<div align="center">
  <img src="assets/png_version_molmo_pixmo.png" alt="Pixmo and Molmo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <p>Datasets in PixMo (left) and the capabilities they enable in Molmo (right).
  <br>
</div>

- [2024/09/24] **🔥 [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)**, a new family of open VLMs, has been released. The Molmo family consists of:

  - [**MolmoE-1B**](https://huggingface.co/allenai/MolmoE-1B-0924): a mixture of experts model with 1B (active) 7B (total)
  - [**Molmo-7B-O**](https://huggingface.co/allenai/Molmo-7B-O-0924): our most open 7B model
  - [**Molmo-7B-D**](https://huggingface.co/allenai/Molmo-7B-D-0924): our best 7B and demo model
  - [**Molmo-72B**](https://huggingface.co/allenai/Molmo-7B-D-0924): our best 72B model


## Installation

We recommend using python 3.10.
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install dependencies, run:

```bash
git clone https://github.com/allenai/molmo.git
cd molmo
pip install -e .[all]
```

For training and evaluating MolmoE-1B, please install megablocks by running `pip install git+https://github.com/Muennighoff/megablocks.git@olmoe`.


## Huggingface Models and Logs

The core models in the Molmo family released so far are:

<table>
  <tr>
    <th>Model</th>
    <th>Vision Encoder</th>
    <th>LLM</th>
    <th align="center">11-benchmark avg</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/MolmoE-1B-0924">MolmoE-1B-0924</a></td>
    <td rowspan="4"><a href="https://huggingface.co/openai/clip-vit-large-patch14-336">OpenAI CLIP ViT-L/14@336</a></td>
    <td><a href="https://huggingface.co/allenai/OLMoE-1B-7B-0924">OLMoE-1B-7B-0924</a></td>
    <td align="center">68.6</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-7B-O-0924">Molmo-7B-O-0924</a></td>
    <td><a href="https://huggingface.co/allenai/OLMo-7B-1024-preview">OLMo-7B-1024-preview</a></td>
    <td align="center">74.6</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-7B-D-0924">Molmo-7B-D-0924</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2-7B">Qwen2-7B</a></td>
    <td align="center">77.3</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-72B-0924">Molmo-72B-0924</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2-72B">Qwen2-72B</a></td>
    <td align="center">81.2</td>
  </tr>
</table>

W&B logs: [pre-training](https://wandb.ai/prior-ai2/molmo/reports/Molmo-Pre-training--VmlldzoxMDQwODE3OA), [fine-tuning](https://wandb.ai/prior-ai2/molmo/reports/Molmo-Fine-tuning--VmlldzoxMDQwOTQ4Mw)

## Data Downloading and Setup 
Molmo uses huggingface datasets for most data, therefore most 
data will be stored in the default huggingface cache. See [here](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
for how to set it. Some additional data is stored separately in the path
set by `MOLMO_DATA_DIR`. 

For example, if you want to store the data in `/data/molmo` you could set

```bash
export MOLMO_DATA_DIR=/data/molmo
export HF_HOME=/data/molmo/huggingface
```

Data can then be downloaded with:

```bash
python3 scripts/download_data.py all --n_proc 12
```

Downloading the pixmo datasets requires downloading images from URLs. The download script
will do this automatically, but it will take some time.
Downloading everything from scratch can take up to a day.
More processes can make it faster, but it also increases the risk of getting rate-limited.

Downloading can be resumed if canceled or an error occurs mid-download.

Some datasets (InfoQa and Scene-Text) require manually downloading the files.
The download scripts will throw an error if those files are not found.

Downloading the android control dataset requires additional dependencies
since it requires parsing the original tfrecords.

To download a specific dataset pass in the dataset name run:
```bash
python3 scripts/download_data.py ChartQa --n_proc 12
```

## Visualizing Data
Once downloaded, datasets can be visualized by using `scripts/dataset_visualize.py` script:

```bash
python3 scripts/dataset_visualize.py chart_qa /path/to/viz/dir
```

## Trained Models
We release model weights both after pre-training and after fine-tuning in a format compatible
with this codebase. The fine-tuned weights match the ones in the hugging face repos,
but have a slightly different format. The config files are backwards-compatible with
this repo, but also have a slightly different format. 

<table>
  <tr>
    <th>Model</th>
    <th>Pretrained</th>
    <th>Fine-Tuned</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/MolmoE-1B-0924">MolmoE-1B-0924</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/MolmoE-1B-0924-Pretrained.tar">pretrained</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/MolmoE-1B-0924.tar">fine-tuned</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-7B-O-0924">Molmo-7B-O-0924</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-O-0924-Pretrained.tar">pretrained</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-O-0924.tar">fine-tuned</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-7B-D-0924">Molmo-7B-D-0924</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-D-0924-Pretrained.tar">pretrained</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-D-0924.tar">fine-tuned</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/allenai/Molmo-72B-0924">Molmo-72B-0924</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-72B-0924-Pretrained.tar">pretrained</a></td>
    <td><a href="https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-72B-0924.tar">fine-tuned</a></td>
  </tr>
</table>

To use them, download the file and untar them. Each folder contains
the needed config file and model weights. For example:

```bash
wget https://storage.googleapis.com/oe-training-public/Molmo-0924/Molmo-7B-D-0924.tar
tar -xf Molmo-7B-D-0924.tar 
```

## Evaluation
Evaluation is done with the `launch_scripts/eval_downstream.py` script. 
FSDP can be used to evaluate large models, or for high-resolution processing. 
Note that the vLLM version of Molmo will be significantly faster for inference, but most of 
our numbers were reported using the results of this local evaluation. 

To eval on a single task pass the `task name`, or `task_name:split`:

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 text_vqa --save_to_checkpoint_dir
```

For most tasks, we evaluate with high resolution:

```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 text_vqa --save_to_checkpoint_dir --high_res --fsdp --device_batch_size=2
```

The `--fsdp` flag will use FSDP which is needed for to avoid OOMs when using high resolution.

To evaluate on our default eval set (including the 11 tasks in the paper):
```bash
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 low-res --save_to_checkpoint_dir
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py Molmo-7B-D-0924 high-res --save_to_checkpoint_dir --high_res --fsdp --device_batch_size=2
```

To get test numbers, use `low-res-test` and `high-res-test`. Some test numbers will require
re-formatting the prediction files and then submitting to test servers.

To evaluate the 72B model with this codebase you will need to run on multiple nodes
and might need to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

These scripts will save the metrics and predictions in the save directory. Future calls to the 
eval script will re-use cached metrics if they exist, to overwrite these cached metrics use
the `--overwrite` flag.

### Evaluation with VLMEvalkit
Evaluation of the HF models is also supported via [open-compass/VLMEvalkit](https://github.com/open-compass/VLMEvalKit). Check [PR#648](https://github.com/open-compass/VLMEvalKit/pull/648) for supported prompts and evaluation settings to reproduce results from the paper.
However a few datasets (e.g., PixMo-Count) are not supported.

## Pretrained Models for Initialization
Training end-to-end requires downloading the pre-trained models used to initialize Molmo.
This can be done with the script `scripts/convert_hf_to_molmo.py`

For example, to load the Qwen2 LLM and OpenAI CLIP model, run:

```bash
python3 scripts/convert_hf_to_molmo.py qwen2_7b
python3 scripts/convert_hf_to_molmo.py openai
```

The model will be downloaded from huggingface, converted into a compatible format,
and then saved into the `MOLMO_DATA_DIR` directory.

## Pre-Training
The main training script is `scripts/train.py`. To train a model you can either construct a config
file to pass to it, or call one of the higher-level helper scripts in `launch_scripts` which
will construct a low-level config from some higher-level settings and then invoke the train script for you.

To start a debugging run:

`torchrun --nproc-per-node=1 launch_scripts/train_captioner.py debug
--save_folder=/path/to/save/folder`

To train with the Qwen2 LLM and the CLIP vision encoder:

`WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb.name=run_name --wandb.entity=entity --wandb.project=project --save_folder=/path/to/save/folder`

You can use other vision encoders including SigLIP, MetaCLIP and DINOv2 with the option `--vision_backbone=model_name`.

To run without wandb, use:

`torchrun --nproc-per-node=8 launch_scripts/train_captioner.py qwen2_7b
--wandb=null --save_folder=/path/to/save/folder`

## Multitask Training
Multitask training can be done with `launch_scripts/multtask_train.py`, for example:

`WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/train_multitask_model.py 3.2-synthetic /path/to/checkpoint
--wandb.name=run_name --wandb.entity=entity --wandb.project=project
--save_folder=/path/to/save/folder
`

Here `3.2-synthetic` refers to what training mixture to use and `/path/to/checkpoint` points to a
model checkpoint to start from, typically a dense captioning model.

To launch a debug run:

`
torchrun --nproc-per-node=1 launch_scripts/train_multitask_model.py debug debug 
--save_folder=dbg --save_overwrite
`

## Training Changes
There are minor differences between the published Molmo models that we trained and what this repo will produce.

- Image URLs might fail to download, which will cause the amount of data to shrink slightly
- PixMo-Clocks is not used by default, it requires a more complex download script that 
we are still considering how to port.

## Multi-Node
Execute the `torchrun` commands on each node with the [appropriate args](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)
should allow multi-node training or evaluation. 

We recommend ensuring the data is downloaded and then using the environment variable 
`HF_DATASETS_OFFLINE=1` to ensure the nodes don't flood HF with requests as they all initialize 
and then potentially get rate limited.

## Citation

```bibtex
@article{molmo2024,
  title={Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models},
  author={Matt Deitke and Christopher Clark and Sangho Lee and Rohun Tripathi and Yue Yang and Jae Sung Park and Mohammadreza Salehi and Niklas Muennighoff and Kyle Lo and Luca Soldaini and Jiasen Lu and Taira Anderson and Erin Bransom and Kiana Ehsani and Huong Ngo and YenSung Chen and Ajay Patel and Mark Yatskar and Chris Callison-Burch and Andrew Head and Rose Hendrix and Favyen Bastani and Eli VanderBilt and Nathan Lambert and Yvonne Chou and Arnavi Chheda and Jenna Sparks and Sam Skjonsberg and Michael Schmitz and Aaron Sarnat and Byron Bischoff and Pete Walsh and Chris Newell and Piper Wolters and Tanmay Gupta and Kuo-Hao Zeng and Jon Borchardt and Dirk Groeneveld and Jen Dumas and Crystal Nam and Sophie Lebrecht and Caitlin Wittlif and Carissa Schoenick and Oscar Michel and Ranjay Krishna and Luca Weihs and Noah A. Smith and Hannaneh Hajishirzi and Ross Girshick and Ali Farhadi and Aniruddha Kembhavi},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```
