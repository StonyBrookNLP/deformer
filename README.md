# **DeFormer**: **De**composing Pre-trained Trans**former**s for Faster Question Answering

This repo is the code for the [DeFormer paper](https://awk.ai/assets/deformer.pdf)  (Accepted to ACL 2020).

<img style="margin:auto;width:50%" src="https://awk.ai/assets/deformer-sketch.png" alt="deformer"/>

<!--ts-->
   * [Installation](#installation)
   * [Usage](#usage)
      * [Data Processing](#dataset-processing)
        * [download dataset](#dataset-processing)
        * [convert dataset](#dataset-processing)
        * [generate examples](#dataset-processing)
      * [Training and Evaluation](#training-and-evaluation)
      * [Experimenting](#experimenting)
        * [tune ebert](#tune-ebert)
        * [tune sbert](#tune-sbert)
      * [Profiling](#profiling)
      * [Demo](#demo)
      * [Tools](#tools)
      * [Handy Commands](#handy-commands)
   * [FAQ](#faq)
   * [Citation](#citation)
<!--te-->

## Installation

Tested on Ubuntu 16.04, 18.04 and macOS. (Windows should also work, but not tested)

You can create a separate python environment, e.g. `virtualenv -p python3.7 .env`
and activate it by `source .env/bin/activate`

1. Requirements: Python>=3.5 and TensorFlow >=1.14.0,<2.0

2. `pip install "tensorflow>=1.14.0,<2.0"` or `pip install tensorflow-gpu==1.15.3` (for GPU)

3. `pip install -r requirements.txt`

**NOTE**: we call `ebert` for DeFormer BERT version, and `sbert` for applying KD & LRS in the paper.

For XLNet, you can check [my fork](https://github.com/csarron/xlnet) for a reference implementation.

## Usage

### Dataset Processing

#### downloading datasets to `data/datasets`
  - GLUE: [link](https://gist.githubusercontent.com/csarron/2a7f5da27f45e7e0795c9946f7c95f76/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py)
  - SQuAD v1.1: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) and 
  [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
  - [RACE dataset](https://www.cs.cmu.edu/~glai1/data/race/)

<!--
  - squad v2.0: [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json) and 
  [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
  - hotpot qa: [train_v1.1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json) and 
  [dev_distractor_v1.json](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json)
-->

   the dataset dir should look like below (use `tree -L 2 data/datasets`):
````log
data/datasets
├── BoolQ
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── mnli
│   ├── dev_mismatched.tsv
│   └── train.tsv
├── qqp
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── RACE
│   ├── dev
│   ├── test
│   └── train
└── squad_v1.1
    ├── dev-v1.1.json
    └── train-v1.1.json
````

#### convert to DeFormer format

convert:

````bash
deformer_dir=data/datasets/deformer
mkdir -p ${deformer_dir}

# squad v1.1
for version in 1.1; do
    data_dir=data/datasets/squad_v${version}
    for split in dev train; do
        python tools/convert_squad.py ${data_dir}/${split}-v${version}.json \
        ${deformer_dir}/squad_v${version}-${split}.jsonl
    done
done

# mnli
data_dir=data/datasets/mnli
python tools/convert_pair_dataset.py ${data_dir}/train.tsv ${deformer_dir}/mnli-train.jsonl -t mnli
python tools/convert_pair_dataset.py ${data_dir}/dev_matched.tsv ${deformer_dir}/mnli-dev.jsonl  -t mnli

# qqp
data_dir=data/datasets/qqp
python tools/convert_pair_dataset.py ${data_dir}/train.tsv ${deformer_dir}/qqp-train.jsonl -t qqp
python tools/convert_pair_dataset.py ${data_dir}/dev.tsv ${deformer_dir}/qqp-dev.jsonl -t qqp

# boolq
data_dir=data/datasets/BoolQ
python tools/convert_pair_dataset.py ${data_dir}/train.jsonl ${deformer_dir}/boolq-train.jsonl -t boolq
python tools/convert_pair_dataset.py ${data_dir}/val.jsonl ${deformer_dir}/boolq-dev.jsonl -t boolq

# race
data_dir=data/datasets/RACE
python tools/convert_race.py ${data_dir}/train ${deformer_dir}/race-train.jsonl
python tools/convert_race.py ${data_dir}/dev ${deformer_dir}/race-dev.jsonl

````

split 10% of train for tuning hyper-parameters:
  
````bash
cd ${deformer_dir}

cat squad_v1.1-train.jsonl | shuf > squad_v1.1-train-shuf.jsonl
head -n8760 squad_v1.1-train-shuf.jsonl > squad_v1.1-tune.jsonl
tail -n78839 squad_v1.1-train-shuf.jsonl > squad_v1.1-train.jsonl

cat boolq-train.jsonl | shuf > boolq-train-shuf.jsonl
head -n943 boolq-train-shuf.jsonl > boolq-tune.jsonl
tail -n8484 boolq-train-shuf.jsonl > boolq-train.jsonl

cat race-train.jsonl | shuf > race-train-shuf.jsonl
head -n8786 race-train-shuf.jsonl > race-tune.jsonl
tail -n79080 race-train-shuf.jsonl > race-train.jsonl

cat qqp-train.jsonl | shuf > qqp-train-shuf.jsonl
head -n36385 qqp-train-shuf.jsonl > qqp-tune.jsonl
tail -n327464 qqp-train-shuf.jsonl > qqp-train.jsonl

cat mnli-train.jsonl | shuf > mnli-train-shuf.jsonl
head -n39270 mnli-train-shuf.jsonl > mnli-tune.jsonl
tail -n353432 mnli-train-shuf.jsonl > mnli-train.jsonl

```` 

#### download BERT vocab

download [bert.vocab](https://github.com/StonyBrookNLP/deformer/releases/download/v1.0/bert.vocab) to `data/res`

#### generating training and evaluation examples:

  usage: `python prepare.py -h`
  
  - e.g., convert `squad_v1.1` for `bert`: 
    ````bash
    python prepare.py -m bert -t squad_v1.1 -s dev
    python prepare.py -m bert -t squad_v1.1 -s tune
    python prepare.py -m bert -t squad_v1.1 -s train -sm tf
    ````

  - e.g., convert `squad_v1.1` for `xlnet`: 
    ````bash
    model=xlnet
    task=squad_v1.1
    python prepare.py -m ${model} -t ${task} -s dev
    python prepare.py -m ${model} -t ${task} -s train -sm tf
    ````

  - convert all available tasks and all models:
    ````bash
    for model in bert ebert; do
      for task in squad_v1.1 mnli qqp boolq race; do
        python prepare.py -m ${model} -t ${task} -s dev
        python prepare.py -m ${model} -t ${task} -s tune
        python prepare.py -m ${model} -t ${task} -s train -sm tf
      done
    done
    ````


### Training and Evaluation

#### SQuAD 1.1 Quickstart

download original fine-tuned BERT-base checkpoints from [bert-base-squad_v1.1.tgz](https://github.com/StonyBrookNLP/deformer/releases/download/v1.0/bert-base-squad_v1.1.tgz)
and DeFormer fine-tuned version from [ebert-base-s9-squad_v1.1.tgz](https://github.com/StonyBrookNLP/deformer/releases/download/v1.0/ebert-base-s9-squad_v1.1.tgz)

`python eval.py -m bert -t squad_v1.1 2>&1 | tee data/bert-base-eval.log`
example output:
````log
INFO:2020-07-01_15:36:30.339:eval.py:65: model.ckpt-8299, em=80.91769157994324, f1=88.33819502660548, metric=88.33819502660548
````

`python eval.py -m ebert -t squad_v1.1 2>&1 | tee data/ebert-base-s9-eval.log`

example output:
````log
INFO:2020-07-01_15:39:15.418:eval.py:65: model.ckpt-8321, em=79.12961210974456, f1=86.99636369864814, metric=86.99636369864814
````

#### Train and Eval

See `config/*.ini` for customizing training and evaluation script

- train: `python train.py` specify model by `-m`(`--model`), task by `-t`(`--task`), eval is similar.
see below example commands for `boolq`: 

    ````bash
    # for running on tpu, should specify gcs bucket data_dir, and set use_tpu to yes
    # also need to set tpu_name=<some_ip_or_just_name> if not exported to environment
    base_dir=<your google cloud storage bucket>
    data_dir=${base_dir} use_tpu=yes \
    python train.py -m bert -t boolq 2>&1 | tee data/boolq-bert-train.log
    
    data_dir=${base_dir} use_tpu=yes \
    python eval.py -m bert -t boolq 2>&1 | tee data/boolq-bert-eval.log
  
    # for list of models and list of tasks
    for task in boolq mnli qqp squad_v1.1; do
      for model in bert ebert; do
        data_dir=${base_dir} use_tpu=yes \
        python train.py -m ${model} -t ${task} 2>&1 | tee data/${task}-${model}-train.log
        
        data_dir=${base_dir} use_tpu=yes \
        python eval.py -m ${model} -t ${task} 2>&1 | tee data/${task}-${model}-eval.log
      done
    done
    ````

- BERT wwm large:

    ````bash
    base_dir=<your google cloud storage bucket>
    for t in boolq qqp squad_v1.1 mnli; do
      use_tpu=yes data_dir=${base_dir} \
      learning_rate=1e-5 epochs=2 keep_checkpoint_max=1 \
      init_checkpoint=${base_dir}/ckpt/init/wwm_uncased_large/bert_model.ckpt \
      checkpoint_dir=${base_dir}/ckpt/bert_large/${t} \
      hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24 \
      python train.py -m bert -t ${t} 2>&1 | tee data/${t}-large-train.log
    
      data_dir=${base_dir} use_tpu=yes init_checkpoint="" \
      checkpoint_dir=${base_dir}/ckpt/bert_large/${t} \
      hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24 \
      python eval.py -m bert -t ${t} 2>&1 | tee data/${t}-large-eval.log
    done || exit 1
  ````

### Experimenting

#### Tune EBert

- fine tuning for separation at different layers for bert base:

    ````bash
    for t in boolq qqp mnli squad_v1.1; do
      for n in `seq 1 1 11`; do
        echo "n=${n}, t=${t}"
        base_dir=${base_dir}

        sep_layers=${n} use_tpu=yes data_dir=${base_dir} keep_checkpoint_max=1 \
        checkpoint_dir="${base_dir}/ckpt/separation/${t}/ebert_s${n}" \
        python train.py -m ebert -t ${t} 2>&1 | tee data/${t}-base-sep${n}-train.log

        sep_layers=${n} use_tpu=yes data_dir=${base_dir} init_checkpoint="" \
        checkpoint_dir="${base_dir}/ckpt/separation/${t}/ebert_s${n}" \
        python eval.py -m ebert -t ${t} 2>&1 | tee data/${t}-base-sep${n}-eval.log
      done
    done
    ````
  
- fine tuning for separation at different layers for wwm large bert:
    
    ````bash
    for t in boolq qqp mnli squad_v1.1; do
      for n in `seq 10 1 23`; do
        echo "n=${n}, t=${t}"
        base_dir=${base_dir}
      
        sep_layers=${n} use_tpu=yes data_dir=${base_dir} \
        learning_rate=1e-5 epochs=2 keep_checkpoint_max=1 \
        init_checkpoint=${base_dir}/ckpt/init/wwm_uncased_large/bert_model.ckpt \
        checkpoint_dir=${base_dir}/ckpt/separation/${t}/ebert_large_s${n} \
        hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24 \
        python train.py -m ebert -t ${t} 2>&1 | tee data/${t}-large-sep${n}-train.log
      
        sep_layers=${n} use_tpu=yes data_dir=${base_dir} init_checkpoint="" \
        checkpoint_dir=${base_dir}/ckpt/separation/${t}/ebert_large_s${n} \
        hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24 \
        output_file=${base_dir}/predictions/${t}-large-sep${n}-dev.json \
        python eval.py -m ebert -t ${t} 2>&1 | tee data/${t}-large-sep${n}-eval.log
      done || exit 1
    done || exit 1
  ````

#### Tune SBert

- [ ] training script needs further verification (due to migrated from old codebase)

- sbert procedure, first get ebert_s0, then merge bert_base and ebert_s0 checkpoints 
using `tools/merge_checkpoints.py` to get initial checkpoint for sbert, then run the training.

    ````bash
    base_dir=gs://xxx
    init_dir="data/ckpt/init"
    large_model="${init_dir}/wwm_uncased_large/bert_model.ckpt"
    base_model="${init_dir}/uncased_base/bert_model.ckpt"
    
    for t in squad_v1.1 boolq qqp mnli; do
      mkdir -p data/ckpt/separation/${t}
      
      # sbert large init
      large_init="data/ckpt/separation/${t}/ebert_large_s0"
      gsutil -m cp -r "${base_dir}/ckpt/separation/${t}/ebert_large_s0" data/ckpt/separation/${t}/
      
      python tools/merge_checkpoints.py -c1 "${large_init}" \
      -c2 "${large_model}" -o ${init_dir}/${t}_sbert_large.ckpt
      gsutil -m cp -r "${init_dir}/${t}_sbert_large.ckpt*" "${base_dir}/ckpt/init"
      
      # sbert large init from ebert_large_s0 all
      python tools/merge_checkpoints.py -c1 "${large_init}" -c2 "${large_model}" \
      -o ${init_dir}/${t}_sbert_large_all.ckpt -fo 
      gsutil -m cp -r "${init_dir}/${t}_sbert_large_all.ckpt*" "${base_dir}/ckpt/init"
    
      # sbert large init from ebert_large_s0 upper, e.g. 20
      python tools/merge_checkpoints.py -c1 "${large_init}" -c2 "${large_model}" \
      -o ${init_dir}/${t}_sbert_large_upper20.ckpt -fo -fou 20
      gsutil -m cp -r "${init_dir}/${t}_sbert_large_upper20.ckpt*" "${base_dir}/ckpt/init"
    
      # sbert base init
      base_init="data/ckpt/separation/${t}/ebert_s0"
    
      gsutil -m cp -r "${base_dir}/ckpt/separation/${t}/ebert_s0" data/ckpt/separation/${t}/
      python tools/merge_checkpoints.py -c1 "${base_init}" -c2 "${base_model}" \
      -o ${init_dir}/${t}_sbert_base.ckpt
      gsutil -m cp -r "${init_dir}/${t}_sbert_base.ckpt*" "${base_dir}/ckpt/init"
    
      python tools/merge_checkpoints.py -c1 "${base_init}" -c2 "${base_model}" \
      -o ${init_dir}/${t}_sbert_base_all.ckpt -fo 
      gsutil -m cp -r "${init_dir}/${t}_sbert_base.ckpt*" "${base_dir}/ckpt/init"
    
      python tools/merge_checkpoints.py -c1 "${base_init}" -c2 "${base_model}" \
      -o ${init_dir}/${t}_sbert_base_upper9.ckpt -fo -fou 9
      gsutil -m cp -r "${init_dir}/${t}_sbert_base.ckpt*" "${base_dir}/ckpt/init"
    done || exit 1
    ````

- sbert finetuning: 

    ````bash
    # squad_v1.1, search 50 params for bert large separated at layer 21
    python tools/explore_hp.py -p data/sbert-squad-large.json -n 50 \
    -s large -sp 1.4 0.3 0.8 -hp 5e-5,3,32 2>&1 | tee data/sbert-squad-explore-s21.log
    ./search.sh squad_v1.1 large 21 bert-tpu2
    
    # race search 50
    python tools/explore_hp.py -p data/race-sbert-s9.json -n 50 -t race 2>&1 | \
    tee data/race-sbert-explore-s9.log
    
    ./search.sh race base 9
    ````


### Profiling
- profile model flops:

    ````bash
    for task in race boolq race qqp mnli squad_v1.1; do
      for size in base large; do
        profile_dir=data/log2-${task}-${size}-profile
        mkdir -p "${profile_dir}"
              
        if [[ "${task}" == "mnli" ]]; then
          cs=1 # cache_segment
        else
          cs=2
        fi

        if [[ ${size} == "base" ]] ; then
          allowed_layers="9 10" # $(seq 1 1 11)
          large_params=""
        else
          allowed_layers="20 21" #$(seq 1 1 23)
          large_params="hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24"
        fi

        if [[ ${task} == "race" ]] ; then
          large_params="num_choices=4 ${large_params}"
        fi

        # bert 
        eval "${large_params}" python profile.py -m bert -t ${task} -pm 2>&1 | \
        tee ${profile_dir}/bert-profile.log

        # ebert 
        for n in "${(@s/ /)allowed_layers}"; do
          eval "${large_params}" sep_layers="${n}" \
          python profile.py -m ebert -t ${task} -pm 2>&1 | \
          tee ${profile_dir}/ebert-s${n}-profile.log
      
          eval "${large_params}" sep_layers="${n}" \
          python profile.py -m ebert -t ${task} -pm -cs ${cs} 2>&1 | \
          tee ${profile_dir}/ebert-s${n}-profile-cache.log
        done
      done
    done
    ````

- benchmarking inference latency:

    ````bash
    python profile.py -npf -pt -b 32 2>&1 | tee data/batch-time-bert.log
    python profile.py -npf -pt -b 32 -m ebert -cs 2 2>&1 | tee data/batch-time-ebert.log
    ````

- analyze bert, ebert, sbert:

    ````bash
    python analyze.py -o data/qa-outputs -m bert 2>&1 | tee data/ana-bert.log
    python tools/compute_rep_variance.py data/qa-outputs -n 20
    
    python tools/compare_rep.py data/qa-outputs -m sbert
    python tools/compare_rep.py data/qa-outputs -m ebert
    ````

### Demo

 - run infer: `python infer_qa.py -m bert` (add `-e` for eager mode)
<!-- - serve qa on a server: `python serve_qa.py`, then use `python tools/ask_question.py` -->

### Tools

- `tools/get_dataset_stats.py`: get dataset statistics (length of tokens mainly)
- `tools/inspect_checkpoint.py`: print variable info in checkpoints (support monitoring variables during training)
- `tools/rename_checkpoint_variables.py`: rename variable names in checkpoint (add `-dr` for dry run)
e.g. `python tools/rename_checkpoint_variables.py "data/ckpt/bert/mnli/" -p "bert_mnli" "mnli" -dr`
- `tools/visualize_model.py`: visualize TensorFlow model structure given inference graph

### Handy Commands

- redis

  ````bash
  redis-cli -p 60001 lrange queue:params 0 -1
  redis-cli -p 60001 lrange queue:results 0 -1
  redis-cli -p 60001 lpop queue:params
  redis-cli -p 60001 rpush queue:results 89.532
  ````
  
- gcloud sdk for TPU access: `pip install --upgrade google-api-python-client oauth2client`

- TPU start: `ctpu up --tpu-size=v3-8 --tpu-only --name=bert-tpu --noconf` 
(can support tf version, e.g.`--tf-version=1.13`)

- TPU stop: `ctpu pause  --tpu-only --name=bert-tpu --noconf`


- move instances: `gcloud compute instances move bert-vm --zone us-central1-b --destination-zone us-central1-a`

- upload and download: 
  ````bash
  cd data
  # upload
  gsutil -m cp -r datasets/qqp/ebert "gs://xxx/datasets/qqp/ebert"
  gsutil -m cp -r datasets/qa/ebert "gs://xxx/datasets/qa/ebert"
  gsutil -m cp -r datasets/mnli/ebert "gs://xxx/datasets/mnli/ebert"
  gsutil -m cp -r "datasets/qa/bert/hotpot-*" "gs://xxx/datasets/qa/bert"

  # download
  gsutil -m cp -r "gs://xxx/datasets/qqp/ebert" qqp/ebert
  
  cd data/ckpt
  # download
  gsutil -m cp -r "gs://xxx/ckpt/bert/qa/model.ckpt-8299*" bert/qa/
  gsutil -m cp -r "gs://xxx/ckpt/ebert_s9/qa/model.ckpt-8321*" ebert_s9/qa/
  gsutil -m cp -r "gs://xxx/ckpt/ebert_s9/mnli/model.ckpt-18407*" ebert_s9/mnli/
  gsutil -m cp -r "gs://xxx/ckpt/ebert_s9/qqp/model.ckpt-17055*" ebert_s9/qqp/
  
  function dl()
  {
    num=$2
    for suffix in meta index data-00000-of-00001; do
      gsutil cp gs://xxx/ckpt/$1/model.ckpt-${num}.${suffix} .
    done;
    echo model_checkpoint_path: \"model.ckpt-${num}\" > checkpoint
  }
  
  ````

## FAQ

If you have any question, please create an issue.

## Citation

If you find our work useful to your research, please consider using the following citation:

````bib
@inproceedings{cao-etal-2020-deformer,
    title = "{D}e{F}ormer: Decomposing Pre-trained Transformers for Faster Question Answering",
    author = "Cao, Qingqing  and
      Trivedi, Harsh  and
      Balasubramanian, Aruna  and
      Balasubramanian, Niranjan",
    booktitle = "Proceedings of the 58th Annual Mdeformering of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.411",
    pages = "4487--4497",
}
````

