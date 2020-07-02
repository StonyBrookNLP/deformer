#!/usr/bin/env zsh
set -e

base_dir=gs://bert-gcs/deformer

# for s9
# 1.1,0.5,0.7 2,0.7,0.6 0.5,0.2,0.3 0.7,0.2,0.4

# for s10
# 2,0.1,0.6 1.6,0.3,0.4 1.7,0.9,0.2 1,0.1,0.2

allowed_task=("squad_v1.1" "mnli" "qqp" "boolq" "race")
if [[ ${allowed_task[(r)$1]} == "$1" ]] ; then
  task=$1
else
  echo "task $1 is not valid, available:" "${allowed_task[@]}"
  exit 1
fi

allowed_size=("base" "large")
if [[ ${allowed_size[(r)$2]} == "$2" ]] ; then
  size="$2"
else
  echo "size $2 is not valid, available:" "${allowed_size[@]}"
  exit 1
fi

sep_layer=$3
if [[ ${size} == "base" ]] ; then
  allowed_layers=$(seq 1 1 11)
  large_params=""
else
  allowed_layers=$(seq 1 1 23)
  large_params="hidden_size=1024 intermediate_size=4096 num_heads=16 num_hidden_layers=24"
fi

if [[ ${task} == "race" ]] ; then
  large_params="num_choices=4 ${large_params}"
fi

if [[ ${allowed_layers[(i)$3]} -le ${#allowed_layers} ]] ; then
  sep_layer=$3
else
  echo "sep_layer $3 is not valid, available:" "${allowed_layers[@]}"
  exit 1
fi

if [[ "$4" == "" ]]; then
  tpu="bert-tpu"
else
  tpu="$4"
fi

if [[ "$5" == "" ]]; then
  mode=""
else
  mode="$5"
fi

# getting hyper parameters from redis
cur_param=$(redis-cli -p 60001 get "hp-${task}-${size}")
hp=("${(@s/,/)cur_param}");
lr=${hp[1]};
n_epochs=${hp[2]};
bs=${hp[3]};

echo "start task=${task}, size=${size} sep_layer=${sep_layer}, tpu=${tpu}, mode=${mode}"
echo "params_queue=queue:params-${task}, results_queue=queue:results-${task}"
echo "learning_rate=${lr} epochs=${n_epochs} train_batch_size=${bs}"
echo "ready to start in 5 seconds..."
sleep 5


ctpu up --tpu-size=v3-8 --tpu-only --name="$tpu" --noconf
export tpu_name=$tpu

while true; do
  echo;
  cur_param=$(redis-cli -p 60001 lpop "queue:params-${task}")
  echo "cur_param=$cur_param"
  if [[ "$cur_param" == "" ]]; then
    echo "no params available, will exit in 30 seconds..."
    sleep 30;
    break;
  else
    arr=("${(@s/,/)cur_param}");
    a=${arr[1]};
    b=${arr[2]};
    c=${arr[3]};
    echo "sep-layer=${sep_layer} a=${a} b=${b} c=${c}"

    tensorboard --logdir="${base_dir}/ckpt/sbert_${size}/${task}/s${sep_layer}a${a}b${b}c${c}" --port=60007 &
    tb_id=$!

    eval "${large_params}" bfloat16=yes sep_layers="${sep_layer}" use_tpu=yes data_dir=${base_dir} \
    learning_rate="${lr}" epochs="${n_epochs}" train_batch_size="${bs}" keep_checkpoint_max=1 \
    distill=yes kd_alpha="${a}" kd_mse_beta="${b}" ce_gama="${c}" \
    checkpoint_dir="${base_dir}/ckpt/sbert_${size}${mode}/${task}/s${sep_layer}a${a}b${b}c${c}" \
    init_checkpoint="${base_dir}/ckpt/init/${task}_sbert_${size}${mode}.ckpt" \
    python train.py -m sbert -t "${task}" 2>&1 | \
    tee data/"${task}-sbert-${size}-sep${sep_layer}a${a}b${b}c${c}"-train.log
    kill $tb_id || echo 0

    eval "${large_params}" bfloat16=yes sep_layers="${sep_layer}" \
     use_tpu=yes data_dir=${base_dir} use_replace_map=no \
    output_file="${base_dir}/predictions/${task}-sbert-${size}-sep${sep_layer}a${a}b${b}c${c}".json \
    init_checkpoint="${base_dir}/ckpt/sbert_${size}${mode}/${task}/s${sep_layer}a${a}b${b}c${c}" \
    python eval.py -m sbert -t  "${task}" 2>&1 | \
    tee data/"${task}-sbert-${size}-sep${sep_layer}a${a}b${b}c${c}"-eval.log

    metric=$(grep -Po 'metric=\K[0-9]{2}\.[0-9]{3}' \
    data/"${task}-sbert-${size}-sep${sep_layer}a${a}b${b}c${c}"-eval.log)

    redis-cli -p 60001 rpush "queue:results-${task}" "${metric}"
    echo "eval result=${metric}, put to results_queue=queue:results-${task}"

    echo "waiting bayes optimizer to return params"
    sleep 30
  fi
done

ctpu pause  --tpu-only --name="$tpu" --noconf
