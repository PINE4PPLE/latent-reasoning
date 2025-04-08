#!/bin/bash

code_dir=/root/paddlejob/workspace/env_run/output/latent-reasoning
python_env=/root/miniconda3/envs/latent/bin/python

checkpoint_dir=$1

model_name=$(basename ${checkpoint_dir})

export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES="${1:-0}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# export CUDA_VISIBLE_DEVICES="${1:-0,1}"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

type=(all)

save_dir=${code_dir}/eval_results/gsm8k/${model_name}

mkdir -p $save_dir

pkill -f occupy.py

temperature=1.0
top_p=0.95
sample_num=1
latent_beam_size=5
max_tokens=8192

for t in ${type[@]}; do

    question_path=/root/paddlejob/workspace/env_run/output/latent-reasoning/data/gsm8k/test.jsonl

    output_file=test.jsonl.prediction.with_${model_name}.temp_${temperature}_top-p_${top_p}_latent-beam-size_${latent_beam_size}_n-sample_1_max-tokens_${max_tokens}_seed_42.jsonl

    ${python_env} -u utils/batch_inference.py \
        --question_file ${question_path} \
        --question_key question \
        --save_dir ${save_dir} \
        --temperature ${temperature} \
        --top_p ${top_p} \
        --checkpoint_dir ${checkpoint_dir} \
        --max_tokens 8192 \
        --seed 42 \
        --latent_beam_size ${latent_beam_size} \
        --output_file ${output_file} \
        --batch_size 16 \

    ${python_env} utils/evaluate.py \
        --input_file ${save_dir}/${output_file} \
        --model ${checkpoint_dir} \
        --save_file > ${save_dir}/result-temp_${temperature}-top-p_${top_p}.txt

done

cd /root/paddlejob/workspace/env_run/output/occupy
${python_env} occupy.py &