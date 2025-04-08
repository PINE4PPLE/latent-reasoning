import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLMWithLatent
from tqdm import tqdm

from utils import load_jsonl, write_jsonl

import ray


def detokenize_with_weight(tokenizer, generated_ids, generated_weights):

    def extract_tokens_and_weights(tokens, weights):
        selected_tokens = tokens[weights > 0]
        selected_weights = weights[weights > 0]
        return selected_tokens, selected_weights

    bsz = generated_ids.shape[0]
    tokens = [[generated_ids[idx][i] for i in range(generated_ids.shape[1])]for idx in range(bsz)]
    weights = [[generated_weights[idx][i] for i in range(generated_weights.shape[1])]for idx in range(bsz)]

    filted_tokens = []
    filted_weights = []
    for t,w in zip(tokens, weights):
        f_t = []
        f_w = []
        for t1, w1 in zip(t,w):
            if t1[0] == tokenizer.eos_token_id and w1[0] == 1:
                break
            tmp_t, tmp_w = extract_tokens_and_weights(t1,w1)
            f_t.append(tokenizer.decode(tmp_t))
            f_w.append(tmp_w)
        filted_tokens.append(f_t)
        filted_weights.append(f_w)
    
    final_answer = ["".join(filted_tokens[idx]).split("</think>")[-1].strip("<｜end▁of▁sentence｜>") for idx in range(bsz)]
    
    return filted_tokens, filted_weights, final_answer

@ray.remote(num_gpus=1)
class InferenceWorker:
    def __init__(
        self, 
        checkpoint: str, 
        max_tokens: int, 
        top_p: float, 
        temperature: float, 
        latent_beam_size: int,
        change_token: str,
        ):

        print("Loading model...")
        self.model = Qwen2ForCausalLMWithLatent.from_pretrained(
            checkpoint, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16,
            device_map="cuda", 
            )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.latent_beam_size = latent_beam_size
        self.change_token = change_token

    def generate(self, prompts):

        tokenized_inputs = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True).to('cuda')

        # run inference
        outputs = self.model.generate(
            tokenized_inputs,
            max_length=self.max_tokens, 
            max_latent_length=self.max_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            change_token_id=self.tokenizer.convert_tokens_to_ids(self.change_token), 
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id, 
        )

        return detokenize_with_weight(
            self.tokenizer, 
            outputs[0], 
            outputs[1]
        )

def build_batches(data, batch_size):
    length = len(data)
    for i in range(0, length, batch_size):
        yield data[i : min(i + batch_size, length)]

def parse_args():
    parser = argparse.ArgumentParser(description="Latent reasoning script")
    # model and data
    parser.add_argument("-c", "--checkpoint_dir", type=str, default="/root/paddlejob/workspace/env_run/output/pretrained_models/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--question_file", type=str, default="/root/paddlejob/workspace/env_run/output/latent-reasoning/data/gsm8k/test.jsonl")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)

    # sampling parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_generated_samples", type=int, default=1)
    parser.add_argument("--latent_beam_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_per_inference", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # set seed for reproducibility
    torch.manual_seed(args.seed)

    # load input file
    inputs = load_jsonl(args.question_file)

    # preprocess questions
    questions = [i[args.question_key] for i in inputs]
    chat_questions = [[{"role": "user", "content": q}] for q in questions]

    batched_chat_questions = list(build_batches(chat_questions, args.batch_size))

    # initialize ray
    ray.init()

    # get available gpus
    available_gpus = int(ray.cluster_resources().get("GPU", 1))
    print(f"Available GPUs: {available_gpus}")

    # Create one worker per available GPU.
    workers = [InferenceWorker.remote(
            checkpoint=args.checkpoint_dir,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            latent_beam_size=args.latent_beam_size,
            change_token="</think>",
            ) for _ in range(available_gpus)]
    
    # futures = [worker.generate.remote(sample) for sample, worker in zip(batched_chat_questions, workers)]

    futures = []
    for idx, batch in enumerate(batched_chat_questions):
        futures.append(workers[idx % available_gpus].generate.remote(batch))

    print(len(futures))

    results = ray.get(futures)

    print(len(futures))

    final_answers = []
    n_steps = []
    for result in results:
        final_answers.extend(result[2])
        for r in result[0]:
            n_steps.append(len(r))

    for idx in range(len(final_answers)):
        inputs[idx]["response"] = [final_answers[idx]]
        inputs[idx]["n_steps"] = n_steps[idx]

    
    write_jsonl(inputs, os.path.join(args.save_dir, args.output_file))
