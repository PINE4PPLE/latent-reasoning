import os 
import sys 
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLMWithLatent

if __name__ == "__main__":
    model_path = '/root/paddlejob/workspace/env_run/output/pretrained_models/DeepSeek-R1-Distill-Qwen-7B'

    # load model
    model = Qwen2ForCausalLMWithLatent.from_pretrained(model_path).to(torch.device("cuda:0"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    question = "What is the result of the following calculation: 3 * (4 + 5)?"

    chat = [
        {
            'role': 'user',
            'content': question
        }
    ]

    text_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    print(text_input)

    inputs = tokenizer([text_input, text_input], return_tensors="pt").to(torch.device("cuda:0"))

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    change_token_id = tokenizer.convert_tokens_to_ids('</think>')

    print(tokenizer.decode(inputs['input_ids'][0]))

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=256, 
            max_latent_length=256,
            temperature=0.8, 
            top_p=0.95, 
            change_token_id=change_token_id, 
            eos_token_id=eos_token_id, 
            pad_token_id=pad_token_id,
            )

    print(outputs["generated_ids"][0].shape)
    print(outputs["generated_ids"][1].shape)

