import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

items = []
with open("./data/sft_data.json", "r", encoding="utf8")as f:
    for line in f:
        item = json.loads(line)
        items.append({"prompt": item["query"], "completion": item["answer"]})
dataset = Dataset.from_list(items)

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,torch_dtype=torch.float16)
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                    ],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
sft_config = SFTConfig(output_dir="/tmp",
                       neftune_noise_alpha=10,
                       per_device_train_batch_size=1,
                       max_seq_length=100,
                       num_train_epochs=10,
                       logging_steps=10,
                       logging_strategy="steps")
response_template = "<|start_header_id|>assistant<|end_header_id""|>\n\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    data_collator=collator

)

trainer.train()