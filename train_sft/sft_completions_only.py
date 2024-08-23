import functools
import json
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset
import torch


class SFTDataset(Dataset):

    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.file_path = file_path
        self.examples = self._load_data(self.file_path)
        self.tokenizer = tokenizer

    @staticmethod
    def _load_data(file_path):
        items = []
        with open(file_path, "r", encoding="utf8")as f:
            for line in f:
                item = json.loads(line)
                items.append(item)
        return items

    def __getitem__(self, index):
        example = self.examples[index]
        dialog = [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": example["query"]},
                  {"role": "assistant", "content": example["answer"]}]
        chat = tokenizer.apply_chat_template(dialog, tokenize=False)
        return chat

    def __len__(self):
        return len(self.examples)


model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
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
model.print_trainable_parameters()
model.to("cuda")

optimizer = torch.optim.AdamW(model.parameters())


def sft_collate(batch, tokenizer, end_str, max_length):
    end_str = "<|start_header_id|>assistant<|end_header_id""|>\n\n"
    inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    input_len = len(input_ids[0])
    end_ids = tokenizer(end_str)["input_ids"]
    end_id_len = len(end_ids)
    loss_mask = []
    for input_id in input_ids:
        for i in range(len(input_id) - end_id_len, -1, -1):
            if input_id[i:i + end_id_len] == end_ids:
                mask = [1] * (input_len - 1)
                mask[:i + end_id_len - 1] = [0] * (i + end_id_len - 1)
                loss_mask.append(mask)
                break
            if i == 0:  # 所有回答部分都被截断
                loss_mask.append([0] * (input_len - 1))
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    loss_mask = torch.tensor(loss_mask)
    return inputs, loss_mask


collate_fn = functools.partial(sft_collate,
                               tokenizer=tokenizer,
                               end_str="<|start_header_id|>assistant<|end_header_id""|>\n\n",
                               max_length=50)

sft_dataset = SFTDataset("./data/sft_data.json", tokenizer)
data_loader = DataLoader(sft_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
epoch = 10

for i in range(epoch):
    for inputs, loss_mask in data_loader:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        loss_mask = loss_mask.to("cuda")
        logits = model(**inputs).logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss_mask = loss_mask.reshape(-1)

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = loss * loss_mask
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss.item())