from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"},
          {"role": "assistant", "content": "这是由于光的散射引起的。"}]

input = tokenizer.apply_chat_template(dialog, return_tensors="pt")
input = {k: v.to("cuda") for k, v in input.items()}

#设置labels和inputs一致
input["labels"] = input["input_ids"].clone()

output = model(**input)

#获取模型的loss
loss = output.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

#保存模型
model.save_pretrained("output_dir")