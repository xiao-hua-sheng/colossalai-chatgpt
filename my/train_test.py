import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from example import text2
# load tokenizer and model
pretrained_model = "IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese"

special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model,
    do_lower_case=True,
    max_length=512,
    truncation=True,
    additional_special_tokens=special_tokens,
)
config = T5Config.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model, config=config)
model.resize_token_embeddings(len(tokenizer))
model.eval()

# tokenize

encode_dict = tokenizer(text2, max_length=512, padding='max_length',truncation=True)

inputs = {
  "input_ids": torch.tensor([encode_dict['input_ids']]).long(),
  "attention_mask": torch.tensor([encode_dict['attention_mask']]).long(),
  }

# generate answer
logits = model.generate(
  input_ids = inputs['input_ids'],
  max_length=100,
  early_stopping=True,
  )

logits=logits[:,1:]
predict_label = [tokenizer.decode(i,skip_special_tokens=True) for i in logits]
print(predict_label)

# model output: 正面
