from chatgpt.trainer.strategies import NaiveStrategy
from chatgpt.nn import GPTActor
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from datasets import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

strategy = NaiveStrategy()

model = GPT2LMHeadModel.from_pretrained('gpt2')
# strategy.load_model(model, "./PPOModel/actor_learn_10_checkpoint.pt")
model.eval()
model.to(device)
text = "Xiao Ming is a primary school student. He likes playing games"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

indexed_tokens = tokenizer.encode(text)
print(tokenizer.decode(indexed_tokens))
tokens_tensor = torch.tensor([indexed_tokens]).to(device)

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)

stopids = tokenizer.convert_tokens_to_ids(["."])[0]
past = None
for i in range(20):
    with torch.no_grad():
        output, past = model(tokens_tensor, past_key_values=past, return_dict=False)

    token = torch.argmax(output[..., -1, :])

    indexed_tokens += [token.tolist()]

    if stopids == token.tolist():
        break
    tokens_tensor = token.unsqueeze(0)

sequence = tokenizer.decode(indexed_tokens)

print("sequence :", sequence)


# data = load_dataset('Dahoas/rm-static')
# train_data = data["train"].select(range(100))
# train_data_prompt = train_data["prompt"]
#
# train_token = tokenizer(train_data_prompt, padding="max_length", max_length=15)
# input_ids = train_token["input_ids"]
# print(type(input_ids))