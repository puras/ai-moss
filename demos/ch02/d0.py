import re
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item .strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)

vocab = {token: integer for integer, token in enumerate(all_tokens)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

print("---")
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    if i >= 50:
        break