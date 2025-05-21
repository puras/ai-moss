import re
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print("Total number of charactor: ", len(raw_text))
print(raw_text[:99])

text = "Hello, world. Is this-- a test?"
ret = re.split(r'([,.:;?_!"()\']|--|\s)', text)
ret = [item for item in ret if item.strip()]
print(ret)


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item .strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

print("---")
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    if i >= 50:
        break