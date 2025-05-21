import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strs = tokenizer.decode(integers)
print(strs)

ids = tokenizer.encode("Akwirw ier")
print(ids)
print(tokenizer.decode(ids))