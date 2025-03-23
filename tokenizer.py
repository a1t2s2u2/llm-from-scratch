"""
## テキストから埋め込み表現を取得するまでの流れ
Input text → Tokenized text → Token IDs → Token embeddings
"""
# %%
"""
データをダウンロードして、中身の表示
"""
import os
import urllib.request

if not os.path.exists("datathe-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print("Preview:", raw_text[:99])
# %% Tokenize text
import re
text = "Hello, world. Is this-- a test?"
# 句読点
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Tokenized text:", preprocessed[:30])
print("len:", len(preprocessed))
# %% Convert token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("vocab_size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
# %% Simple Tokenizer V1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
decoded_res = tokenizer.decode(ids)
print("idsencode(text):", ids)
print("decode(ids):", decoded_res)
# %%
