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
# %%
"""
# トークナイザーの実装
## encoder
テキストをIDに変換する
1. テキストの分割：記号も含める
2. トークン化：未知の単語は`<|unk|>`に
3. IDを付与：
## decoder
辞書を用いて、文字列に戻す
"""
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizer(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
decoded_res = tokenizer.decode(ids)
print("idsencode(text):", ids)
print("decode(ids):", decoded_res)
# %%
"""
# tiktokenを用いたトークナイザー
バイトペアエンコーディング（BPE）に基づいたトークナイザーで、GPTシリーズにも用いられる
"""
import tiktoken

tokenizeere = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
# %%
