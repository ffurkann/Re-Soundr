import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
dataset = load_dataset("opus100", "en-tr")

def tokenize_data(batch):
    src_texts = [item["en"] for item in batch["translation"]]
    tgt_texts = [item["tr"] for item in batch["translation"]]
    return tokenizer(
        src_texts,
        text_target=tgt_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True
    )

tokenized_dataset = dataset.map(tokenize_data, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=16)

class translation_model(nn.Module):
    def __init__(self):
        super(translation_model, self).__init__()
        self.embeddings = nn.Embedding(tokenizer.vocab_size, 768, padding_idx=tokenizer.pad_token_id)
        self.pos_embeddings = nn.Embedding(768, 768)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.out = nn.Linear(768, tokenizer.vocab_size)

    def forward(self, x, y):
        x_embed = self.embeddings(x)
        x_pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x_embed = x_embed + self.pos_embeddings(x_pos)

        y_embed = self.embeddings(y)
        y_pos = torch.arange(0, y.size(1), device=y.device).unsqueeze(0)
        y_embed = y_embed + self.pos_embeddings(y_pos)

        tgt_seq_len = y.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=y.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

        memory = self.encoder(x_embed)
        output = self.decoder(y_embed, memory, tgt_mask=tgt_mask)

        return self.out(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = translation_model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(15):
    model.train()
    for batch in train_loader:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        output = output.reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    for batch in test_loader:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        output_logits = model(src, tgt[:, :-1])
        predicted_ids = output_logits.argmax(-1)

        for i in range(2):
            src_text = tokenizer.decode(src[i], skip_special_tokens=True)
            tgt_text = tokenizer.decode(tgt[i], skip_special_tokens=True)
            pred_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
            print(f"EN: {src_text}")
            print(f"TR_true: {tgt_text}")
            print(f"TR_pred: {pred_text}\n")
        break