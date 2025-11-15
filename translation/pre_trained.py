from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import torch
import os

token = os.getenv("HF_TOKEN", "")
if not token:
    raise RuntimeError("HF_TOKEN environment variable not set")

login(token)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

src_lang = "eng_Latn"
tgt_lang = "tur_Latn"


def translate(text):
    text = text.strip()
    if not text:
        return ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    tgt_lang_id = tokenizer.lang_code_to_id["tur_Latn"]

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
