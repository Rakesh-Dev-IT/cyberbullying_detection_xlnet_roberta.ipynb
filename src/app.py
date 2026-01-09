import torch
import torch.nn.functional as F
import gradio as gr

from transformers import XLNetTokenizer, RobertaTokenizer
from model import HybridXLNetRoBERTa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = HybridXLNetRoBERTa()
model.load_state_dict(
    torch.load("hybrid_xlnet_roberta_model.pth", map_location=device)
)
model.to(device)
model.eval()

# Load tokenizers
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet_tokenizer")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta_tokenizer")

def predict(text):
    xlnet_enc = xlnet_tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=128
    )
    roberta_enc = roberta_tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=128
    )

    with torch.no_grad():
        logits = model(
            xlnet_enc["input_ids"],
            xlnet_enc["attention_mask"],
            roberta_enc["input_ids"],
            roberta_enc["attention_mask"]
        )
        pred = torch.argmax(F.softmax(logits, dim=1)).item()

    return "Not Cyberbullying" if pred == 0 else "Cyberbullying"

def predict_ui(text):
    result = predict(text)
    color = "red" if result == "Cyberbullying" else "green"

    return f"""
    <div style='padding:1.5em;background:{color};
    color:white;font-size:22px;font-weight:bold;
    border-radius:12px;text-align:center'>
        Prediction: {result}
    </div>
    """

with gr.Blocks() as app:
    gr.Markdown("# 🔍 Cyberbullying Detection (XLNet + RoBERTa)")
    inp = gr.Textbox(lines=4, placeholder="Enter text")
    out = gr.HTML()
    btn = gr.Button("Predict")

    btn.click(predict_ui, inp, out)

app.launch()
