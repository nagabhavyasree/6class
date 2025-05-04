import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import gdown
import os

# Setup download
file_id = '1_G4H54--qshMaEVnFAUZK5deQfzX3pEU'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output_model_path = '6_bert_model.pth'

# Download if not exists
if not os.path.exists(output_model_path):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(url, output_model_path, quiet=False)
    st.success("✅ Download complete!")

# Load tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained("./6_bert_tokenizer")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    model.load_state_dict(torch.load(output_model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    st.error(f"❌ Failed to load model or tokenizer: {e}")
    st.stop()

# UI
st.title("📰 Fake News Classifier (6 Classes)")
user_input = st.text_area("Enter news text:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label].item()

        label_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        label_name = label_names[label].replace("-", " ").capitalize()
        st.info(f"🧠 Predicted Label: {label_name} (confidence: {confidence:.2f})")

        # Show class confidences
        st.subheader("Prediction Confidence per Class")
        for i, prob in enumerate(probs[0]):
            st.write(f"{label_names[i].replace('-', ' ').capitalize()}: {prob.item():.2f}")
