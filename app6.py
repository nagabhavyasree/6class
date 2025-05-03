
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import gdown
import os

# Google Drive download
file_id = '1_G4H54--qshMaEVnFAUZK5deQfzX3pEU'  # Replace with your own file ID
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output_model_path = 'bert_model.pth'

if not os.path.exists(output_model_path):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(url, output_model_path, quiet=False)
    st.success("✅ Download complete!")

# Define label names
label_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("./6_bert_tokenizer")  # assumes this is already uploaded/extracted
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
model.load_state_dict(torch.load(output_model_path, map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
st.title("📰 Fake News Classifier (6 Classes)")
st.write("Enter a news statement, and we'll classify it into one of six fact-checking labels.")

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

        label_name = label_names[label].replace("-", " ").capitalize()
        st.info(f"🧠 Predicted Label: {label_name} (confidence: {confidence:.2f})")

        # Optional: show full class probabilities
        st.subheader("Prediction Confidence per Class")
        for i, prob in enumerate(probs[0]):
            st.write(f"{label_names[i].replace('-', ' ').capitalize()}: {prob.item():.2f}")
