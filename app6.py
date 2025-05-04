import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import requests

# ------------------- Download from Google Drive -------------------
def download_model_from_gdrive(file_id, dest_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)

# ------------------- Download Model If Needed -------------------
file_id = '1_G4H54--qshMaEVnFAUZK5deQfzX3pEU'  # Your Google Drive file ID
output_model_path = '6_bert_model.pth'

if not os.path.exists(output_model_path):
    st.info("📥 Downloading model from Google Drive...")
    try:
        download_model_from_gdrive(file_id, output_model_path)
        st.success("✅ Model download complete!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# ------------------- Load Model & Tokenizer -------------------
label_names = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]

try:
    tokenizer = BertTokenizer.from_pretrained("./6_bert_tokenizer")  # Ensure this folder is uploaded in your app directory
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    model.load_state_dict(torch.load(output_model_path, map_location=torch.device("cpu"), weights_only=False))

    model.eval()
except Exception as e:
    st.error(f"❌ Failed to load model or tokenizer: {e}")
    st.stop()

# ------------------- Streamlit UI -------------------
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
        st.info(f"🧠 Predicted Label: **{label_name}** (confidence: {confidence:.2f})")

        # Optional: Show full class probabilities
        st.subheader("Prediction Confidence per Class")
        for i, prob in enumerate(probs[0]):
            st.write(f"{label_names[i].replace('-', ' ').capitalize()}: {prob.item():.2f}")
