import gdown
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

# Download the model from Google Drive
file_id = '1KAFrkjH2gI6c-AyF2-mAZ6yIBb1P0EqN'  # Replace with your actual file ID
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'bert_model.pth'

# Download the file from Google Drive
gdown.download(url, output, quiet=False)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(output, map_location=torch.device("cpu")))
model.eval()

# Define the class labels (e.g., you can adjust them according to your use case)
class_labels = ["False", "True"]  # Adjust this based on your model's classification labels

# Streamlit UI
st.title("Fake News Classifier")
st.write("Enter a news statement, and we'll tell you whether it's likely **True** or **False**.")

user_input = st.text_area("Enter news text:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Print class probabilities and class labels
            st.write(f"Class probabilities: {probs.tolist()}")
            st.write(f"Class labels: {class_labels}")
            
            label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label].item()

        if label == 1:
            st.success(f"🟢 Likely **TRUE** with confidence {confidence:.2f}")
        else:
            st.error(f"🔴 Likely **FALSE** with confidence {confidence:.2f}")
