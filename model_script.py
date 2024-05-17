import json
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config

# Load TensorFlow model
def load_tf_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Predict sentiment using TensorFlow model
def predict_tf_model(model, image_path):
    img = Image.open(image_path).resize((150, 150)).convert('RGB')  # Ensure the image is RGB
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)
    sentiment_class = np.argmax(predictions, axis=1)[0]
    sentiment_labels = ['Happy', 'Sad', 'Neutral']  # Adjust according to your classes
    sentiment = sentiment_labels[sentiment_class]
    return sentiment

# Load GPT-2 model
def load_torch_model(model_path, config_path):
    with open(config_path, 'r') as config_file:
        config = GPT2Config.from_dict(json.load(config_file))
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

# Generate music sequence using GPT-2 model
def generate_music(model, sentiment, sentiment_dict, start_token=60, max_length=50, temperature=1.0):
    sentiment_tensor = torch.tensor([sentiment_dict[sentiment]], dtype=torch.long)
    generated = [start_token]

    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
            outputs = model(input_ids=input_ids, labels=None)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_logits = next_token_logits / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
            generated.append(next_token)

    return generated