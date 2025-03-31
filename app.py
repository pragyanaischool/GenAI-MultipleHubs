import streamlit as st
from transformers import pipeline
import torch
import torchvision.models as models
import tensorflow_hub as hub

def load_hf_model(task):
    return pipeline(task)

def load_torch_model(model_name):
    model = getattr(models, model_name)(pretrained=True)
    model.eval()
    return model

def load_tf_model(model_url):
    return hub.load(model_url)

st.image("PragyanAI_Transperent.png")
st.title("Pretrained Model Hub Interface")

hub_choice = st.selectbox("Select Model Hub", ["Hugging Face", "PyTorch", "TensorFlow Hub"])

if hub_choice == "Hugging Face":
    task = st.selectbox("Select Task", ["text-classification", "sentiment-analysis", "question-answering", "summarization", "translation"])
    user_input = st.text_area("Enter Input Text")
    if st.button("Run Model"):
        model = load_hf_model(task)
        output = model(user_input)
        st.write("### Model Output:")
        st.json(output)

elif hub_choice == "PyTorch":
    model_name = st.selectbox("Select Model", ["resnet18", "resnet50", "mobilenet_v2", "densenet121"])
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file and st.button("Run Model"):
        from PIL import Image
        import torchvision.transforms as transforms
        
        model = load_torch_model(model_name)
        image = Image.open(uploaded_file)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        st.write("### Model Output:")
        st.write(output)

elif hub_choice == "TensorFlow Hub":
    model_url = st.text_input("Enter Model URL")
    user_input = st.text_area("Enter Input Data (comma-separated)")
    if st.button("Run Model"):
        model = load_tf_model(model_url)
        input_data = [float(x) for x in user_input.split(",")]
        output = model(input_data)
        st.write("### Model Output:")
        st.write(output)
