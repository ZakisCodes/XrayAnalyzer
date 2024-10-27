import torch
import chromadb
from PIL import Image
import streamlit as st  
import time
from utilities import *  # Importing utility functions (assumed to be defined elsewhere)
import os
from torchvision import models, transforms
from chromadb.errors import UniqueConstraintError  # Importing error handling for unique constraints

# Initialize a ChromaDB client
client = chromadb.Client()

# Use st.session_state to prevent creating duplicate collections in the database
if "collection" not in st.session_state:
    try:
        # Attempt to create a new collection named "Xray_collection"
        st.session_state.collection = client.create_collection("Xray_collection")
    except UniqueConstraintError:
        # If the collection already exists, retrieve the existing collection
        st.session_state.collection = client.get_collection("Xray_collection")
else:
    # Access the existing collection from session state if it has been created before
    st.session_state.collection = st.session_state.collection

# Load the pre-trained DenseNet-121 model
from torchvision.models import DenseNet121_Weights
model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)  # Load the model with default weights
model.eval()  # Set the model to evaluation mode

# Remove the classifier layer to focus on feature extraction
model = torch.nn.Sequential(*list(model.children())[:-1])  # Extract features only

# Define the preprocessing transformations for input images
preprocess = transforms.Compose([
    transforms.Resize(256),                 # Resize the image so the smaller side is 256 pixels
    transforms.CenterCrop(224),             # Center crop the image to 224x224, the input size for DenseNet
    transforms.ToTensor(),                  # Convert the image to a PyTorch tensor and scale pixel values to [0, 1]
    transforms.Normalize(                   # Normalize the image using the mean and std of ImageNet dataset
        mean=[0.485, 0.456, 0.406],         # ImageNet mean values
        std=[0.229, 0.224, 0.225]           # ImageNet standard deviations
    )
])

# Define the directory where X-ray images are stored
directory = 'data/xray_images'

# Retrieve the file paths of all images in the specified directory
file_paths = [os.path.join(directory, file)
              for file in os.listdir(directory)
              if os.path.isfile(os.path.join(directory, file))]

# Load and preprocess multiple images
images = [Image.open(file_path).convert("RGB") for file_path in file_paths]  # Load images in RGB format
preprocessed_images = [preprocess(img).unsqueeze(0) for img in images]  # Preprocess each image
# Concatenate all preprocessed images into a single tensor for batch processing
batch = torch.cat(preprocessed_images, dim=0)  # Shape: [num_images, 3, 224, 224]

# Measure the time taken to ingest images and create embeddings
start_ingestion_time = time.time()

with torch.no_grad():  # Disable gradient calculation for inference
    image_embeddings = model(batch).numpy()  # Generate image embeddings using the DenseNet model

# Flatten and convert embeddings to a list format for storage
image_embeddings = [embedding.flatten().tolist() for embedding in image_embeddings]

# Measure the total ingestion time
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

# Add the image embeddings and associated metadata to the ChromaDB collection
st.session_state.collection.add(
    embeddings=image_embeddings,  # Add the generated embeddings
    metadatas=[{'images': file_path} for file_path in file_paths],  # Store the corresponding image file paths
    ids=[str(i) for i in range(len(file_paths))]  # Create unique IDs for each embedding
)

# Log the time taken for image data ingestion
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Set up the Streamlit app layout to wide mode
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)  # Create two columns for layout

with col1:
    st.header("X-ray Image Search Engine")  # Header for the search engine
    btn = st.button("Adding Sample Xray Image", on_click=add_sample())
    if btn:
       file_image = add_sample()
    else:
       file_image = st.file_uploader("Upload a Xray Reference Image", type=["jpg", "jpeg", "png"])  # File uploader for reference images
    
# If a file image is uploaded, process the image in the second column
if file_image is not None:
    with col2:
        search_image(file_image, model, image_embeddings)  # Call the function to search using the uploaded image

# Extract data related to the session state result file (assumed function defined elsewhere)
data_extraction(st.session_state.result_file)
