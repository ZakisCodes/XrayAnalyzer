import os
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import pandas as pd
from io import BytesIO

# Function to load and preprocess an image
def preprocess_image(image_path):
    """
    Loads and preprocesses a single image from a given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: A 4D tensor of shape [1, 3, 224, 224] representing the preprocessed image.
    """
    img = Image.open(image_path).convert("RGB")  # Open image and ensure it's in RGB format
    img_t = preprocess(img)                       # Apply preprocessing transformations
    img_t = img_t.unsqueeze(0)                   # Add batch dimension: [1, 3, 224, 224]
    return img_t

# Define the preprocessing transformations for the images
preprocess = transforms.Compose([
    transforms.Resize(256),                 # Resize the image to have the smaller side as 256 pixels
    transforms.CenterCrop(224),             # Crop the image to 224x224, the input size for most models
    transforms.ToTensor(),                  # Convert the image to a PyTorch tensor and scale pixel values to [0, 1]
    transforms.Normalize(                   # Normalize using the mean and std of the ImageNet dataset
        mean=[0.485, 0.456, 0.406],         # ImageNet means
        std=[0.229, 0.224, 0.225]           # ImageNet standard deviations
    )
])

def search_image(query, model, image_embeddings):
    """
    Searches for an image similar to the uploaded query image using the model and embeddings.

    Args:
        query (str): Path to the uploaded query image.
        model (torch.nn.Module): The pre-trained model for generating image embeddings.
        image_embeddings (list): List of pre-computed image embeddings.

    Returns:
        None: Displays the result image if found, otherwise shows an error message.
    """
    # Simple validation to check if the query image is provided
    if not query:
        st.error("Please upload a X-ray reference image")
        return

    # Convert the uploaded user image to an embedding
    preprocessed_image = preprocess_image(query)

    # Use the preprocessed image with the model to get its features
    with torch.no_grad():
        image_features = model(preprocessed_image).flatten().numpy()

    # Query the image collection for similar images
    results = st.session_state.collection.query(
        query_embeddings=image_features,
        n_results=1  # Get the top 1 result
    )
    
    # Extract the matched image path from results
    image_path = results["metadatas"][0][0]["images"]
    
    # Store the result filename in session state
    if "result_file" not in st.session_state:
        st.session_state.result_file = extract_filename(image_path)
    
    st.write(st.session_state.result_file)

    # Retrieve and display the matched image
    result_image_path = results['metadatas'][0][0].get('images', None)
    if result_image_path:
        result_img = st.image(result_image_path, caption='Result Image')
        return result_img
    else:
        st.error("No image path found in query results.")

def extract_filename(file_path):
    """
    Extracts the filename from a given file path.

    Args:
        file_path (str): The complete path to the file.

    Returns:
        str: The extracted filename.
    """
    return os.path.basename(file_path)

def calculate_accuracy(image_embedding, query_embedding):
    """
    Calculates the cosine similarity between two image embeddings.

    Args:
        image_embedding (np.array): The embedding of the stored image.
        query_embedding (np.array): The embedding of the query image.

    Returns:
        float: Cosine similarity score between the two embeddings.
    """
    # Calculate the cosine similarity between the embeddings
    similarity = cosine_similarity([image_embedding], [query_embedding])[0]
    return similarity

def data_extraction(result_img):
    """
    Extracts diagnosis details from a CSV based on the result image.

    Args:
        result_img (str): The filename of the result image.

    Returns:
        None: Displays the diagnosis details or an error if not found.
    """
    xray_diagnosis_data = pd.read_csv('xray_diagnosis_data.csv')  # Load diagnosis data

    # Filter the DataFrame for the specific image name
    filtered_data = xray_diagnosis_data[xray_diagnosis_data['file_name'] == result_img]

    # Check if there is a matching row
    if not filtered_data.empty:
        # Extract details from the filtered row
        row = filtered_data.iloc[0]  # Get the first (and only) row

        # Display the image name and findings
        st.header("Details for X-ray Image: " + row['file_name'])

        # Display findings
        st.subheader("Findings")
        st.markdown("---")
        st.subheader("Lung Opacity")
        st.write(row['Lung Opacity'])  # Display findings

        st.markdown("---")
        st.subheader("Pleural Effusion")
        st.write(row['Pleural Effusion'])  # Display findings

        st.markdown("---")
        st.subheader("Pneumothorax")
        st.write(row['Pneumothorax'])  # Display findings

        st.markdown("---")
        st.subheader("Cardiomegaly")
        st.write(row['Cardiomegaly'])  # Display findings

        st.markdown("---")
        st.subheader("Atelectasis")
        st.write(row['Atelectasis'])  # Display findings

        st.markdown("---")
        st.subheader("Consolidation")
        st.write(row['Consolidation'])  # Display findings

        st.markdown("---")
        st.subheader("Diagnosis")
        st.write(row['Diagnosis Summary'])  # Display the overall diagnosis

    else:
        st.error("No data found for the specified image.")




def add_sample():
    file_path = 'sample/00006605_038.png'
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
    return BytesIO(file_bytes)  # Return a BytesIO object for Streamlit to read