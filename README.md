
# XrayAnalyzer: Intelligent X-ray Diagnosis Application


## Project Purpose

**XrayVision** is an intelligent X-ray diagnosis application that leverages the power of vector search through ChromaDB to provide accurate and efficient retrieval of relevant X-ray images based on user input. The application aims to assist medical professionals in diagnosing lung conditions by matching user-provided X-ray images against a database of labeled X-ray images with their diagnosis.

## Key Features

- **Vector Search Integration**: Seamlessly integrates ChromaDB for fast and effective vector search, allowing users to find similar X-ray images based on embedded features.
- **Accurate Results**: Utilizes a pre-trained deep learning model for extracting image features, ensuring robust and reliable search results.
- **User-Friendly Interface**: Designed with a simple and intuitive interface for easy image upload and result retrieval.
- **Detailed Diagnostics**: Provides insights and details about the retrieved X-ray images, including various lung conditions.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Streamlit
- PyTorch
- ChromaDB
- Other dependencies (listed in `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZakisCodes/XrayAnalyzer.git
   cd xray-vision
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Upload your X-ray images**: Use the application interface to upload X-ray images for diagnosis.
   You can find sample Xray image in the app itself.

## Application Flow

1. **User Uploads Image**: The user uploads an X-ray image to the application.
2. **Image Preprocessing**: The uploaded image is preprocessed and converted into an embedding using a pretrained deep learning model.
3. **Vector Search**: ChromaDB is utilized to perform a vector search, retrieving the most similar X-ray images from the database.
4. **Display Results**: The application displays the retrieved images along with detailed diagnostic information from the dataset.


## Video Demo

[![Watch the video](https://img.youtube.com/vi/332PO56vYjg/0.jpg)](https://youtu.be/332PO56vYjg?si=WDWgWgH3kD5JxevE)


## Code Quality and Documentation

The code is organized into several modules, with each function clearly defined and commented. Key functions include:

- `preprocess_image(image_path)`: Loads and preprocesses the uploaded X-ray image.
- `search_image(query, model, image_embeddings)`: Performs a vector search using ChromaDB to find similar images.
- `data_extraction(result_img)`: Extracts diagnostic information based on the retrieved image.

All functions have been documented with clear docstrings explaining their purpose and usage.

## Optimizations

- Enhanced performance by optimizing the image preprocessing pipeline and reducing memory usage during vector queries.

## Acknowledgments

- [ChromaDB](https://www.chromadb.com) for their vector database solution.
- [PyTorch](https://pytorch.org) for providing the deep learning framework.

## Contact

For questions or feedback, feel free to reach out to me at [zakinabdul.tech@gmail.com].

---

Thank you for considering XrayAnalyzer for your evaluation!
