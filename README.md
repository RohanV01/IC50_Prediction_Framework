# QSAR Web App

This project is a comprehensive web application for performing Quantitative Structure-Activity Relationship (QSAR) analysis using Streamlit. The application allows users to fetch bioactivity data from a ChEMBL database, compute molecular fingerprints, and train machine learning models for regression and classification tasks. It supports SMILES input files and outputs predictions based on trained models.

Importance and Advantages of the Solution
Importance:

Efficient Drug Discovery: QSAR analysis is crucial in predicting the bioactivity of chemical compounds, significantly reducing the time and cost associated with drug discovery.
Data-Driven Decisions: By leveraging bioactivity data and molecular fingerprints, researchers can make more informed decisions about which compounds to prioritize for further testing.
Advantages:

Streamlined Workflow: The web app integrates data fetching, fingerprint computation, and model training into a single platform, simplifying the workflow for users.
User-Friendly Interface: With Streamlit, the application provides an intuitive interface for users to interact with, making complex processes accessible to those with limited programming knowledge.
Scalable and Customizable: The modular design allows users to easily update the models and datasets, ensuring the tool remains relevant and effective as new data becomes available.
Reproducibility: By automating the QSAR process, the web app ensures that results are reproducible and consistent, which is critical for scientific research.
Accessibility: Researchers and developers can access and use the app from any location, facilitating collaboration and data sharing.

## Features

- Fetch bioactivity data from ChEMBL using ChEMBL IDs.
- Compute molecular fingerprints using RDKit and PaDEL-Descriptor.
- Train and save the best regression and classification models.
- Process SMILES files to predict bioactivity classes and pIC50 values.
- Provide downloadable results as CSV files.

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/QSAR_Web_App.git
    cd QSAR_Web_App
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have the ChEMBL database (`chembl_34.db`) in the specified directory.

4. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Usage

### Fetching Bioactivity Data

1. Enter the ChEMBL ID of the target.
2. Click "Fetch Data" to retrieve bioactivity data for the specified target.
3. The data will be displayed in the app and saved as CSV and SMI files.

### Computing Molecular Fingerprints

1. Ensure the SMI file is saved in the specified directory.
2. Click "Compute Fingerprints" to generate PaDEL and RDKit fingerprints.
3. The computed fingerprints will be saved as CSV files.

### Training Models

1. Train the best regression model:
    - Click "Train Regression Model".
    - The best model will be trained and saved based on R² score.
2. Train the best classification model:
    - Click "Train Classification Model".
    - The best model will be trained and saved based on accuracy.

### Processing New SMILES Files

1. Upload a SMILES file in CSV or SMI format.
2. Click "Process SMILES File".
3. The app will compute fingerprints, apply trained models, and output predictions.

## Directory Structure

- `bioactivity_data/`: Directory for storing fetched bioactivity data CSV files.
- `smi_folder/`: Directory for storing SMILES files.
- `fingerprints/`: Directory for storing computed fingerprint CSV files.
- `regression_models/`: Directory for storing trained regression models.
- `classification_models/`: Directory for storing trained classification models.
- `predictions/`: Directory for storing prediction results.

## Dependencies

- pandas
- numpy
- sqlite3
- glob
- zipfile
- padelpy
- tempfile
- shutil
- rdkit
- sklearn
- joblib
- streamlit

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

---

Feel free to reach out for any questions or support. Happy QSAR analysis!

---

**Note:** Ensure all paths and directories are correctly set up in the code before running the application.