import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import glob
import zipfile
from padelpy import padeldescriptor
from tempfile import TemporaryDirectory
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib

# Define paths
BIOACTIVITY_DATA_PATH = 'C:\\Workplace\\QSAR_Web_App\\bioactivity_data'
SMI_FOLDER_PATH = 'C:\\Workplace\\QSAR_Web_App\\smi_folder'
FINGERPRINTS_PATH = 'C:\\Workplace\\QSAR_Web_App\\fingerprints'
XML_DIR = 'C:\\Workplace\\QSAR_Web_App'
REGRESSION_MODEL_SAVE_DIR = 'C:\\Workplace\\QSAR_Web_App\\regression_models'
CLASSIFICATION_MODEL_SAVE_DIR = 'C:\\Workplace\\QSAR_Web_App\\classification_models'
PREDICTIONS_SAVE_DIR = 'C:\\Workplace\\QSAR_Web_App\\predictions'

def fetch_data(chembl_id):
    conn = sqlite3.connect('C:\\Workplace\\QSAR_Web_App\\chembl_34.db')
    query = """
    SELECT act.standard_value, act.standard_units, cs.canonical_smiles
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    WHERE td.chembl_id = ? AND act.standard_type = 'IC50';
    """
    result = pd.read_sql_query(query, conn, params=(chembl_id,))
    conn.close()
    result['pIC50'] = -np.log10(result['standard_value'] * 1e-9)
    result['bioactivity_class'] = (result['pIC50'] > 6.8).astype(int)
    result.drop(columns=['standard_value', 'standard_units'], inplace=True)
    result.drop_duplicates(subset=['canonical_smiles'], keep='first', inplace=True)
    result.sort_values(by='pIC50', ascending=False, inplace=True)
    return result

def display_data(data, chembl_id):
    st.write("### Bioactivity Data")
    st.write(data.head())
    st.write(f"**Total molecules:** {len(data)}")
    st.write(f"**Active molecules:** {data['bioactivity_class'].sum()}")
    st.write(f"**Inactive molecules:** {len(data) - data['bioactivity_class'].sum()}")
    save_data(data, chembl_id)

def save_data(data, chembl_id):
    csv_file_path = os.path.join(BIOACTIVITY_DATA_PATH, f'{chembl_id}.csv')
    smi_file_path = os.path.join(SMI_FOLDER_PATH, f'{chembl_id}.smi')
    data.to_csv(csv_file_path, index=False)
    data[['canonical_smiles']].to_csv(smi_file_path, index=False, header=False)

def compute_padel_fingerprints(smi_file_path, output_dir):
    xml_files = glob.glob(os.path.join(XML_DIR, "*.xml"))
    xml_files.sort()
    FP_list = [
        "AtomicPairs2DCount",
        "AtomicPairs2D",
        "Estate",
        "CDKextended",
        "CDK",
        "CDKGraphOnly",
        "KlekotaRothCount",
        "KlekotaRoth",
        "MACCS",
        "Pubchem",
        "SubstructureCount",
        "Substructure"
    ]

    fp = dict(zip(FP_list, xml_files))

    for fingerprint in FP_list:
        if fingerprint not in fp:
            st.write(f"Fingerprint descriptor XML not found for {fingerprint}")
            continue

        output_csv = os.path.join(output_dir, f"{fingerprint}.csv")
        fingerprint_descriptortypes = fp[fingerprint]

        try:
            padeldescriptor(mol_dir=smi_file_path,
                            d_file=output_csv,
                            descriptortypes=fingerprint_descriptortypes,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=4,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
            df = pd.read_csv(output_csv)
            if 'Name' in df.columns:
                df.drop(columns=['Name'], inplace=True)
            df.to_csv(output_csv, index=False)
        except Exception as e:
            st.write(f"Error computing fingerprints for {fingerprint}: {e}")

def compute_rdkit_fingerprints(smi_file_path, output_dir):
    df_smiles = pd.read_csv(smi_file_path, header=None, names=['smiles' , 'canonical_smiles' , 'SMILES'])
    mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles['smiles']]

    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    morgan_df = pd.DataFrame([list(fp) for fp in morgan_fps])
    morgan_df.to_csv(os.path.join(output_dir, "RDKit_Morgan.csv"), index=False)

    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    maccs_df = pd.DataFrame([list(fp) for fp in maccs_fps])
    maccs_df.to_csv(os.path.join(output_dir, "RDKit_MACCS.csv"), index=False)

    rdkit_fps = [RDKFingerprint(m) for m in mols]
    rdkit_df = pd.DataFrame([list(fp) for fp in rdkit_fps])
    rdkit_df.to_csv(os.path.join(output_dir, "RDKit_Fingerprint.csv"), index=False)

def remove_name_column(fingerprints_dir):
    for fingerprint_csv in glob.glob(os.path.join(fingerprints_dir, "*.csv")):
        df = pd.read_csv(fingerprint_csv)
        if 'Name' in df.columns:
            df = df.drop(columns=['Name'])
            df.to_csv(fingerprint_csv, index=False)

def merge_bioactivity_with_fingerprints(csv_file_path, fingerprints_dir):
    bioactivity_data = pd.read_csv(csv_file_path)

    for fingerprint_csv in glob.glob(os.path.join(fingerprints_dir, "*.csv")):
        df = pd.read_csv(fingerprint_csv)
        merged_df = pd.concat([df, bioactivity_data[['pIC50', 'bioactivity_class']]], axis=1)
        merged_df.to_csv(fingerprint_csv, index=False)

def handle_new_file(file_path):
    if file_path.endswith('.smi'):
        chembl_id = os.path.splitext(os.path.basename(file_path))[0]
        csv_file_path = os.path.join(BIOACTIVITY_DATA_PATH, f'{chembl_id}.csv')

        if os.path.exists(csv_file_path):
            with TemporaryDirectory() as temp_dir:
                smi_file_path = file_path

                with st.spinner("Computing fingerprints, please wait..."):
                    compute_padel_fingerprints(smi_file_path, temp_dir)
                    compute_rdkit_fingerprints(smi_file_path, temp_dir)
                    remove_name_column(temp_dir)
                    merge_bioactivity_with_fingerprints(csv_file_path, temp_dir)

                    zip_file_name = f"fingerprint_{chembl_id}.zip"
                    zip_path = os.path.join(temp_dir, zip_file_name)
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for root, _, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith('.csv'):
                                    zipf.write(os.path.join(root, file), arcname=file)

                    shutil.move(zip_path, os.path.join(FINGERPRINTS_PATH, zip_file_name))

                    st.session_state['new_file_detected'] = True
                    st.session_state['new_file_path'] = os.path.join(FINGERPRINTS_PATH, zip_file_name)

                st.success("Fingerprints computed and saved successfully.")

def train_best_regression_model(zip_file_path, chembl_id):
    with TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]

        best_regression_model = None
        best_r2 = float('-inf')
        best_regression_model_name = ""

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = df.drop(columns=['bioactivity_class'])
            df = df.dropna(subset=['pIC50'])
            X = df.drop(columns=['pIC50'])
            y = df['pIC50']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_regression_model = model
                fingerprint_name = os.path.splitext(os.path.basename(csv_file))[0]
                best_regression_model_name = f"{chembl_id}_{fingerprint_name}_regressor.pkl"

        if best_regression_model is not None:
            model_save_path = os.path.join(REGRESSION_MODEL_SAVE_DIR, best_regression_model_name)
            joblib.dump(best_regression_model, model_save_path)
            st.write(f"Best regression model trained with R^2 score: {best_r2}")
        else:
            st.write('No regression models were trained.')

def train_best_classification_model(zip_file_path, chembl_id):
    with TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]

        best_classification_model = None
        best_accuracy = 0
        best_classification_model_name = ""

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = df.drop(columns=['pIC50'])
            df = df.dropna(subset=['bioactivity_class'])
            X = df.drop(columns=['bioactivity_class'])
            y = df['bioactivity_class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classification_model = model
                fingerprint_name = os.path.splitext(os.path.basename(csv_file))[0]
                best_classification_model_name = f"{chembl_id}_{fingerprint_name}_classifier.pkl"

        if best_classification_model is not None:
            model_save_path = os.path.join(CLASSIFICATION_MODEL_SAVE_DIR, best_classification_model_name)
            joblib.dump(best_classification_model, model_save_path)
            st.write(f"Best classification model trained with accuracy: {best_accuracy}")
        else:
            st.write('No classification models were trained.')

def read_input_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            return pd.read_csv(uploaded_file, encoding='latin1')
    elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")

def get_smiles_column(df):
    for col in df.columns:
        if col.lower() in ['canonical_smiles', 'smiles', 'smile']:
            return col
    raise ValueError("No SMILES column found in the input file")

def process_smiles_file(uploaded_file):
    smiles_df = read_input_file(uploaded_file)
    smiles_col = get_smiles_column(smiles_df)
    with TemporaryDirectory() as temp_dir:
        smiles_file = os.path.join(temp_dir, "smiles.smi")
        smiles_df[smiles_col].to_csv(smiles_file, index=False, header=False)
        
        compute_padel_fingerprints(smiles_file, temp_dir)
        compute_rdkit_fingerprints(smiles_file, temp_dir)
        
        clf_results = []
        reg_results = []

        for model_path in glob.glob(os.path.join(CLASSIFICATION_MODEL_SAVE_DIR, '*.pkl')):
            clf = joblib.load(model_path)
            chembl_id, fp_name, _ = os.path.basename(model_path).split('_')
            fingerprint_csv = os.path.join(temp_dir, f"{fp_name}.csv")
            df = pd.read_csv(fingerprint_csv)
            df['bioactivity_class'] = clf.predict(df)
            clf_results.append(df[['smiles', 'bioactivity_class']])
        
        clf_combined_df = pd.concat(clf_results, axis=0)
        clf_combined_df = clf_combined_df.loc[:, ~clf_combined_df.columns.duplicated()]
        clf_combined_df = clf_combined_df[clf_combined_df['bioactivity_class'] == 1]
        clf_combined_file = os.path.join(temp_dir, "classification_processed.csv")
        clf_combined_df.to_csv(clf_combined_file, index=False)

        for model_path in glob.glob(os.path.join(REGRESSION_MODEL_SAVE_DIR, '*.pkl')):
            reg = joblib.load(model_path)
            chembl_id, fp_name, _ = os.path.basename(model_path).split('_')
            fingerprint_csv = os.path.join(temp_dir, f"{fp_name}.csv")
            df = pd.read_csv(fingerprint_csv)
            df['predicted_pIC50'] = reg.predict(df)
            reg_results.append(df[['smiles', 'predicted_pIC50']])
        
        reg_combined_df = pd.concat(reg_results, axis=0)
        reg_combined_df = reg_combined_df.loc[:, ~reg_combined_df.columns.duplicated()]

        final_df = pd.merge(smiles_df[[smiles_col]], clf_combined_df[['bioactivity_class']], left_index=True, right_index=True)
        final_df = pd.merge(final_df, reg_combined_df[['predicted_pIC50']], left_index=True, right_index=True)
        
        if not os.path.exists(PREDICTIONS_SAVE_DIR):
            os.makedirs(PREDICTIONS_SAVE_DIR)

        results_file = os.path.join(PREDICTIONS_SAVE_DIR, "predictions.csv")
        final_df.to_csv(results_file, index=False)
        
        return results_file

def main():
    st.set_page_config(page_icon="🧪" , page_title="IC50 analysis and Prediction")
    st.title('QSAR using advanced ML algorithms')

    st.markdown("""
    This application allows users to retrieve and analyze bioactivity data for specific compounds using CHEMBL IDs.
    It is tailored to support researchers and developers in the field of drug discovery and pharmacology.
    Start by entering a CHEMBL ID below to fetch data and begin your ML modeling and predictive analysis.
    """)

    chembl_id = st.text_input("Enter CHEMBL ID:", "").strip()
    
    if st.button("Fetch Data"):
        if chembl_id:
            data = fetch_data(chembl_id)
            if not data.empty:
                st.session_state['data'] = data
                st.session_state['chembl_id'] = chembl_id
                display_data(data, chembl_id)
            else:
                st.error("No data found for the provided CHEMBL ID.")
                clear_state()
        else:
            st.error("Please enter a valid CHEMBL ID.")
    elif 'data' in st.session_state:
        display_data(st.session_state['data'], st.session_state['chembl_id'])

    if st.button("Compute Fingerprints"):
        chembl_id = st.session_state.get('chembl_id')
        if chembl_id:
            csv_file_path = os.path.join(BIOACTIVITY_DATA_PATH, f'{chembl_id}.csv')
            smi_file_path = os.path.join(SMI_FOLDER_PATH, f'{chembl_id}.smi')

            if os.path.exists(csv_file_path) and os.path.exists(smi_file_path):
                with st.spinner("Computing fingerprints, please wait..."):
                    handle_new_file(smi_file_path)
                st.success("Fingerprints computed successfully")
            else:
                st.error("Required files not found. Please fetch data first.")
        else:
            st.error("Please fetch data first.")

    st.markdown("### Train ML Models")
    st.write("Training multiple machine learning models with computed fingerprints involves using the structural descriptors of chemical compounds (fingerprints) as input features to build predictive models. These models can include various algorithms like Random Forest, Support Vector Machines, or Neural Networks, each trained to predict specific properties or activities of the compounds. By evaluating and comparing the performance of these models, the best-performing one can be selected for further analysis or deployment.")

    if st.button("Train Models"):
        if st.session_state.get('new_file_detected'):
            zip_file_path = st.session_state.get('new_file_path')
            chembl_id = st.session_state.get('chembl_id')
            with st.spinner("Training models, please wait..."):
                train_best_regression_model(zip_file_path, chembl_id)
                train_best_classification_model(zip_file_path, chembl_id)
            st.success("Models trained and saved successfully!")
            st.session_state.new_file_detected = False
            st.session_state.new_file_path = ''
        else:
            st.write("No new files detected to train models.")

    st.markdown("### Predict using pre-trained models")
    uploaded_file = st.file_uploader("Upload your SMILES file (CSV or Excel)", type=["csv", "xls", "xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Processing your file..."):
            results_file = process_smiles_file(uploaded_file)
        st.success("Processing complete.")
        st.download_button("Download Predictions", data=open(results_file, 'rb'), file_name="predictions.csv", mime='text/csv')

def clear_state():
    for key in ['data', 'chembl_id', 'new_file_detected', 'new_file_path']:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main()