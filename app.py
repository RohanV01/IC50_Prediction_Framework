from flask import Flask, render_template, request, Response
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client
from rdkit import RDLogger

# Disable RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

app = Flask(__name__)

# Initialize ChEMBL API clients
targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-data', methods=['POST'])
def get_data():
    uniprot_id = request.form['uniprot_id']
    chembl_id = request.form['chembl_id']
    try:
        data = retrieve_data(uniprot_id, chembl_id)
        # Filter the DataFrame to only include 'smiles' and 'pIC50' columns
        filtered_data = data[['smiles', 'pIC50']]
        return Response(
            filtered_data.to_csv(index=False),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={chembl_id}.csv"}
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

def retrieve_data(uniprot_id, chembl_id):
    targets = targets_api.get(target_compounds__accession=uniprot_id).only(
        "target_chembl_id", "organism", "pref_name", "target_type"
    )
    targets_df = pd.DataFrame.from_records(targets)

    if targets_df.empty:
        raise ValueError("No matching target found for the provided CHEMBL ID.")

    target = targets_df[targets_df['target_chembl_id'] == chembl_id].iloc[0]
    bioactivities = bioactivities_api.filter(
        target_chembl_id=target['target_chembl_id'],
        type="IC50",
        relation="=",
        assay_type="B"
    ).only(
        "activity_id", "assay_chembl_id", "assay_description", "assay_type",
        "molecule_chembl_id", "type", "standard_units", "relation", "standard_value",
        "target_chembl_id", "target_organism"
    )
    bioactivities_df = pd.DataFrame.from_dict(bioactivities)
    
    if bioactivities_df.empty:
        raise ValueError("No bioactivities found for the selected target.")

    bioactivities_df.rename(columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True)
    compounds = compounds_api.filter(
        molecule_chembl_id__in=bioactivities_df["molecule_chembl_id"].tolist()
    ).only("molecule_chembl_id", "molecule_structures")

    compounds_df = pd.DataFrame(compounds)
    bioactivities_df = bioactivities_df.merge(compounds_df, on="molecule_chembl_id")
    bioactivities_df['smiles'] = bioactivities_df['molecule_structures'].apply(
        lambda x: x.get('canonical_smiles', '') if isinstance(x, dict) else ''
    )

    bioactivities_df['pIC50'] = bioactivities_df['IC50'].apply(
        lambda x: 9 - np.log10(float(x)) if pd.notnull(x) and float(x) > 0 else np.nan
    )
    PandasTools.AddMoleculeColumnToFrame(bioactivities_df, smilesCol='smiles')
    bioactivities_df.sort_values(by='pIC50', ascending=False, inplace=True)

    return bioactivities_df

if __name__ == '__main__':
    app.run(debug=True)
