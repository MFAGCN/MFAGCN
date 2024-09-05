# Input File Format

The model requires a CSV file with the following columns:

* SMILES: String representation of the chemical structure in SMILES format.
* Activity: Binary label indicating the activity of the compound (1 for active, 0 for inactive).


# Installation

## Required Libraries

To run the code, you need to install the following Python packages:
`pip install pandas torch torch_geometric scikit-learn rdkit networkx numpy imbalanced-learn`

Additional Dependencies
RDKit: For generating molecular fingerprints and handling chemical structures.
PyTorch: Deep learning framework for building and training neural networks.
PyTorch Geometric: Extension library for deep learning on graph-structured data.

# Feature Engineering

The model utilizes the following molecular features:

* MACCS Keys: A type of structural key fingerprint used for chemical informatics.
* ECFP (Extended Connectivity Fingerprint): A circular fingerprint used for molecular similarity.
* PubChem Fingerprint: A binary vector representing the presence of various substructures in a molecule.
