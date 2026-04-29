"""Convert SMILES to PyTorch Geometric molecular graph Data objects.

Each molecule is represented as a graph where:
    - Nodes = atoms (with features: element, degree, charge, etc.)
    - Edges = bonds (with features: bond type, stereo, conjugation, etc.)

Designed for on-the-fly conversion in Dataset.__getitem__ (~0.1ms per molecule).

Usage:
    from target_affinity_ml.features.molecular_graphs import smiles_to_graph
    data = smiles_to_graph("CCO")  # ethanol → Data(x, edge_index, edge_attr)
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Atom feature dimensions
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # Periodic table
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# Most common elements in drug-like molecules (one-hot, rest → "other")
COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
ATOM_DIM = len(COMMON_ATOMS) + 1 + 7 + 5 + 5 + 5 + 3  # 35 total

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, choices: list) -> list[float]:
    """One-hot encode a value from a list of choices."""
    encoding = [0.0] * (len(choices) + 1)  # +1 for "other"
    try:
        idx = choices.index(value)
        encoding[idx] = 1.0
    except ValueError:
        encoding[-1] = 1.0  # "other" category
    return encoding


def atom_features(atom: Chem.Atom) -> list[float]:
    """Compute feature vector for a single atom.

    Features (35 dimensions total):
        - Atomic number (one-hot, 9 common + other = 10)
        - Degree (one-hot, 0-6 + other = 8)
        - Formal charge (one-hot, -2 to +2 + other = 6)
        - Num Hs (one-hot, 0-4 + other = 6)
        - Hybridization (one-hot, 5 types + other = 6)
        - Is aromatic (1)
        - Is in ring (1)
        - Chirality tag (one-hot, 3 values = 3 for compact encoding)
    """
    features = []
    features.extend(_one_hot(atom.GetAtomicNum(), COMMON_ATOMS))
    features.extend(_one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"]))
    features.extend(_one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]))
    features.extend(_one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]))
    features.extend(_one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"]))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    # Chirality (compact: none/CW/CCW)
    chiral = atom.GetChiralTag()
    features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0)
    features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0)
    features.append(1.0 if chiral != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0)
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    """Compute feature vector for a single bond.

    Features (11 dimensions):
        - Bond type (one-hot, 4 + other = 5)
        - Stereo (one-hot, 4 + other = 5)
        - Is conjugated (1)
    """
    features = []
    features.extend(_one_hot(bond.GetBondType(), BOND_TYPES))
    features.extend(_one_hot(bond.GetStereo(), BOND_STEREO))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    return features


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    Data or None
        PyG Data with x (node features), edge_index, edge_attr.
        Returns None if SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))

    x = torch.tensor(atom_feats, dtype=torch.float32)

    # Bond features (bidirectional edges)
    if mol.GetNumBonds() == 0:
        # Isolated atom (rare but possible)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 11), dtype=torch.float32)
    else:
        edges_src, edges_dst = [], []
        edge_feats = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_features(bond)

            # Add both directions
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edge_feats.extend([bf, bf])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_atom_feature_dim() -> int:
    """Return the dimensionality of atom feature vectors."""
    # Compute from a dummy atom
    mol = Chem.MolFromSmiles("C")
    return len(atom_features(mol.GetAtomWithIdx(0)))


def get_bond_feature_dim() -> int:
    """Return the dimensionality of bond feature vectors."""
    mol = Chem.MolFromSmiles("CC")
    return len(bond_features(mol.GetBondWithIdx(0)))
