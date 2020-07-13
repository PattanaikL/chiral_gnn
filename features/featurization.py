import os
from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from .bond_utils import get_cis_trans_atom_pairs
from .utils import read_sdf

import torch
import numpy as np
import pandas as pd

import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3], 
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
CHIRALTAG_PARITY = {
    ChiralType.CHI_TETRAHEDRAL_CW: +1,
    ChiralType.CHI_TETRAHEDRAL_CCW: -1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0, # default
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14 + 2 # was 14, removed GetStereo()  added 2 for cis/trans


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features +=  onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown

    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6))) # remove global cis/trans tags
        fbond += [0, 0] # special cis/trans message edge type
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.parity_atoms = []  # mapping from atom index to CW (+1), CCW (-1) or undefined tetra (0)
        self.edge_index = []  # list of tuples indicating presence of bonds

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        
        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
            self.parity_atoms.append(parity_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue
                    
                self.edge_index.extend([(a1, a2), (a2, a1)])

                f_bond = bond_features(bond)

                if args.atom_messages:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.parity_atoms

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


class MolDataset(Dataset):

    def __init__(self, smiles, labels, args, mode='train'):
        super(MolDataset, self).__init__()

        if args.split_path:
            self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
            self.split = np.load(args.split_path, allow_pickle=True)[self.split_idx]
        else:
            self.split = list(range(len(smiles)))  # fix this
        self.smiles = [smiles[i] for i in self.split]
        self.labels = [labels[i] for i in self.split]
        self.data_map = {k: v for k, v in zip(range(len(self.smiles)), self.split)}
        self.args = args

        if mode == 'train':
            self.mean = np.mean(self.labels)
            self.std = np.std(self.labels)

        if args.confs_dir:
            self.confs_dir = args.confs_dir
            self.zfill = len(os.listdir(self.confs_dir)[0].split('.')[0])

    def process_key(self, key):
        smi = self.smiles[key]
        molgraph = MolGraph(smi, self.args)
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)

        if self.args.confs_dir:
            path = os.path.join(self.confs_dir, f'{self.data_map[key]}'.zfill(self.zfill) + '.sdf')
            pos, sym = read_sdf(path, n=1)[0]
            assert [a.GetSymbol() for a in Chem.MolFromSmiles(self.smiles[key]).GetAtoms()] == sym  # ordering
            pos = torch.tensor(pos)
            data.pos = pos - pos.mean(dim=0)
            data.atoms = torch.tensor([a.GetAtomicNum() for a in Chem.MolFromSmiles(self.smiles[key]).GetAtoms()])

        return data

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, key):
        return self.process_key(key)


def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]

    data_df = pd.read_csv(args.data_path)

    smiles = data_df.iloc[:, 0].values
    labels = data_df.iloc[:, 1].values.astype(np.float32)

    loaders = []
    for mode in modes:
        dataset = MolDataset(smiles, labels, args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
