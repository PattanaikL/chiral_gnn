from argparse import Namespace
from typing import List, Tuple, Union
import gzip
import pickle

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

import torch
import numpy as np
import pandas as pd
from itertools import chain

import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

# Atom feature sizes
ATOMIC_SYMBOLS = ['H', 'C', 'B', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
CIP_CHIRALITY = ['R', 'S']
ATOM_FEATURES = {
    'atomic_num': ATOMIC_SYMBOLS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'global_chiral_tag': CIP_CHIRALITY,
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
    return 14


def onek_encoding_unk(value, choices: List) -> List[int]:
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


def atom_features(atom: Chem.rdchem.Atom, args) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if args.chiral_features:
        features += onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    if args.global_chiral_features:
        if atom.HasProp('_CIPCode'):
            features += onek_encoding_unk(atom.GetProp('_CIPCode'), ATOM_FEATURES['global_chiral_tag'])
        else:
            features += onek_encoding_unk(None, ATOM_FEATURES['global_chiral_tag'])
    return features

def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown

    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]


def bond_features(bond: Chem.rdchem.Bond, args) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    bond_fdim = get_bond_fdim(args)

    if bond is None:
        fbond = [1] + [0] * (bond_fdim - 1)
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
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
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

    def __init__(self, mol_block: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.parity_atoms = []  # mapping from atom index to CW (+1), CCW (-1) or undefined tetra (0)
        self.edge_index = []  # list of tuples indicating presence of bonds

        # Convert mol block to molecule with hydrogens
        mol = Chem.MolFromMolBlock(mol_block)
        mol = Chem.AddHs(mol)
        self.smiles = Chem.MolToSmiles(mol)

        # remove stereochem label from atoms with less/more than 4 neighbors
        H_ids = [a.GetIdx() for a in mol.GetAtoms() if CHIRALTAG_PARITY[a.GetChiralTag()] != 0]
        for i in H_ids:
            a = mol.GetAtomWithIdx(i)
            if len(a.GetNeighbors()) != 4:
                a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()
        
        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom, args))
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

                f_bond = bond_features(bond, args)

                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)

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

    def __init__(self, mol_blocks, c_shifts, h_shifts, args, mode='train'):
        super(MolDataset, self).__init__()

        if args.split_path:
            self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
            self.split = np.load(args.split_path, allow_pickle=True)[self.split_idx]
        else:
            self.split = list(range(len(mol_blocks)))  # fix this
        self.mol_blocks = [mol_blocks[i] for i in self.split]
        self.c_shifts = [c_shifts[i] for i in self.split]
        self.h_shifts = [h_shifts[i] for i in self.split]
        self.data_map = {k: v for k, v in zip(range(len(self.mol_blocks)), self.split)}
        self.args = args

        if mode == 'train':
            shifts = self.c_shifts if args.shift == 'C' else self.h_shifts
            shifts_real = [s for arr in shifts for s in arr if s != -1000]
            self.mean = np.mean(shifts_real)
            self.std = np.std(shifts_real)

    def process_key(self, key):
        mol_block = self.mol_blocks[key]
        molgraph = MolGraph(mol_block, self.args)
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.parity_atoms = torch.tensor(molgraph.parity_atoms, dtype=torch.long)
        data.smiles = molgraph.smiles
        data.c_shifts = torch.tensor(self.c_shifts[key], dtype=torch.float)
        data.c_mask = data.c_shifts != -1000
        data.h_shifts = torch.tensor(self.h_shifts[key], dtype=torch.float)
        data.h_mask = data.h_shifts != -1000

        return data

    def __len__(self):
        return len(self.mol_blocks)

    def __getitem__(self, key):
        return self.process_key(key)


class StereoSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        groups = [[i, i + 1] for i in range(0, len(self.data_source), 2)]
        np.random.shuffle(groups)
        indices = list(chain(*groups))
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]

    with gzip.open(args.data_path, 'rb') as f:
        data_df = pickle.load(f)

    data_df = data_df.dropna(subset=['spectrum_C']) if args.shift == 'C' else data_df.dropna(subset=['spectrum_H'])

    mol_blocks = data_df.iloc[:, 6].values
    c_shifts = data_df.iloc[:, 1].values
    h_shifts = data_df.iloc[:, 2].values

    loaders = []
    for mode in modes:
        dataset = MolDataset(mol_blocks, c_shifts, h_shifts, args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=not args.no_shuffle if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            sampler=StereoSampler(dataset) if args.shuffle_pairs else None)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
