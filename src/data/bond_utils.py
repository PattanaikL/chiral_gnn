import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir, BondStereo

BondDirOpposite = {BondDir.ENDUPRIGHT: BondDir.ENDDOWNRIGHT,
                   BondDir.ENDDOWNRIGHT: BondDir.ENDUPRIGHT,
                   BondDir.NONE: BondDir.NONE}
PLEVEL = 0

def get_atoms_across_double_bonds(mol, labeling_func=lambda a:a.GetIdx()):
    '''
    NOTE: This function was taken directly from rdchiral without modificationn


    This function takes a molecule and returns a list of cis/trans specifications
    according to the following:
    (idxs, dirs)
    where atoms = (a1, a2, a3, a4) and dirs = (d1, d2)
    and (a1, a2) defines the ENDUPRIGHT/ENDDOWNRIGHT direction of the "front"
    of the bond using d1, and (a3, a4) defines the direction of the "back" of 
    the bond using d2.
    This is used to initialize reactants with a SINGLE definition constraining
    the chirality. Templates have their chirality fully enumerated, so we can
    match this specific definition to the full set of possible definitions
    when determining if a match should be made.
    NOTE: the atom idxs are returned. This is so we can later use them
    to get the old_mapno property from the corresponding product atom, which is
    an outcome-specific assignment
    We also include implicit chirality here based on ring membership, but keep
    track of that separately
    '''

    atoms_across_double_bonds = []
    atomrings = None

    for b in mol.GetBonds():
        if b.GetBondType() != BondType.DOUBLE:
            continue

        # Define begin and end atoms of the double bond
        ba = b.GetBeginAtom()
        bb = b.GetEndAtom()

        # Now check if it is even possible to specify
        if ba.GetDegree() == 1 or bb.GetDegree() == 1:
            continue

        ba_label = labeling_func(ba)
        bb_label = labeling_func(bb)
            
        if PLEVEL >= 5: print('Found a double bond with potential cis/trans (based on degree)')
        if PLEVEL >= 5: print('{} {} {}'.format(ba_label,
                               b.GetSmarts(),
                               bb_label))
        
        # Try to specify front and back direction separately
        front_idxs = None 
        front_dir = None 
        back_idxs = None 
        back_dir = None
        is_implicit = False 
        bab = None; bbb = None;
        for bab in (z for z in ba.GetBonds() if z.GetBondType() != BondType.DOUBLE):
            if bab.GetBondDir() != BondDir.NONE:
                front_idxs = (labeling_func(bab.GetBeginAtom()), labeling_func(bab.GetEndAtom()))
                front_dir = bab.GetBondDir()
                break 
        for bbb in (z for z in bb.GetBonds() if z.GetBondType() != BondType.DOUBLE):
            if bbb.GetBondDir() != BondDir.NONE:
                back_idxs = (labeling_func(bbb.GetBeginAtom()), labeling_func(bbb.GetEndAtom()))
                back_dir = bbb.GetBondDir()
                break 

        # If impossible to spec, just continue
        if (bab is None or bbb is None):
            continue

        # Did we actually get a specification out?
        if (front_dir is None or back_dir is None):

            if b.IsInRing(): 
                # Implicit cis! Now to figure out right definitions...
                if atomrings is None:
                    atomrings = mol.GetRingInfo().AtomRings() # tuple of tuples of atomIdx
                for atomring in atomrings:
                    if ba.GetIdx() in atomring and bb.GetIdx() in atomring:
                        front_idxs = (labeling_func(bab.GetOtherAtom(ba)), ba_label)
                        back_idxs = (bb_label, labeling_func(bbb.GetOtherAtom(bb)))
                        if (bab.GetOtherAtomIdx(ba.GetIdx()) in atomring) != \
                                (bbb.GetOtherAtomIdx(bb.GetIdx()) in atomring):
                            # one of these atoms are in the ring, one is outside -> trans
                            if PLEVEL >= 10: print('Implicit trans found')
                            front_dir = BondDir.ENDUPRIGHT
                            back_dir = BondDir.ENDUPRIGHT
                        else:
                            if PLEVEL >= 10: print('Implicit cis found')
                            front_dir = BondDir.ENDUPRIGHT 
                            back_dir = BondDir.ENDDOWNRIGHT
                        is_implicit = True 
                        break

            else:
                # Okay no, actually unspecified
                continue

        # We want to record all bonds as if they can be read from left-to-right. So make sure
        # that the two double-bonded carbons are "in the middle" of the 4-tuple
        if front_idxs[0] == ba_label: # need to flip
            font_idxs = front_idxs[::-1]
            front_dir = BondDirOpposite[front_dir]
        if back_idxs[1] == bb_label:
            back_idxs = back_idxs[::-1]
            back_dir = BondDirOpposite[back_dir]

        # Save this (a1, a2, a3, a4) -> (d1, d2) spec
        atoms_across_double_bonds.append(
            (
                front_idxs + back_idxs,
                (front_dir, back_dir),
                is_implicit,
            )
        )

    return atoms_across_double_bonds



def get_cis_trans_atom_pairs(mol, include_implicit=False):
    cistrans_pairs = {}
    for a_idxs, dirs, is_implicit in get_atoms_across_double_bonds(mol):
        
        if is_implicit and not include_implicit:
            continue

        if BondDir.NONE in dirs:
            continue

        # Rename from idx for good bookkeeping
        front_idx = a_idxs[0]
        back_idx = a_idxs[3]



        # print(Chem.MolToSmiles(mol))
        # print(a_idxs)
        # print(dirs)
        # print(is_implicit)


        if dirs == (BondDir.ENDDOWNRIGHT, BondDir.ENDDOWNRIGHT) or dirs == (BondDir.ENDUPRIGHT, BondDir.ENDUPRIGHT):
            tag = 'trans'
            tag_oppo = 'cis'
        elif dirs == (BondDir.ENDDOWNRIGHT, BondDir.ENDUPRIGHT) or dirs == (BondDir.ENDUPRIGHT, BondDir.ENDDOWNRIGHT):
            tag = 'cis'
            tag_oppo = 'trans'
        else:
            continue # unspecified
        # print(tag)

        # Record
        cistrans_pairs[tuple(sorted([front_idx, back_idx]))] = tag
        


        # Figure out what the "other" possible left/right atoms are, if any
        other_front_idx = [nb.GetIdx() for nb in mol.GetAtomWithIdx(a_idxs[1]).GetNeighbors() if nb.GetIdx() not in a_idxs]
        other_back_idx = [nb.GetIdx() for nb in mol.GetAtomWithIdx(a_idxs[2]).GetNeighbors() if nb.GetIdx() not in a_idxs]

        if other_front_idx:
            cistrans_pairs[tuple(sorted([other_front_idx[0], back_idx]))] = tag_oppo
        if other_back_idx:
            cistrans_pairs[tuple(sorted([front_idx, other_back_idx[0]]))] = tag_oppo
        if other_front_idx and other_back_idx:
            cistrans_pairs[tuple(sorted([other_front_idx[0], other_back_idx[0]]))] = tag

        
        
        

    return cistrans_pairs

        
        