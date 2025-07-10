import torch
from Bio.PDB import PDBParser
import pandas as pd
import json


class StructureToCoords: #_scale:
    """
    Convert PDB to atoms at the interface with coordinates and type features
    """
    def __init__(self, interface_cutoff=8):
        self.Atom_df = None
        self.Atom_coords = None
        self.AA_df = None
        self.interface_cutoff = interface_cutoff
        self.AA_list = ['ARG','MET','VAL','ASN','PRO','THR','PHE','ASP','ILE','ALA','GLY','GLU', 'LEU','SER','LYS','TYR','CYS','HIS','GLN','TRP']

    def process_pdb(self, pdbfile:str, aa_feature_path = 'AA_At_dict.json'):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdbfile)[0]

        with open(aa_feature_path, 'r') as file:
            AA_dict = json.load(file)
        AA_dict_list = list(AA_dict.keys())
        AA_df = {'res_name': [], 'chain_id': [], 'atom_names': [],\
            'atom_id_range': []}
        atom_coords = []
        
        idx, chain_count = 0, 0
        # === Build AA_df and collect atomic coordinates ===
        for chain in structure:
            chain.atom_to_internal_coordinates()
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':
                    resname = residue.get_resname()
                    if resname == 'HIE':
                        resname = 'HIS'  # remap HIE to HIS
                    if resname in AA_dict_list:
                        heavy_atoms = [atom.get_name() for atom in residue.get_atoms() if atom.get_name()[0] != 'H']
                        AA_df['res_name'].append(resname)
                        AA_df['chain_id'].append(chain_count)
                        AA_df['atom_id_range'].append((idx, idx + len(heavy_atoms)))
                        AA_df['atom_names'].append(heavy_atoms)
                        
                        atom_coords.extend([atom.get_coord().tolist() for atom in residue.get_atoms() if atom.get_name()[0] != 'H'])
                        idx += len(heavy_atoms)
            chain_count += 1

        atom_coords = torch.tensor(atom_coords, dtype=torch.float32)
        # === Compute pairwise atomic distances using efficient broadcasting ===
        atom_distance = torch.cdist(atom_coords, atom_coords)
        # check intface residues
        AA_df['if_interface'] = [False]*len(AA_df['res_name'])
        sort_list = []
        for idx in range(len(AA_df['res_name'])):
            for jdx in range(idx + 1, len(AA_df['res_name'])):
                residue_pair_dist_mtx = atom_distance[AA_df['atom_id_range'][idx][0]:AA_df['atom_id_range'][idx][1],\
                                        AA_df['atom_id_range'][jdx][0]:AA_df['atom_id_range'][jdx][1]]
                if (residue_pair_dist_mtx < self.interface_cutoff).any():
                    if AA_df['chain_id'][idx] != AA_df['chain_id'][jdx]:
                        AA_df['if_interface'][idx] = True
                        AA_df['if_interface'][jdx] = True
        # === Extract Interface Atom coords and other info ===
        At_df_interface = {'res_id': [], 'res_name':[],'atom_name':[],\
                            'atom_id': [], 'chain_id': [], 'atom_coords': []}
        for ind, at_interface in enumerate(AA_df['if_interface']):
            if at_interface:
                At_df_interface['res_id'].extend([ind]*len(AA_df['atom_names'][ind]))
                At_df_interface['res_name'].extend([AA_df['res_name'][ind]]*len(AA_df['atom_names'][ind]))
                At_df_interface['atom_id'].extend([x for x in range(AA_df['atom_id_range'][ind][0],AA_df['atom_id_range'][ind][1])])
                At_df_interface['atom_name'].extend([x for x in AA_df['atom_names'][ind]])
                At_df_interface['chain_id'].extend([AA_df['chain_id'][ind]]*len(AA_df['atom_names'][ind]))
                At_df_interface['atom_coords'].extend(atom_coords[AA_df['atom_id_range'][ind][0]:AA_df['atom_id_range'][ind][1]].tolist())

        self.AA_df = pd.DataFrame(AA_df)
        self.Atom_df = pd.DataFrame(At_df_interface)
        self.Atom_coords = At_df_interface['atom_coords']