#!/usr/bin/env python3

import argparse
import pickle
import glob
from os.path import basename
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def cache_distances(pdb, atom_type="CA"):

    backbone_ids = ["N", "CA", "C", "O"]
    model = pdb
    reslist = [_ for _ in model.get_residues() if PDB.is_aa(_)]
    nres = len(reslist)

    coords = np.zeros((nres, 3))
    sequence = ""

    for i, residue in enumerate(reslist):
        centroid_counter = 1
        for atom in residue.get_atoms():
            if atom_type == "CA" and atom.get_id() == "CA":
                coords[i, :] = atom.get_coord()
                sequence += seq1(residue.get_resname())
            elif atom_type == "centroid" and atom.get_id() not in backbone_ids:
                coords[i, :] += atom.get_coord()
                if centroid_counter == 1:
                    sequence += seq1(residue.get_resname())
                centroid_counter += 1
        coords[i, :] /= centroid_counter

    distances = np.sqrt(
        np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2)
    ).astype(np.float32)
    distances[np.where(np.sum(coords, axis=1) == 0), :] = distances[
        :, np.where(np.sum(coords, axis=1) == 0)
    ] = 0
    distances = distances.astype(np.float32)
    return sequence, distances


def run(args):
    
    pdb_list = glob.glob(f"{args.in_folder}/*")
    
    data = {}
    
    for pdb_file in pdb_list:
        pdb_name = basename(pdb_file)
        
        try:
            parser = PDBParser()
            struct = parser.get_structure("struct", pdb_file)[0]

        except Exception as e:
            print(e)

        data[pdb_name] = cache_distances(struct, atom_type="CA")

        print(pdb_name)
    with open(args.out, "wb") as output_file:
        pickle.dump(data, output_file)

    return


def main():

    parser = argparse.ArgumentParser(
        description="Extracts distances and sequences for a PDB dataset onto a pickle archive"
    )
    parser.add_argument("in_folder", metavar="path", type=str, help="Path to PDB folder")
    parser.add_argument("out", metavar="pickle.pkl", type=str, help="Path to output pkl file")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
