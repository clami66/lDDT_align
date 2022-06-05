#!/usr/bin/env python3

from sys import argv, maxsize
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist
np.set_printoptions(threshold=maxsize)

def cache_distances(pdb):
    
    backbone_ids = ["N", "CA", "C", "O"]    
    model = pdb
    nres = len([_ for _ in model.get_residues() if PDB.is_aa(_)])

    coords = np.zeros((2, nres, 3))
    sequence = ""

    for i, residue in enumerate([_ for _ in model.get_residues() if PDB.is_aa(_)]):
        centroid_counter = 1
        for atom in residue.get_atoms():
            if atom.get_id() == "CA":
                coords[0, i,     :] = atom.get_coord()                
                coords[1, i, :] = atom.get_coord()
                sequence += seq1(residue.get_resname())
            elif atom.get_id() not in backbone_ids:
                coords[1, i, :] += atom.get_coord()
                centroid_counter += 1
        coords[1, i, :] /= centroid_counter

    ca_distances = cdist(coords[0], coords[0])
    ce_distances = cdist(coords[1], coords[1])
    ca_distances[np.where(np.sum(coords, axis=1) == 0), :] = ca_distances[:, np.where(np.sum(coords, axis=1) == 0)] = 0
    ce_distances[np.where(np.sum(coords, axis=1) == 0), :] = ce_distances[:, np.where(np.sum(coords, axis=1) == 0)] = 0
    
    distances = np.array([ca_distances, ce_distances])
    return sequence, distances

def distance_difference(dist1, dist2, thresholds):
    d = 0
    distance_diff = np.abs(dist1 - dist2)
    n_total = len(thresholds) * dist1.shape[0] if dist1.shape[0] > 0 else len(thresholds)
    for threshold in thresholds:
        n_conserved = np.count_nonzero(distance_diff < threshold)
        d += n_conserved/n_total
    
    return d

def lDDT(dist1, dist2, thresholds=[0.5, 1, 2, 4], r0=15.0):

    selection = (dist1 < r0) & (dist1 != 0)
    dist2[dist1 > r0] = max(thresholds) + 1
    dist2[dist2 == 0] = max(thresholds) + 1

    lddt = distance_difference(dist1[selection], dist2[selection], thresholds)
        
    return lddt

def backtrack(picks, seq1, seq2, i, j):
    aln = ""
    
    if i > 0 or j > 0:
        if picks[i, j] == 0:
            return backtrack(picks, seq1, seq2, i-1, j-1) + seq1[i] + seq2[j] + aln
        elif picks[i, j] == 1:
            return backtrack(picks, seq1, seq2, i, j-1)  + "-" + seq2[j] + aln
        elif picks[i, j] == 2:
            return backtrack(picks, seq1, seq2, i-1, j) + seq1[i] + "-" + aln
    else:
        if picks[i, j] == 0:
            return aln + seq1[i] + seq2[j]
        elif picks[i, j] == 1:
            return aln + "-" + seq2[j]
        elif picks[i, j] == 2:
                return aln + seq1[i] + "-"

def needle2(dist1, dist2, seq1, seq2, thresholds=[0.5, 1, 2, 4], r0=15.0):
    l1 = dist1.shape[-1]
    l2 = dist2.shape[-1]
    table = np.zeros((l1, l2))
    picks = np.zeros((l1, l2))
    
    selection = (dist1 < r0) & (dist1 != 0)
    dist2[dist2 == 0] = max(thresholds) + 1
    
    # fill in table
    for i in range(l1):
        for j in range(l2):
            selected_ca = np.where(selection[0,i,:l2-j])
            selected_ce = np.where(selection[1,i,:l2-j])
            delete = table[i-1, j] if i > 0 else 0
            insert = table[i, j-1] if j > 0 else 0

            match = table[i-1, j-1] if (i > 0 and j > 0) else 0
            match += (distance_difference(dist1[0, i, selected_ca], dist2[0, j, selected_ca], thresholds) + distance_difference(dist1[1, i, selected_ce], dist2[1, j, selected_ce], thresholds))/2

            if match >= insert and match >= delete:
                table[i,j] = match
                picks[i,j] = 0
            elif insert >= delete:
                table[i,j] = insert
                picks[i,j] = 1
            else:
                table[i,j] = delete
                picks[i,j] = 2

    global_lddt = table[-1, -1]/min(l1, l2)
    
    # backtracking
    alignments = backtrack(picks, seq1, seq2, picks.shape[0]-1, picks.shape[1]-1)
    
    print(alignments[::2])
    print(alignments[1::2])
    with open("foo", "w") as out:
        out.write("/" + " ".join(seq2) + "\n")
        for i,row in enumerate(picks):
            out.write(seq1[i] + " ")
            for j,col in enumerate(row):
                out.write(str(int(col)) + " ")
            out.write("\n")
    
    return global_lddt

    
def needle(dist1, dist2, seq1, seq2, thresholds=[0.5, 1, 2, 4], r0=15.0):
    l1 = dist1.shape[0]
    l2 = dist2.shape[0]
    table = np.zeros((l1, l2))
    picks = np.zeros((l1, l2))
    selection = (dist1 < r0) & (dist1 != 0)
    dist2[dist2 == 0] = max(thresholds) + 1
    
    # fill in table
    selected = np.where(selection[0,:l2])
    init = distance_difference(dist1[1,selected], dist2[1,selected], thresholds)
    picks[0, :] = 1
    picks[:, 0] = 2
    picks[0, 0] = 0
    table[0, :] = table[:, 0] = init    
    for i in range(1, l1):
        for j in range(1, l2):
            selected = np.where(selection[i,:l2-j])
            delete = table[i-1, j]
            insert = table[i, j-1]

            match = table[i-1, j-1] + distance_difference(dist1[i, selected], dist2[j, selected], thresholds)

            if match >= insert and match >= delete:
                table[i,j] = match
                picks[i,j] = 0
            elif insert >= delete:
                table[i,j] = insert
                picks[i,j] = 1
            else:
                table[i,j] = delete
                picks[i,j] = 2

    global_lddt = table[-1, -1]/min(l1, l2)
    
    # backtracking
    picks = picks[1::2,1::2]
    alignments = backtrack(picks, seq1, seq2, picks.shape[0]-1, picks.shape[1]-1)
    
    print(alignments[::2])
    print(alignments[1::2])
    with open("foo", "w") as out:
        out.write("/" + " ".join(seq2) + "\n")
        for i,row in enumerate(picks):
            out.write(seq1[i] + " ")
            for j,col in enumerate(row):
                out.write(str(int(col)) + " ")
            out.write("\n")
    
    return global_lddt
    
def main():
    pdb1 = argv[1]
    pdb2 = argv[2]
    
    try:
        parser = PDBParser()
        
        decoy = parser.get_structure("decoy", pdb1)[0]
        ref = parser.get_structure("reference", pdb2)[0]
        
    except Exception as e:
        print(e)
        
    decoy_seq, decoy_distances = cache_distances(decoy)        
    ref_seq, ref_distances = cache_distances(ref)
    #lddt = lDDT(ref_distances, decoy_distances)
    lddt = needle2(ref_distances, decoy_distances, ref_seq, decoy_seq)
    print(lddt)

if __name__ == "__main__":
    main()
