#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
from numba import jit
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy import ndimage

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
    )
    distances[np.where(np.sum(coords, axis=1) == 0), :] = distances[
        :, np.where(np.sum(coords, axis=1) == 0)
    ] = 0

    return sequence, distances


def lDDT(dist1, dist2, thresholds=[0.5, 1, 2, 4], r0=15.0):

    selection = (dist1 < r0) & (dist1 != 0)
    dist2[dist1 > r0] = max(thresholds) + 1
    dist2[dist2 == 0] = max(thresholds) + 1

    lddt = distance_difference(dist1[selection], dist2[selection], thresholds)

    return lddt


def traceback(trace, seq1, seq2, i, j):
    aln1 = ""
    aln2 = ""
    pipes = ""
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    upper_band = j - i
    lower_band = j - i
    path = np.zeros((i + 1, j + 1))
    path[i - 1, j - 1] = 1
    while i >= 0 if j > i else j >= 0:
        while j >= 0 if i > j else i >= 0:
            if trace[i, j] == 0:
                aln1 += seq1[i]
                aln2 += seq2[j]
                pipes += ":"
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                aln1 += "-"
                aln2 += seq2[j]
                pipes += " "
                j -= 1
            else:
                aln1 += seq1[i]
                aln2 += "-"
                pipes += " "
                i -= 1
            path[i, j] = 1
        while i >= 0:
            aln1 += seq1[i]
            aln2 += "-"
            pipes += " "
            i -= 1
            path[i, j] = 1
        while j >= 0:
            aln2 += seq2[j]
            aln1 += "-"
            pipes += " "
            j -= 1
            path[i, j] = 1
    return aln1[::-1], aln2[::-1], pipes[::-1], path


@jit
def count_shared(dist1, dist2, threshold):
    distance_diff = np.abs(dist1 - dist2) < threshold
    return np.count_nonzero(distance_diff)


@jit
def count_non_shared(dist1, dist2, threshold):
    distance_diff = np.abs(dist1 - dist2) > threshold
    return np.count_nonzero(distance_diff)


@jit
def score_match(dist1, dist2, diff, selection1, thresholds, n_dist):
    selection2 = selection1 + diff
    selection2 = selection2[(selection2 < dist2.shape[-1]) & (selection2 >= 0)][:n_dist]
    selection1 = selection1[:selection2.shape[-1]]
    return count_shared(dist1[selection1], dist2[selection2], thresholds) / n_dist


def align(
    dist1,
    dist2,
    seq1,
    seq2,
    thresholds=[2],
    r0=15.0,
    gap_pen=0,
    scale=1,
    path=None,
):
    l1_orig = dist1.shape[-1]
    l2_orig = dist2.shape[-1]
    l1 = l1_orig // scale
    l2 = l2_orig // scale
    local_lddt = np.zeros((l1, l2))
    table = np.zeros((l1, l2))
    trace = np.zeros((l1, l2))

    if scale > 1:
        # downscale structures
        dist1 = dist1[::scale, ::scale]
        dist2 = dist2[::scale, ::scale]
        seq1 = seq1[::scale]
        seq2 = seq2[::scale]

    selection = (dist1 < r0) & (dist1 != 0)
    n_total_dist = np.count_nonzero(selection, axis=0)
    n_total_dist[n_total_dist == 0] = 1

    selection2 = (dist2 < r0) & (dist2 != 0)
    n_total_dist2 = np.count_nonzero(selection2, axis=0)
    n_total_dist2[n_total_dist2 == 0] = 1

    selection_i = [np.where(selection[i, :])[0] for i in range(l1)]

    # fill in table
    for i in range(0, l1):
        for j in range(0, l2):
            if path[i, j] if path is not None else True:
                delete = table[i - 1, j] - gap_pen if i > 0 else -gap_pen
                insert = table[i, j - 1] - gap_pen if j > 0 else -gap_pen

                match = table[i - 1, j - 1] if i > 0 and j > 0 else 0
                if match + n_total_dist2[j]/n_total_dist[i] > delete and match + n_total_dist2[j]/n_total_dist[i] > insert:
                    local_lddt[i, j] = score_match(
                        dist1[i],
                        dist2[j],
                        j - i,
                        selection_i[i],
                        thresholds[0],
                        n_total_dist[i],)
                    match += local_lddt[i, j]
                if match > insert and match > delete:
                    table[i, j] = match
                    trace[i, j] = 0
                elif insert > delete:
                    table[i, j] = insert
                    trace[i, j] = 1
                else:
                    table[i, j] = delete
                    trace[i, j] = 2

    # lddt is normalized by the reference length
    global_lddt = table[-1, -1] / l1

    alignment1, alignment2, pipes, path = traceback(
        trace, seq1, seq2, trace.shape[0] - 1, trace.shape[1] - 1
    )

    if scale > 1:
        path = ndimage.binary_dilation(path, iterations=3)
        path = np.kron(path, np.ones((scale, scale)))
        path = np.pad(
            path,
            ((0, l1_orig - path.shape[0]), (0, l2_orig - path.shape[1])),
            "maximum",
        )
    return global_lddt, (alignment1, alignment2, pipes), path


def align_pair(ref_seq, ref_distances, decoy_seq, decoy_distances, args):
    if args.scale > 1:
        # The initial search is done by scaling down the structure of a factor args.scale
        _, _, path = align(
            ref_distances,
            decoy_distances,
            ref_seq,
            decoy_seq,
            thresholds=args.thresholds,
            r0=args.r0,
            scale=args.scale,
            gap_pen=args.gap_pen,
        )
    else:
        path = None

    # The second search is full-scale, but follows the neighborhood of the path found in the first search
    lddt, alignments, _ = align(
        ref_distances,
        decoy_distances,
        ref_seq,
        decoy_seq,
        thresholds=args.thresholds,
        r0=args.r0,
        path=path,
        gap_pen=args.gap_pen,
    )

    return lddt, alignments    


def run(args):

    try:
        parser = PDBParser()

        ref = parser.get_structure("reference", args.ref)[0]
        decoy = parser.get_structure("decoy", args.query)[0]

    except Exception as e:
        print(e)

    decoy_seq, decoy_distances = cache_distances(decoy, atom_type=args.atom_type)
    ref_seq, ref_distances = cache_distances(ref, atom_type=args.atom_type)
    
    lddt, alignments = align_pair(ref_seq, ref_distances, decoy_seq, decoy_distances, args)
    return lddt, alignments

def run_db(args):

    try:
        parser = PDBParser()
        decoy = parser.get_structure("decoy", args.query)[0]

    except Exception as e:
        print(e)

    decoy_seq, decoy_distances = cache_distances(decoy, atom_type=args.atom_type)

    with open(args.ref, "rb") as f:
        ref_data = pickle.load(f)
    path = None

    for name, (ref_seq, ref_distances) in ref_data.items():

        lddt, alignments = align_pair(ref_seq, ref_distances, decoy_seq, decoy_distances, args)

        print(f"Reference: {name}")
        print(f"Query: {args.query}")
        print(f"Global lDDT score: {lddt}")
    return

def main():

    parser = argparse.ArgumentParser(
        description="Performs structural alignment of two proteins in order to optimize their mutual global lDDT"
    )
    parser.add_argument("ref", metavar="ref", type=str, help="Reference protein PDB")
    parser.add_argument("query", metavar="query", type=str, help="Query protein PDB")
    parser.add_argument(
        "--thresholds",
        "-t",
        metavar="thr",
        type=float,
        nargs="+",
        help="List of thresholds for lDDT scoring (default: %(default)s)",
        default=[2],
    )
    parser.add_argument(
        "--inclusion-radius",
        "-r0",
        dest="r0",
        type=float,
        default=15.0,
        help="Inclusion radius (default: %(default)s)",
    )
    parser.add_argument(
        "--atom-type",
        "-a",
        metavar="type",
        type=str,
        choices=["CA", "centroid"],
        default="CA",
        help="Atom type to calculate distances (choices: {%(choices)s}, default: %(default)s)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        dest="scale",
        default=3,
        type=int,
        help="Scale factor for the initial alignment (default: %(default)s)",
    )
    parser.add_argument(
        "--gap-penalty",
        "-g",
        dest="gap_pen",
        type=float,
        default=0.0,
        help="Penalty to open or extend a gap in the alignment (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.ref[-3:] == "pkl":
        run_db(args)
    else:
        lddt, alignments = run(args)

        print(f"Reference: {args.ref}")
        print(f"Query: {args.query}")
        print(f"Global lDDT score: {lddt}")
        print(alignments[0])
        print(alignments[2])
        print(alignments[1])


if __name__ == "__main__":
    main()
