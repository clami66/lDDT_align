#!/usr/bin/env python3

import argparse
from sys import argv, maxsize
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import matplotlib.pyplot as plt
from scipy import ndimage


def cache_distances(pdb, atom_type="CA"):

    backbone_ids = ["N", "C", "O"]
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
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                aln1 += "-"
                aln2 += seq2[j]
                j -= 1
            else:
                aln1 += seq1[i]
                aln2 += "-"
                i -= 1
            path[i, j] = 1
        while i >= 0:
            aln1 += seq1[i]
            aln2 += "-"
            i -= 1
            path[i, j] = 1
        while j >= 0:
            aln2 += seq2[j]
            aln1 += "-"
            j -= 1
            path[i, j] = 1
    return aln1[::-1], aln2[::-1], path


def distance_difference(dist1, dist2, thresholds):
    d = 0
    n_dist1 = dist1.shape[-1]
    n_dist2 = dist2.shape[-1]

    n_dist = min(n_dist1, n_dist2)
    distance_diff = np.abs(dist1[:, :n_dist] - dist2[:, :n_dist])
    n_total = n_dist1 if n_dist1 > 0 else 1

    for threshold in thresholds:
        n_conserved = np.count_nonzero(distance_diff < threshold)
        d += n_conserved / n_total

    return d / len(thresholds)


def score_match(dist1, dist2, i, j, selection1, thresholds):
    # selection1 = np.where(selection[i, :])
    selection2 = selection1[0] + j - i
    selection2 = (selection2[(selection2 < dist2.shape[-1]) & (selection2 >= 0)],)

    return distance_difference(dist1[i, selection1], dist2[j, selection2], thresholds)


def align(
    dist1,
    dist2,
    seq1,
    seq2,
    thresholds=[0.5, 1, 2, 4],
    r0=15.0,
    gap_pen=0,
    scale=1,
    path=None,
):
    l1 = dist1.shape[-1] // scale
    l2 = dist2.shape[-1] // scale
    local_lddt = np.zeros((l1, l2))
    table = np.zeros((l1, l2))
    trace = np.zeros((l1, l2))

    selection = (dist1[::scale, ::scale] < r0) & (dist1[::scale, ::scale] != 0)
    selection_i = [np.where(selection[i, :]) for i in range(l1)]
    dist2[dist2 == 0] = max(thresholds) + 1

    # fill in table
    for i in range(0, l1):
        for j in range(0, l2):
            if path[i, j] if path is not None else True:
                delete = table[i - 1, j] - gap_pen if i > 0 else -gap_pen
                insert = table[i, j - 1] - gap_pen if j > 0 else -gap_pen

                match = table[i - 1, j - 1] if i > 0 and j > 0 else 0
                local_lddt[i, j] = score_match(
                    dist1[::scale, ::scale],
                    dist2[::scale, ::scale],
                    i,
                    j,
                    selection_i[i],
                    thresholds,
                )
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

    global_lddt = table[-1, -1] / min(l1, l2)

    alignment1, alignment2, path = traceback(
        trace, seq1[::scale], seq2[::scale], trace.shape[0] - 1, trace.shape[1] - 1
    )
    path = ndimage.binary_dilation(path, iterations=scale)
    path = np.kron(path, np.ones((scale, scale)))
    path = np.pad(
        path,
        ((0, dist1.shape[-1] - path.shape[0]), (0, dist2.shape[-1] - path.shape[1])),
        "maximum",
    )
    return global_lddt, (alignment1, alignment2), path


def run(args):

    try:
        parser = PDBParser()

        ref = parser.get_structure("decoy", args.ref)[0]
        decoy = parser.get_structure("reference", args.query)[0]

    except Exception as e:
        print(e)

    decoy_seq, decoy_distances = cache_distances(decoy, atom_type=args.atom_type)
    ref_seq, ref_distances = cache_distances(ref, atom_type=args.atom_type)

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
    parser.add_argument("--atom-type", "-a", metavar="type", type=str, choices=["CA", "centroid"], default="CA", help="Atom type to calculate distances (choices: {%(choices)s}, default: %(default)s)")
    parser.add_argument(
        "--scale",
        "-s",
        dest="scale",
        default=3,
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

    lddt, alignments = run(args)

    print(f"Global lDDT score: {lddt}")
    print(alignments[0])
    print(alignments[1])


if __name__ == "__main__":
    main()
