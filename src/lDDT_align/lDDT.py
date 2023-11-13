#!/usr/bin/env python3

import argparse
import pickle
import warnings
from os.path import basename
from pathlib import Path
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy import ndimage
from numpy.lib.stride_tricks import sliding_window_view
from .dynamic_programming import fill_table, traceback


def cache_distances(pdb, atom_type="CA"):

    backbone_ids = ["N", "C", "O"]
    model = pdb
    reslist = [_ for _ in model.get_residues() if PDB.is_aa(_)]
    nres = len(reslist)

    coords = np.zeros((nres, 3))
    sequence = ["*"] * nres

    for i, residue in enumerate(reslist):
        centroid_counter = 1
        for atom in residue.get_atoms():
            if atom_type == "CA" and atom.get_id() == "CA":
                coords[i, :] = atom.get_coord()
                sequence[i] = seq1(residue.get_resname())
            elif atom_type == "centroid" and atom.get_id() not in backbone_ids:
                coords[i, :] += atom.get_coord()
                if centroid_counter == 1:
                    sequence[i] = seq1(residue.get_resname())
                centroid_counter += 1
            if sequence[i] == "*": # missing CA or centroid get_atoms, get any coordinate
                coords[i, :] = atom.get_coord()
                sequence[i] = seq1(residue.get_resname())
        coords[i, :] /= centroid_counter

    distances = np.sqrt(
        np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2)
    ).astype(np.float32)
    distances[np.where(np.sum(coords, axis=1) == 0), :] = distances[
        :, np.where(np.sum(coords, axis=1) == 0)
    ] = 0

    return "".join(sequence), distances


def lDDT(dist1, dist2, thresholds=[0.5, 1, 2, 4], r0=15.0):

    selection = (dist1 < r0) & (dist1 != 0)
    dist2[dist1 > r0] = max(thresholds) + 1
    dist2[dist2 == 0] = max(thresholds) + 1

    lddt = distance_difference(dist1[selection], dist2[selection], thresholds)

    return lddt


def fill_table_broadcast(dist1, dist2, threshold, r0, gap_pen, path):
    l1 = dist1.shape[0]
    l2 = dist2.shape[0]
    local_lddt = np.zeros((l1, l2))
    table = np.zeros((l1, l2))
    trace = np.zeros((l1, l2))

    selection = (dist1 < r0) & (dist1 != 0)
    deselection = (dist1 > r0) | (dist1 == 0)

    n_total_dist = np.count_nonzero(selection, axis=0)
    n_total_dist[n_total_dist == 0] = 1

    dist2_pad = np.pad(dist2, (0, l1 - 1))
    dist2_pad[l2:, l2:] = dist2[: l1 - 1, : l1 - 1]

    dist2_pad = sliding_window_view(dist2_pad, (l1, l1))
    dist2_pad = np.diagonal(dist2_pad)
    dist1[deselection] = 10000

    diff = np.abs(dist1[:, :, np.newaxis] - dist2_pad)
    local_lddt = (
        np.count_nonzero(diff < threshold, axis=0) / n_total_dist[:, np.newaxis]
    )

    for i in range(0, l1):
        for j in range(0, l2):
            # if a previous rough path has been established, fill only around that
            if path[i, j] if path is not None else True:
                delete = table[i - 1, j] if i > 0 else -gap_pen
                delete -= gap_pen if i > 0 and not trace[i - 1, j] else 0
                insert = table[i, j - 1] if j > 0 else -gap_pen
                insert -= gap_pen if j > 0 and not trace[i, j - 1] else 0

                match = table[i - 1, j - 1] if i > 0 and j > 0 else 0
                match += local_lddt[i, j - i] if j >= i else local_lddt[i, l2 - (i - j)]
                if match > insert and match > delete:
                    table[i, j] = match
                    trace[i, j] = 0
                elif insert > delete:
                    table[i, j] = insert
                    trace[i, j] = 1
                else:
                    table[i, j] = delete
                    trace[i, j] = 2

    # lddt is normalized by the query length
    global_lddt = table[-1, -1] / l2
    return table, trace, global_lddt


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
    fp=False,
):
    l1 = dist1.shape[-1]
    l2 = dist2.shape[-1]

    # downscale structures
    dist1 = dist1[::scale, ::scale]
    dist2 = dist2[::scale, ::scale]
    seq1 = seq1[::scale]
    seq2 = seq2[::scale]

    # two dynamic programming steps:
    trace, global_lddt, local_lddt = fill_table(
        dist1, dist2, thresholds, r0, gap_pen, path, fp
    )
    alignment1, alignment2, pipes, path = traceback(
        trace,
        local_lddt,
        seq1,
        seq2,
    )

    if scale > 1:
        path = ndimage.binary_dilation(path, iterations=2)
        path = np.kron(path, np.ones((scale, scale))).astype(np.uint8)

    return global_lddt, (alignment1, alignment2, pipes), path


def align_pair(ref_seq, ref_distances, decoy_seq, decoy_distances, args):
    path = np.ones((ref_distances.shape[0], decoy_distances.shape[0])).astype(np.uint8)
    if args.scale > 1:
        # The initial search is done by scaling down the structure of a factor args.scale
        lddt, alignments, path = align(
            ref_distances,
            decoy_distances,
            ref_seq,
            decoy_seq,
            thresholds=args.thresholds,
            r0=args.r0,
            path=path,
            scale=args.scale,
            gap_pen=args.gap_pen,
            fp=args.fp,
        )

    if args.scale == 1 or lddt > args.prefilter:
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
            fp=args.fp,
        )

    return lddt, alignments


def run(args):

    try:
        parser = PDBParser()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = parser.get_structure("reference", args.ref)[0]
            query = parser.get_structure("query", args.query)[0]
            
            if args.query_chains:
                remove_extra_chains(query, args.query_chains)
            if args.reference_chains:
                remove_extra_chains(ref, args.reference_chains)
    except Exception as e:
        print(e)

    decoy_seq, decoy_distances = cache_distances(query, atom_type=args.atom_type)
    ref_seq, ref_distances = cache_distances(ref, atom_type=args.atom_type)

    lddt, alignments = align_pair(
        ref_seq, ref_distances, decoy_seq, decoy_distances, args=args
    )
    return lddt, alignments


def run_db(args):

    try:
        parser = PDBParser()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            query = parser.get_structure("query", args.query)[0]

    except Exception as e:
        print(e)

    ref_seq, ref_distances = cache_distances(query, atom_type=args.atom_type)
    ref_name = basename(args.query)
    with open(args.ref, "rb") as f:
        ref_data = pickle.load(f)
    path = None
    print("Reference Target lDDT")
    for query_name, (query_seq, query_distances) in ref_data.items():

        lddt, alignments = align_pair(
            ref_seq, ref_distances, query_seq, query_distances, args=args
        )

        print(f"{ref_name} {query_name} {lddt:.3f}")
    return


def format_alignment(alignments, local_lddt, aln_type="standard", hit_id=""):
    if aln_type == "horizontal":
        formatted_alignment = "".join(alignments[0]) + "\n"
        formatted_alignment += "".join([":" if (element[0] != "-" and element[1] != "-" and element[2] != "-") else " " for element in zip(local_lddt, alignments[0], alignments[1])]) + "\n"
        formatted_alignment += "".join(alignments[1]) + "\n"
    elif aln_type == "stockholm": # minimal stokholm format that works in alphafold
        # GS record, header
        formatted_alignment = f"#=GS {hit_id}/1-{len(alignments[0])} DE [subseq from] mol:protein length:{len([aa for aa in alignments[0] if aa != '-'])}\n\n"
        # actual alignment
        formatted_alignment +=  f"{hit_id}/1-{len(alignments[0])}           "
        formatted_alignment += "".join([aa if alignments[1][i] != '-' else "." for i, aa in enumerate(alignments[0])]) + "\n"
        formatted_alignment += "\n"
    else:
        formatted_alignment = f"Ref.\tScore\tQuery\n"
        for i in range(len(alignments[0])):
            formatted_alignment += f"{alignments[0][i]}\t{local_lddt[i][:4]}\t{alignments[1][i]}\n"

    return formatted_alignment


def remove_extra_chains(model, chains_to_keep):
    chains = [chain.id for chain in model.get_chains()]

    chains_to_remove = set(chains).difference(set(chains_to_keep))
    for chain in chains_to_remove:
        model.detach_child(chain)

def main():

    parser = argparse.ArgumentParser(
        description="Performs structural alignment of two proteins in order to optimize their mutual lDDT"
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
        default=[0.5, 1, 2, 4],
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
        default=1,
        type=int,
        help="Scale factor for the initial alignment (default: %(default)s)",
    )
    parser.add_argument(
        "--prefilter",
        "-f",
        dest="prefilter",
        default=0.3,
        type=float,
        help="Minimum scaled lDDT score to calculate lDDT on full-scale protein (default: %(default)s)",
    )
    parser.add_argument(
        "--gap-penalty",
        "-g",
        dest="gap_pen",
        type=float,
        default=0.0,
        help="Penalty to open or extend a gap in the alignment (default: %(default)s)",
    )
    parser.add_argument(
        "--false-positives",
        "-fp",
        dest="fp",
        action="store_true",
        help="Penalise distances within inclusion radius in the query that don't match the reference",
    )
    parser.add_argument(
        "--alignment_type",
        "-at",
        dest="alignment_type",
        default="long",
        type=str,
        help="Style of ouput alignment [long, horizontal, stockholm] (default: %(default)s))",
    )
    parser.add_argument(
        "--query_chains",
        "-qc",
        type=str,
        help="Align specific chain(s) in the query",
        nargs="+",
    )
    parser.add_argument(
        "--reference_chains",
        "-rc",
        type=str,
        help="Align specific chain(s) in the reference",
        nargs="+",
    )
    args = parser.parse_args()

    if args.ref[-3:] == "pkl":
        run_db(args)
    else:
        lddt, alignments = run(args)
        local_lddt = alignments[2].split()
        if args.alignment_type != "stockholm":
            print(f"Reference: {args.ref}")
            print(f"Query: {args.query}")
            print(f"Total lDDT score: {lddt}\n")
        print(format_alignment(alignments, local_lddt, aln_type=args.alignment_type, hit_id=Path(args.ref).stem))
