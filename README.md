# lDDT_align
## A tool to align protein structures while maximizing global lDDT

Here, the lDDT score is calculated on single-atom amino acid representation (either CAs or sidechain centroids). The algorithm uses dynamic programming to find the optimal alignment, with some heuristics to speed up computation on larger structures.

### Usage:

```python
src/lDDT.py -h
usage: lDDT.py [-h] [--thresholds thr [thr ...]] [--inclusion-radius R0] [--atom-type type] [--scale SCALE]
               [--gap-penalty GAP_PEN]
               ref query

Performs structural alignment of two proteins in order to optimize their mutual global lDDT

positional arguments:
  ref                   Reference protein PDB
  query                 Query protein PDB

optional arguments:
  -h, --help            show this help message and exit
  --thresholds thr [thr ...], -t thr [thr ...]
                        List of thresholds for lDDT scoring (default: [2])
  --inclusion-radius R0, -r0 R0
                        Inclusion radius (default: 15.0)
  --atom-type type, -a type
                        Atom type to calculate distances (choices: {CA, centroid}, default: CA)
  --scale SCALE, -s SCALE
                        Scale factor for the initial alignment (default: 3)
  --gap-penalty GAP_PEN, -g GAP_PEN
                        Penalty to open or extend a gap in the alignment (default: 0.0)

```
