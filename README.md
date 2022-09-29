# lDDT_align
## A tool to align protein structures while maximizing lDDT

Here, the lDDT score is calculated on single-atom amino acid representation (either $C\alpha$ or sidechain centroids).
**No validation of stereochemical plausibility is performed.**
The algorithm uses dynamic programming to find the optimal alignment, with some heuristics to speed up computation on larger structures.

## Installation

To download and install the software, run:

```
git clone https://github.com/clami66/lDDT_align.git
cd lDDT_align
python -m pip install .
```

## Usage

```
lDDT_align [-h] [--thresholds thr [thr ...]] [--inclusion-radius R0] [--atom-type type] [--scale SCALE]
               [--gap-penalty GAP_PEN]
               ref query

Performs structural alignment of two proteins in order to optimize their lDDT

positional arguments:
  ref                   Reference protein PDB
  query                 Query protein PDB

optional arguments:
  -h, --help            show this help message and exit
  --thresholds thr [thr ...], -t thr [thr ...]
                        List of thresholds for lDDT scoring (default: [0.5, 1, 2, 4])
  --inclusion-radius R0, -r0 R0
                        Inclusion radius (default: 15.0)
  --atom-type type, -a type
                        Atom type to calculate distances (choices: {CA, centroid}, default: CA)
  --scale SCALE, -s SCALE
                        Scale factor for the initial alignment (default: 1)
  --prefilter PREFILTER, -f PREFILTER
                        Minimum scaled lDDT score to calculate lDDT on full-scale protein (default: 0.3)
  --gap-penalty GAP_PEN, -g GAP_PEN
                        Penalty to open or extend a gap in the alignment (default: 0.0)

```

## Example: aligning two PDB files

Aligning a CASP14 model to the native PDB structure:

```
$ lDDT_align test/6t1z.pdb test/T1024TS472_1-D1

Reference: test/6t1z.pdb
Query: test/T1024TS472_1-D1
Total lDDT score: 0.711

Ref.	Score	Query
G	-	-
K	0.64	K
E	0.72	E
D	0.78	D
...
K	-	-
T	-	-
K	-	-
```

## Example: aligning a query against a database of structures

It is also posible to quickly align a query against a reference DB to discover proteins with similar fold. In this case, only the total lDDT is output for each query-reference pair.
First, build the DB from a folder of PDB files into a pickle file:

```
$ python src/lDDT_align/buildDB.py data/scop-pdb/ data/scop-pdb.pkl
```

Then launch the alignments against the pickle file:

```
$ lDDT_align data/scop-pdb.pkl test/T1024TS472_1-D1

Reference Target lDDT
d1ejfa_ T1024TS472_1-D1 0.192
d1ekea_ T1024TS472_1-D1 0.243
d1e32a3 T1024TS472_1-D1 0.243
...

```
