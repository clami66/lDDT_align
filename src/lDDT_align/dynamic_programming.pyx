import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray

def traceback(unsigned char [:, :] trace, str seq1, str seq2):
    cdef:
        str aln1 = ""
        str aln2 = ""
        str pipes = ""
        int i = trace.shape[0] - 1
        int j = trace.shape[1] - 1

        int upper_band = j - i
        int lower_band = j - i
        unsigned char [:,:] path = np.zeros((trace.shape[0], trace.shape[1])).astype(np.uint8)
        unsigned char [:] trace_i

    path[i - 1, j - 1] = 1
    trace_i = trace[i]
    while i >= 0 if j > i else j >= 0:
        
        while j >= 0 if i > j else i >= 0:
            if trace_i[j] == 0:
                aln1 = seq1[i] + aln1
                aln2 = seq2[j] + aln2
                pipes = ":" + pipes
                i -= 1
                j -= 1
                trace_i = trace[i]
            elif trace_i[j] == 1:
                aln1 = "-" + aln1
                aln2 = seq2[j] + aln2
                pipes = " " + pipes
                j -= 1
            else:
                aln1 = seq1[i] + aln1
                aln2 = "-" + aln2
                pipes = " " + pipes
                i -= 1
                trace_i = trace[i]
            path[i, j] = 1
        while i >= 0:
            aln1 = seq1[i] + aln1
            aln2 = "-" + aln2
            pipes = " " + pipes
            i -= 1
            path[i, j] = 1
        while j >= 0:
            aln2 = seq2[j] + aln2
            aln1 = "-" + aln1
            pipes = " " + pipes
            j -= 1
            path[i, j] = 1
    return aln1, aln2, pipes, path


cdef float score_match(float [:] dist1, float [:] dist2, int diff, long [:] selection1, float threshold, int n_dist):
    cdef:
        int i, l1, l2, sel1, sel2, n_sel
        float c = 0
        float d
        float tpr

    l1 = dist1.shape[0]
    l2 = dist2.shape[0]
    n_sel = selection1.shape[0]

    if n_sel == 0:
        return 1.0

    for i in range(n_sel):
        sel1 = selection1[i]
        sel2 = sel1 + diff
        if sel2 < l2 and sel2 >= 0:
            d = dist1[sel1] - dist2[sel2]
            if d < threshold and d > -threshold:
                c += 1
    tpr = c/n_dist

    return tpr


cdef select(float [:,:] dist, float r0, int l):
    cdef:
        int i, j
        unsigned char [:,:] selection = np.zeros((l, l)).astype(np.uint8)
    
    for i in range(l):
        for j in range(i+5, l):
            if dist[i, j] < r0:
                selection[i, j] = selection[j, i] = 1
    return selection


def fill_table(float [:,:] dist1, float [:,:] dist2, list thresholds, float r0, float gap_pen, unsigned char [:,:] path):
    cdef:
        int i, j, t, n_total_dist_i, diff
        int n_thr = len(thresholds)
        int l1 = dist1.shape[0]
        int l2 = dist2.shape[0]
        list selections
        float delete, insert, match, global_lddt, threshold
        float score
        float [:, :] table = np.zeros((l1, l2)).astype(np.float32)
        float [:, :] local_lddt = np.zeros((l1, l2)).astype(np.float32)
        long [:] selection_i
        long [:] n_total_dist
        unsigned char [:, :] trace = np.zeros((l1, l2)).astype(np.uint8)
        unsigned char [:,:] selection
        float [:] dist1_i

    # A copy of the thresholds list as an array so that it can be accessed quicker
    thresholds_array = cvarray(shape=(n_thr, ), itemsize=sizeof(float), format="f")
    cdef float [:] thresholds_view = thresholds_array
    for t in range(n_thr):
        thresholds_view[t] = thresholds[t]

    selection = select(dist1, r0, l1)

    n_total_dist = np.count_nonzero(selection, axis=0).astype(np.int)
    for i in range(l1):
        if n_total_dist[i] == 0:
            n_total_dist[i] = 1

    selections = [np.where(selection[i, :])[0] for i in range(l1)]

    for i in range(0, l1):
        n_total_dist_i = n_total_dist[i]
        selection_i = selections[i]
        dist1_i = dist1[i]
        for j in range(0, l2):            
            # if a previous rough path has been established, fill only around that
            if path[i, j]:
                delete = table[i-1, j] if i > 0 else -gap_pen
                delete -= gap_pen if i > 0 and not trace[i - 1, j] else 0
                insert = table[i, j - 1] if j > 0 else -gap_pen
                insert -= gap_pen if j > 0 and not trace[i, j - 1] else 0

                match = table[i-1, j - 1] if i > 0 and j > 0 else 0
                score = 0
                for t in range(n_thr):
                    threshold = thresholds_view[t]
                    diff = j - i
                    score = score + score_match(dist1_i, dist2[j], diff, selection_i, threshold, n_total_dist_i,)
                local_lddt[i, j] = score/n_thr
                match = match + local_lddt[i,j]
                if match >= insert and match >= delete:
                    table[i, j] = match
                    trace[i, j] = 0
                elif insert > delete:
                    table[i, j] = insert
                    trace[i, j] = 1
                else:
                    table[i, j] = delete
                    trace[i, j] = 2

    # lddt is normalized by the reference length
    global_lddt = table[l1-1, l2-1] / l1
    return trace, global_lddt
