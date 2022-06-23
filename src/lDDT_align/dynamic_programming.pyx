import numpy as np
cimport numpy as np

cdef float score_match(double [:] dist1, double [:] dist2, int diff, long [:] selection1, float threshold, int n_dist):
    cdef:
        int i, l1, l2, sel1, sel2, n_sel
        float c = 0
        float d
        float tpr

    l1 = dist1.shape[0]
    l2 = dist2.shape[0]
    n_sel = selection1.shape[0]
    for i in range(n_sel):
        sel1 = selection1[i]
        sel2 = sel1 + diff
        if sel2 < l2 and sel2 >= 0:
            d = dist1[sel1] - dist2[sel2]
            if d < threshold and d > -threshold:
                c += 1
    tpr = c/n_dist
    return tpr


cdef select(double [:,:] dist, float r0, int l):
    cdef:
        int i, j
        unsigned char [:,:] selection = np.zeros((l, l)).astype(np.uint8)
    
    for i in range(l):
        for j in range(i+5, l):
            if dist[i, j] < r0:
                selection[i, j] = selection[j, i] = 1
    return selection


def fill_table(double [:,:] dist1, double [:,:] dist2, list thresholds, float r0, float gap_pen, unsigned char [:,:] path):
    cdef:
        int i, j, n_total_dist_i, diff
        int n_thr = len(thresholds)
        int l1 = dist1.shape[0]
        int l2 = dist2.shape[0]
        list selections
        float delete, insert, match, global_lddt, threshold
        float score
        float [:, :] table = np.zeros((l1, l2)).astype(np.float32)
        long [:] selection_i
        unsigned char [:, :] trace = np.zeros((l1, l2)).astype(np.uint8)
        long [:] n_total_dist
        unsigned char [:,:] selection
        double [:] dist1_i

    selection = select(dist1, r0, l1)

    n_total_dist = np.count_nonzero(selection, axis=0).astype(np.int)
    #n_total_dist[n_total_dist == 0] = 1
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
                    threshold = thresholds[t]
                    diff = j - i
                    score = score + score_match(dist1_i, dist2[j], diff, selection_i, threshold, n_total_dist_i,)
                match = match + score/n_thr
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
    global_lddt = table[l1-1, l2-1] / l2
    return trace, global_lddt
