import numpy as np

def count_shared(dist1, dist2, threshold):
    distance_diff = np.abs(dist1 - dist2) < threshold
    return np.count_nonzero(distance_diff)


def score_match(dist1, dist2, diff, selection1, thresholds, n_dist):
    selection2 = selection1 + diff
    selection2 = selection2[(selection2 < dist2.shape[-1]) & (selection2 >= 0)][:n_dist]
    selection1 = selection1[:selection2.shape[-1]]
    return count_shared(dist1[selection1], dist2[selection2], thresholds) / n_dist


def fill_table(dist1, dist2, thresholds, r0, gap_pen, path):
    l1 = dist1.shape[0]
    l2 = dist2.shape[0]    
    local_lddt = np.zeros((l1, l2))
    table = np.zeros((l1, l2))
    #trace = np.zeros((l1, l2))
    #table = [[0 for j in range(l2)] for i in range(l1)]
    trace = [[0 for j in range(l2)] for i in range(l1)]
    n_thr = len(thresholds)
    selection = (dist1 < r0) & (dist1 != 0)
    for i in range(5):
        np.fill_diagonal(selection[:,i:], False)
        np.fill_diagonal(selection[i:,:], False)

    n_total_dist = np.count_nonzero(selection, axis=0)
    n_total_dist[n_total_dist == 0] = 1

    selection2 = (dist2 < r0) & (dist2 != 0)
    n_total_dist2 = np.count_nonzero(selection2, axis=0)
    n_total_dist2[n_total_dist2 == 0] = 1

    selections = [np.where(selection[i, :])[0] for i in range(l1)]

    for i in range(0, l1):
        n_total_dist_i = n_total_dist[i]
        selection_i = selections[i]
        dist1_i = dist1[i]
        
        trace_i = trace[i]
        #table_i_1 = table[i-1]
        table_i = table[i]
        for j in range(0, l2):
            # if a previous rough path has been established, fill only around that
            if path is None or path[i, j]:
                delete = table[i-1, j] if i > 0 else -gap_pen
                delete -= gap_pen if i > 0 and not trace[i - 1][j] else 0
                insert = table[i, j - 1] if j > 0 else -gap_pen
                insert -= gap_pen if j > 0 and not trace_i[j - 1] else 0

                match = table[i-1, j - 1] if i > 0 and j > 0 else 0
                if match + n_total_dist2[j]/n_total_dist_i > delete and match + n_total_dist2[j]/n_total_dist_i > insert:
                    for threshold in thresholds:
                        #local_lddt[i, j] += score_match(dist1_i, dist2[j], j - i, selection_i, threshold, n_total_dist_i,)
                        match += score_match(dist1_i, dist2[j], j - i, selection_i, threshold, n_total_dist_i,) / n_thr
                    #match /= n_thr
                    #match += local_lddt[i, j] / n_thr
                if match > insert and match > delete:
                    table[i, j] = match
                    trace_i[j] = 0
                elif insert > delete:
                    table[i, j] = insert
                    trace_i[j] = 1
                else:
                    table[i, j] = delete
                    trace_i[j] = 2

    # lddt is normalized by the query length
    global_lddt = table[-1][-1] / l2
    return trace, global_lddt
