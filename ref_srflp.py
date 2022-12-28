import numpy as np

def load_problem(fname):
    with open(fname) as inp:
        dim = int(inp.readline())
        ls = np.fromstring(inp.readline().strip(), dtype=int, sep=' ')
        c = np.loadtxt(inp)
        c = np.maximum( c, c.transpose())
        i, j = c.shape

        return (ls, c)

def srflp_d(l, q, r, perm):
    length = l[perm[q]]/2 + l[perm[r]]/2    
    # the number of possible elements between
    for s in range(q + 1, r):
        length = length + l[perm[s]]

    return length

def srflp_permutation(perm, instance):
    l, c = instance[0], instance[1]
    n = len(l)
    fit = 0

    for q in range(n - 1):
        for r in range(q + 1, n):
            print('', perm[q], perm[r], c[perm[q]][perm[r]], srflp_d(l, q, r, perm))
            fit = fit + c[perm[q]][perm[r]] * srflp_d(l, q, r, perm)

    return fit

l, c = load_problem('Y-10_t.txt')

perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(srflp_permutation(perm, (l, c)))

perm = [0, 4, 1, 9, 6, 3, 7, 2, 5, 8]
print(srflp_permutation(perm, (l,c)))

