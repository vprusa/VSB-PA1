import itertools as it        # req. step 2
import multiprocessing as mp  # req. step 3
import numpy as np
from pprint import pprint

'''
https://homel.vsb.cz/~kro080/PAI-2022/U1/

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0
'''

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

def srflp_permutation(perm, l, c):
    n = len(l)
    fit = 0

    for q in range(n - 1):
        for r in range(q + 1, n):
            # print('', perm[q], perm[r], c[perm[q]][perm[r]], srflp_d(l, q, r, perm))
            fit = fit + c[perm[q]][perm[r]] * srflp_d(l, q, r, perm)

    return fit


def evaluate_srflp_permutation(perm, l, c, best_found):
  '''
  Postupne pocita cenu permutace.
  V pripade, ze prekroci best_found, vrati False a index prvku,
  u ktereho to nastalo.
  '''
  res = srflp_permutation(perm, l,c)
  if res >= best_found:
    return False, res
  return True, res

def perms_srflp_permutation(l, c, s, best_found, lock):
  '''
  Vyhodnoti vsechny permutace, ktere zacinaji na [0, s, ..., n ].
  Nejlepsi nalezene cesty uklada do sdilene promenne _best_found_.
  Pouziva zamek _lock_ pro hlidani pristupu do ni.
  '''
  n = len(c)

  perm_gen_base = [a for a in range(n) if a != s]

  print(s, perm_gen_base)
  for perm_gen in it.permutations(perm_gen_base):
      perm = [s]
      perm.extend(perm_gen)

      val = srflp_permutation(perm, l, c)
      with lock:
        if val < best_found.value[0]:
            best_found.value = [val, perm]
            print(s, best_found.value[0], best_found.value[1])


  return best_found.value[0]

def run():
  l, c = load_problem('Y-10_t.txt')

  pool_size = 8
  # budeme pracovat ve sdilene pameti
  with mp.Manager() as manager:
    lock = manager.Lock()
    best_found = manager.Value('d', [1000000.0, []])

    # spustime ve vice procesech
    with mp.Pool(processes=pool_size) as pool:
      ret = pool.starmap(perms_srflp_permutation, zip(it.repeat(l), it.repeat(c), list(range(0, len(c)-1)), it.repeat(best_found), it.repeat(lock)))
    print(best_found.value[0])
    print(best_found.value[1])

  # print(ret)

run()
