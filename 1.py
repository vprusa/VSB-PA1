import itertools as it
import multiprocessing as mp
import numpy as np

'''
https://homel.vsb.cz/~kro080/PAI-2022/U1/

https://youtrack.jetbrains.com/issue/PY-52273/Debugger-multiprocessing-hangs-pycharm-2021.3
https://youtrack.jetbrains.com/issue/PY-37366/Debugging-with-multiprocessing-managers-no-longer-working-after-update-to-2019.2-was-fine-in-2019.1#focus=Comments-27-4690501.0-0
'''

max_val = 1000000.0

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

def srflp_permutation(perm, l, c, cur_best_solution):
    '''
    Vyhodnoceni permutave
    '''
    n = len(l)
    fit = 0
    for q in range(n - 1):
        for r in range(q + 1, n):
            # print('', perm[q], perm[r], c[perm[q]][perm[r]], srflp_d(l, q, r, perm))
            fit = fit + c[perm[q]][perm[r]] * srflp_d(l, q, r, perm)
            if cur_best_solution < fit:
                 return [True, fit, q, perm[q], r, perm[r]]
    return [False, fit]

def paral_srflp_permutation(l, c, s, best_found, lock):
  '''
  Vyhodnoceni casti permutaci dle 's'. 's' je prefix [s , a ... len(c)].
  Efektivita paralelismu je ovlivnena 'len(c)' a 'pool_size'.
  '''
  n = len(c)
  perm_gen_base = [a for a in range(n) if a != s]
  print(s, perm_gen_base)

  skip = False
  index_to_skip = -1
  index_val_to_skip = -1
  for perm_gen in it.permutations(perm_gen_base):
      perm = [s]
      perm.extend(perm_gen)

      if index_to_skip != -1 and perm[index_to_skip] == index_val_to_skip:
          continue
      cur_best_solution = max_val

      with lock:
          cur_best_solution = best_found.value[0]

      vals = srflp_permutation(perm, l, c, cur_best_solution)
      skip = vals[0]
      val = vals[1]
      if skip:
          index_to_skip = vals[2]
          index_val_to_skip = vals[3]
          continue
      else:
          index_to_skip = -1
          index_val_to_skip = -1

      with lock:
        if val < best_found.value[0]:
            best_found.value = [val, perm]
            print(s, best_found.value[0], best_found.value[1])

  return best_found.value[0]

'''
Hlavni funkce, ktera se stara o 
- nacteni dat
- nastaveni paralelizmu - manager, lock, shared_var 
'''
def run():
  l, c = load_problem('Y-10_t.txt')

  pool_size = 8
  # budeme pracovat ve sdilene pameti
  with mp.Manager() as manager:
    lock = manager.Lock()
    # max val and empty permutation
    best_found = manager.Value('d', [max_val, []])
    splitter = list(range(0, len(c)-1))
    # spustime ve vice procesech
    with mp.Pool(processes=pool_size) as pool:
      # kazde vlakno v poolu se stara o cast vypoctu, rozdeleno dle seznamu 'splitter'
      ret = pool.starmap(paral_srflp_permutation, zip(it.repeat(l), it.repeat(c), splitter, it.repeat(best_found), it.repeat(lock)))
    # vypsani nejlepsich reseni
    print(best_found.value[0])
    print(best_found.value[1])
    # ulozeni vysledku do souboru
    output_file = open('1.py.res.txt', 'w')
    output_file.write('Val:\n' + str(best_found.value[0]))
    output_file.write('Perm:\n' + " ".join(best_found.value[1]))
    output_file.close()
  # print(ret)

run()
