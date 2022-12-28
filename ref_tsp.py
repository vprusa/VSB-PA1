import itertools as it        # req. step 2
import multiprocessing as mp  # req. step 3
import numpy as np

def evaluate_bnb(distances, perm, n, best_found):
  '''
  Postupne pocita cenu permutace. 
  V pripade, ze prekroci best_found, vrati False a index prvku, 
  u ktereho to nastalo.
  '''
  sum = 0
  for i in range(n-1):
    sum = sum + distances[perm[i]*n + perm[i+1]]
    if sum >= best_found:
      #print('Skipping because: ', sum, 'at', i)
      return False, i
  sum = sum + distances[perm[-1]*n + perm[0]]
  return True, sum

def perms_prefix_bnb_shared(distances, n, s, best_found, lock):
  '''
  Vyhodnoti vsechny permutace, ktere zacinaji na [0, s, ..., n ].
  Nejlepsi nalezene cesty uklada do sdilene promenne _best_found_.
  Pouziva zamek _lock_ pro hlidani pristupu do ni.
  '''
  perm = [a for a in range(n) if a != 0 and a != s]
  skip = False
  val = 0
  current = 0
  success = True
  
  for p in it.permutations(perm):
    perm = [0, s]
    perm.extend(p)
    
    if skip:
      if perm[val] == current:
        continue
      else:
        skip = False
        success, val = evaluate_bnb(distances, perm, n, best_found.value)
    else:
      success, val = evaluate_bnb(distances, perm, n, best_found.value)
    
    if success:
    # nalezl novou nejlepsi permutaci, hodnotu si ulozi  
      with lock:
        if val < best_found.value:
          best_found.value = val
    else:
    # nova nejlepsi nebyla nalezena, budeme muset 'urezat vetev'.
    # to se pozna tak, ze na pozici, kde doslo k preteceni best_val
    # se objevi jiny prvek. 
      current = perm[val]
      skip = True
  
  return best_found.value

def run_step6():
  n = 8
  np.random.seed(42)
  distances = np.random.random(size=(n, n))
  distances = distances + distances.T
  distances = distances.flatten()
  
  Is = list(range(1, n))
  
  # budeme pracovat ve sdilene pameti
  with mp.Manager() as manager:
    lock = manager.Lock()
    best_found = manager.Value('d', 1000000.0)   

    # spustime ve vice procesech
    with mp.Pool(processes=4) as pool:
      ret = pool.starmap(perms_prefix_bnb_shared, zip(it.repeat(distances), it.repeat(n), Is, it.repeat(best_found), it.repeat(lock)))

  print(ret)

run_step6()
