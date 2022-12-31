import numpy as np

'''
https://homel.vsb.cz/~kro080/PAI-2022/U3/ukol3.html

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


'''
'''
def run():
  l, c = load_problem('Y-10_t.txt')


run()
