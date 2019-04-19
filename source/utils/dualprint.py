import os
def dualprint(s, f=None):
    print(s)
    if f is not None:
        f.writelines(['%s\n' % s])
        f.flush()

