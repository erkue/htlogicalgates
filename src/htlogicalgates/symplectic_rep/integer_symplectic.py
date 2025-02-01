# canonical ordering of symplectic group elements
# from ”How to efficiently select an arbitrary clifford group element”
# by Robert Koenig and John A. Smolin

from numpy import *
from random import randint, shuffle


def directsum(m1, m2):
    n1 = len(m1[0])
    n2 = len(m2[0])
    output = zeros((n1+n2, n1+n2), dtype=int32)
    for i in range(0, n1):
        for j in range(0, n1):
            output[i, j] = m1[i, j]
    for i in range(0, n2):
        for j in range(0, n2):
            output[i+n1, j+n1] = m2[i, j]
    return output


def inner(v, w):
    t = 0
    for i in range(0, size(v) >> 1):
        t += v[2*i]*w[2*i+1]
        t += w[2*i]*v[2*i+1]
    return t % 2


def transvection(k, v):
    return (v+inner(k, v)*k) % 2


def int2bits(i, n):
    # i = int(i)
    output = zeros(n, dtype=int32)
    for j in range(0, n):
        output[j] = i & 1
        i >>= 1
    return output


def findtransvection(x, y):
    output = zeros((2, size(x)), dtype=int32)
    if array_equal(x, y):
        return output
    if inner(x, y) == 1:
        output[0] = (x+y) % 2
        return output
    z = zeros(size(x))
    for i in range(0, size(x) >> 1):
        ii = 2*i
        if ((x[ii]+x[ii + 1]) != 0) and ((y[ii]+y[ii + 1]) != 0):
            z[ii] = (x[ii]+y[ii]) % 2
            z[ii + 1] = (x[ii + 1]+y[ii + 1]) % 2
            if (z[ii] + z[ii + 1]) == 0:  # they were the same so they added t o 00
                z[ii + 1] = 1
                if x[ii] != x[ii + 1]:
                    z[ii] = 1
            output[0] = (x+z) % 2
            output[1] = (y+z) % 2
            return output
    for i in range(0, size(x) >> 1):
        ii = 2 * i
        if ((x[ii]+x[ii + 1]) != 0) and ((y[ii]+y[ii + 1]) == 0):  # found the p ai r
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break
    for i in range(0, size(x) >> 1):
        ii = 2 * i
        if ((x[ii]+x[ii + 1]) == 0) and ((y[ii]+y[ii + 1]) != 0):  # found the p ai r
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break
    output[0] = (x+z) % 2
    output[1] = (y+z) % 2
    return output


def symplectic(i, n):
    nn = 2*n
    s = ((1 << nn)-1)
    k = (i % s)+1
    i //= s  # i/=s
    # i = int(round(i))
    f1 = int2bits(k, nn)
    e1 = zeros(nn, dtype=int32)
    e1[0] = 1
    T = findtransvection(e1, f1)
    bits = int2bits(i % (1 << (nn-1)), nn-1)
    eprime = copy(e1)
    for j in range(2, nn):
        eprime[j] = bits[j-1]
    h0 = transvection(T[0], eprime)
    h0 = transvection(T[1], h0)
    if bits[0] == 1:
        f1 *= 0
    id2 = zeros((2, 2), dtype=int32)
    id2[0, 0] = 1
    id2[1, 1] = 1
    if n != 1:
        g = directsum(id2, symplectic(i >> (nn-1), n-1))
    else:
        g = id2
    for j in range(0, nn):
        g[j] = transvection(T[0], g[j])
        g[j] = transvection(T[1], g[j])
        g[j] = transvection(h0, g[j])
        g[j] = transvection(f1, g[j])
    return g


def bits2int(b, nn):
    output = 0
    tmp = 1
    for j in range(nn):
        if b[j] == 1:
            output = output + tmp
        tmp = tmp*2
    return output


def numberofcosets(n):
    x = power(2, 2*n-1)*(power(2, 2*n)-1)
    return x


def numberofsymplectic(n):
    x = 1
    for j in range(1, n+1):
        x = x*numberofcosets(j)
    return x


def symplecticinverse(n, gn):
    nn = 2*n
    v = gn[0]
    w = gn[1]

    e1 = zeros(nn, dtype=int32)
    e1[0] = 1
    T = findtransvection(v, e1)

    tw = copy(w)
    tw = transvection(T[0], tw)
    tw = transvection(T[1], tw)
    b = tw[0]
    h0 = zeros(nn, dtype=int32)
    h0[0] = 1
    h0[1] = 0
    for j in range(2, nn):
        h0[j] = tw[j]

    bb = zeros(nn-1, dtype=int32)
    bb[0] = b
    for j in range(2, nn):
        bb[j-1] = tw[j]
    zv = bits2int(v, nn) - 1
    zw = bits2int(bb, nn-1)
    cvw = zw*(power(2, 2*n) - 1)+zv

    if n == 1:
        return cvw

    gprime = copy(gn)
    if b == 0:
        for j in range(nn):
            gprime[j] = transvection(T[1], transvection(T[0], gn[j]))
            gprime[j] = transvection(h0, gprime[j])
            gprime[j] = transvection(e1, gprime[j])
    else:
        for j in range(nn):
            gprime[j] = transvection(T[1], transvection(T[0], gn[j]))
            gprime[j] = transvection(h0, gprime[j])

    gnew = gprime[2:nn, 2:nn]
    gnidx = symplecticinverse(n-1, gnew)*numberofcosets(n)+cvw
    return gnidx


###################
###################
###################


_traf_mat = {}


def _get_traf_mat(n: int):
    t = zeros((2*n, 2*n), dtype=int32)
    for i in range(n):
        t[i, 2*i] = t[-i-1, -2*i-1] = 1
    return t


def symplectic_matrix(i: int, n: int):
    if i == -1:
        pr = 1
        for j in range(n):
            pr *= 4**(j+1) - 1
        i = randint(0, 2**(n**2)*pr - 1)
    global _traf_mat
    if n not in _traf_mat.keys():
        _traf_mat[n] = _get_traf_mat(n)
    return _traf_mat[n]@symplectic(i, n)@_traf_mat[n].transpose()


def symplectic_matrix_inverse(gn: ndarray, n: int):
    global _traf_mat
    if n not in _traf_mat.keys():
        _traf_mat[n] = _get_traf_mat(n)
    return symplecticinverse(n, _traf_mat[n].transpose()@gn@_traf_mat[n])


def is_symplectic(arr: ndarray):
    n = arr.shape[0]//2
    O = zeros((2*n, 2*n), dtype=int32)
    O[:n, n:2*n] = O[n:2*n, :n] = eye(n, dtype=int32)
    return all((arr.transpose()@O@arr) % 2 == O)


def expand_symplectic(arr: ndarray, b: int, a: int):
    c = arr.shape[0]//2
    new_d = c + b + a
    new_arr = eye(2*new_d, dtype=int32)
    new_arr[b: c+b, b: c+b] = arr[:c, :c]
    new_arr[b+new_d: c+b+new_d, b: c+b] = arr[c:2*c, :c]
    new_arr[b: c+b, b+new_d: c+b+new_d] = arr[:c, c:2*c]
    new_arr[b+new_d: c+b+new_d, b+new_d: c+b+new_d] = arr[c:2*c, c:2*c]
    return new_arr
