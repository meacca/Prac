import numpy as np


def gen_pow_matrix(primpoly):
    deg = len(bin(primpoly)) - 3
    max_dec = 2 ** deg
    solve_poly = primpoly % max_dec
    degs = np.zeros(max_dec - 1, dtype=int)
    cur_poly = 2   # cur_poly = x
    degs[0] = cur_poly
    for i in range(1, max_dec - 1):
        res = cur_poly * 2  # mul on x == shift left on 1 bit
        if res >= max_dec:
            res = res % max_dec
            res = res ^ solve_poly
        cur_poly = res
        degs[i] = cur_poly
    pos, values = np.unique(degs, return_index=True)
    matrix = np.zeros((max_dec - 1, 2), dtype=int)
    matrix[pos - 1, 0] = values + 1
    matrix[:, 1] = degs
    return matrix


def add(X, Y):
    return X ^ Y


def sum(X, axis=0):
    return np.bitwise_xor.reduce(X, axis=axis)


def prod(X, Y, pm):
    def prod_one(x, y):
        if x == 0 or y == 0:
            return 0
        x_pow = pm[x - 1, 0]
        y_pow = pm[y - 1, 0]
        res = (x_pow + y_pow) % pm.shape[0]
        return pm[res - 1, 1]
    return np.vectorize(prod_one)(X, Y)


def divide(X, Y, pm):
    def divide_one(x, y):
        assert y != 0
        if x == 0:
            return 0
        x_pow = pm[x - 1, 0]
        y_pow = pm[y - 1, 0]
        res = (x_pow - y_pow) % pm.shape[0]
        return pm[res - 1, 1]
    return np.vectorize(divide_one)(X, Y)


def linsolve(A, b, pm):
    matrix = np.hstack((A, b[:, np.newaxis]))
    for i in range(A.shape[0] - 1):
        matrix_part = matrix[i:, i:]
        if (np.all(matrix_part[:, 0] == 0)):
            return np.nan
        if matrix_part[0, 0] == 0:
            non_zero = np.where(matrix_part[:, 0] != 0)[0][0]
            matrix_part[0], matrix_part[non_zero] = matrix_part[non_zero], matrix_part[0].copy()
        arr1 = matrix_part[0][np.newaxis, :]
        arr2 = matrix_part[1:, 0][:, np.newaxis]
        mul = prod(arr1, divide(arr2, matrix_part[0][0], pm), pm)
        matrix_part[1:] = add(matrix_part[1:], mul)
    if matrix[-1, -2] == 0:
        return np.nan
    res = np.zeros(A.shape[0], dtype=np.int64)
    for i in range(A.shape[0], 0, -1):
        matrix_part = matrix[:i, :i + 1]
        numer = matrix_part[-1, -1]
        denom = matrix_part[-1, -2]
        root = divide(numer, denom, pm)
        res[i - 1] = root
        matrix_part[:, -2] = add(matrix_part[:, -1], prod(matrix_part[:, -2],
                                 root, pm))
    return res


def minpoly(x, pm):
    roots = set()
    for elem in x:
        cycle = set()
        cycle.add(elem)
        new = int(prod(elem, elem, pm))
        while new not in cycle:
            cycle.add(new)
            new = int(prod(new, new, pm))
        roots.update(cycle)
    roots = np.array(list(roots))
    roots = np.sort(roots)
    res = np.array([1, roots[0]])
    for i, root in enumerate(roots[1:]):
        res = polyprod(res, np.array([1, root]), pm)
    return res, roots

def polyval(p, x, pm):
    res = np.zeros_like(x)
    cur = 1
    for i, koef in enumerate(list(p)[::-1]):
        value = prod(cur, koef, pm)
        res = res ^ value
        cur = prod(cur, x, pm)
    return res

def normalize_poly(p):
    for i in range(p.shape[0]):
        if p[i] != 0:
            break
    return p[i:]

def polyprod(p1, p2, pm):
    p1_pow = pm[p1 - 1, 0]
    p2_pow = pm[p2 - 1, 0]
    res = np.zeros(p1.shape[0] + p2.shape[0] - 1, dtype=np.int64)
    for i1, pow1 in enumerate(p1_pow[::-1]):
        for i2, pow2 in enumerate(p2_pow[::-1]):
            if p1[::-1][i1] == 0 or p2[::-1][i2] == 0:
                continue
            res_pow = (pow1 + pow2) % pm.shape[0]
            res[i1 + i2] = res[i1 + i2] ^ pm[:, 1][res_pow - 1]
    return normalize_poly(res[::-1])


def polydivmod(p1, p2, pm):
    p1 = normalize_poly(p1)
    p2 = normalize_poly(p2)
    max_deg = p2.shape[0] - 1
    cur_poly = p1
    cur_deg = p1.shape[0] - 1
    if cur_deg < max_deg:
        return np.array([0]), p1
    res = np.zeros(cur_deg - max_deg + 1, dtype = np.int64)
    while cur_deg >= max_deg:
        koef = int(divide(cur_poly[0], p2[0], pm))
        res[cur_deg - max_deg] = koef
        tmp_poly = np.zeros(cur_deg - max_deg + 1, dtype = np.int64)
        tmp_poly[0] = koef
        tmp_poly = polyprod(tmp_poly, p2, pm)
        cur_poly = cur_poly ^ tmp_poly
        flag = True
        for i in range(cur_poly.shape[0]):
            if cur_poly[i] != 0:
                index = i
                flag = False
                break
        if flag:
            break
        cur_poly = cur_poly[index:]
        cur_deg = cur_poly.shape[0] - 1
    return normalize_poly(res[::-1]), normalize_poly(cur_poly)

def polyadd(p1, p2):
    if len(p1) > len(p2):
        p1, p2 = p2, p1.copy()
    p1 = np.concatenate([np.zeros(len(p2) - len(p1), dtype=np.int64), p1])
    res = p1 ^ p2
    return normalize_poly(res)

def euclid(p1, p2, pm, max_deg=0):
    p1_deg = p1.shape[0] - 1
    p2_deg = p2.shape[0] - 1
    if p1_deg < p2_deg:
        p1, p2 = p2, p1.copy()
        p1_deg, p2_deg = p2_deg, p1_deg
    r_old = p1
    r_new = p2
    x_old = np.array([1])
    x_new = np.array([0])
    y_old = np.array([0])
    y_new = np.array([1])
    deg_new = p2_deg
    while deg_new > max_deg:
        q, r = polydivmod(r_old, r_new, pm)
        r_old = r_new
        r_new = r
        deg_new = r_new.shape[0] - 1
        x = polyadd(x_old, polyprod(q, x_new, pm))
        x_old = x_new
        x_new = x
        y = polyadd(y_old, polyprod(q, y_new, pm))
        y_old = y_new
        y_new = y
    r_new = normalize_poly(r_new)
    x_new = normalize_poly(x_new)
    y_new = normalize_poly(y_new)
    return r_new, x_new, y_new
