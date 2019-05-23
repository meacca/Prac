import numpy as np
import gf


class BCH:
    primpolies = [0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179,
                  8219, 16427, 32813, 78045]
    def __init__(self, n, t):
        assert t <= int((n - 1) / 2)
        q = int(np.log2(n + 1))
        assert (q >= 2) and (q <= 16)
        assert 2 ** q - 1 == n
        self.pm = gf.gen_pow_matrix(self.primpolies[q])
        self.R = self.pm[:2 * t, 1]
        self.g, _ = gf.minpoly(self.R, self.pm)
        check_poly = np.zeros(n + 1, dtype=np.int64)
        check_poly[0] = 1
        check_poly[-1] = 1
        assert gf.polydivmod(check_poly, self.g, self.pm)[1] == 0
        mask = (self.g == 0) | (self.g == 1)
        assert mask.all()

    def encode(self, U):
        n = self.pm.shape[0]
        m = self.g.shape[0] - 1
        x = np.zeros(m + 1, dtype=np.int64)
        x[0] = 1
        res = np.zeros((U.shape[0], U.shape[1] + m), dtype=np.int64)
        for i in range(U.shape[0]):
            u = U[i]
            code = gf.polyprod(x, u, self.pm)
            _, mod = gf.polydivmod(code, self.g, self.pm)
            code = gf.polyadd(code, mod)
            len_code = code.shape[0]
            res[i][-len_code:] = code
        return res

    def decode(self, W, method='euclid'):
        assert method == 'euclid' or method == 'pgz'
        t = self.R.shape[0] // 2
        n = W.shape[1]
        is_nan = False
        assert n == self.pm.shape[0]
        res = np.zeros_like(W, dtype=object)
        for i in range(W.shape[0]):
            w = W[i]
            s = gf.polyval(w, self.R, self.pm)
            if (s == 0).all():
                res[i] = w
                continue
            if method == 'euclid':
                s = s[::-1]
                z = np.zeros(2*t + 2, dtype=np.int64)
                z[0] = 1
                s = np.concatenate((s, np.array([1])))
                r, a, lam = gf.euclid(z, s, self.pm, max_deg=t)
            else:
                lam = np.nan
                for errors in range(t, 0, -1):
                    A = [[s[k] for k in range(j, j + errors)] for j in range(errors)]
                    A = np.array(A)
                    b = [s[k] for k in range(errors, errors * 2)]
                    b = np.array(b)
                    lam = gf.linsolve(A, b, self.pm)
                    if lam is not np.nan:
                        break
                if lam is np.nan:
                    res[i] = np.nan
                    is_nan = True
                    continue
                lam = np.concatenate((lam, np.array([1])))    
            values = gf.polyval(lam, self.pm[:, 1], self.pm)
            num_roots = 0
            #res[i] = w
            for j in range(values.shape[0]):
                if values[j] == 0:
                    root = self.pm[j, 1]
                    alpha = gf.divide(1, root, self.pm)
                    index = self.pm[alpha - 1, 0]
                    w[n - index - 1] = 1 - w[n - index - 1]
                    num_roots += 1
            if num_roots != lam.shape[0] - 1:
                res[i] = np.nan
                is_nan = True
                continue
            res[i] = w
        if not is_nan:
            res = res.astype(np.int64)
        return res
    

    def dist(self):
        n = self.pm.shape[0]
        m = self.g.shape[0] - 1
        k = n - m
        res = n
        for i in range(1, 2 ** k):
            word = np.array([int(j) for j in bin(i)[2:]])
            code = self.encode(np.array(word)[np.newaxis, :]).ravel()
            res = min(res, np.count_nonzero(code))
        t = self.R.shape[0] // 2
        assert res >= (2*t + 1)
        return res 
