import numpy as np
import copy

def gen_pow_matrix(primpoly):
    int_part = copy.copy(primpoly)
    q=0
    while int_part > 0:
        int_part //= 2
        q += 1
    q = q - 1
    pm = np.zeros(shape=(2**q-1, 2), dtype=int)
    
    primary_elem = 2
    current_elem = 2
    oldest_digit = 2**q
    oldest_digit_change = primpoly - oldest_digit
        
    list_values_elems_in_F = [2]
    pm[current_elem-1] = 1
    
    for i in range(1, 2**q-1):
        current_elem = current_elem * primary_elem

        if current_elem >= oldest_digit:
            current_elem -= oldest_digit
            current_elem ^= oldest_digit_change
        list_values_elems_in_F.append(current_elem)
        
        
        pm[current_elem-1, 0] = i+1
    
    pm[:,1] = np.array(list_values_elems_in_F)
    
    return pm

def add(X, Y):
    return np.bitwise_xor(X, Y)
    
def sum(X, axis=0):
    res = np.bitwise_xor.reduce(X, axis=axis)
    return res


def prod(X, Y, pm):
    degrees = (pm[:,0][X-1] + pm[:,0][Y-1] - 1) % pm.shape[0]
    nil_mask = ((X==0) | (Y==0))
    res = pm[:,1][degrees]
    if X.shape == Y.shape == (): #problems with res[nil_mask] = 0
        if X==0 or Y==0:
            return 0
        else:
            return res
    res[nil_mask] = 0
    return res
    
   
    

def divide(X, Y, pm):
    degrees = (pm[:,0][X-1] - pm[:,0][Y-1] - 1) % pm.shape[0]
    return pm[:,1][degrees]

def extend_poly(pol, required_len):
    assert required_len > len(pol)
    ext_pol =  np.zeros((required_len,), dtype=int)
    distinguish = required_len - len(pol)
    ext_pol[distinguish:] = pol
    return ext_pol
    
def polyprod(p1, p2, pm):
    distinguish = np.abs(len(p1) - len(p2))
    if len(p1) > len(p2):
        p2 = extend_poly(p2, len(p1))
    elif len(p2) > len(p1):
        p1 = extend_poly(p1, len(p2))
    
    matr_pr = prod(np.tile(p1[::-1], [len(p1), 1]), 
                   np.tile(p2, [len(p2),1]).T, pm)
    
    num_diags = len(p1)*2 - 1
    res = np.zeros(shape=(num_diags,), dtype=int)
    degree = 0
    for i in range( -(num_diags-1)//2, (num_diags-1)//2 + 1):
        res[num_diags-degree-1] = sum(np.diag(matr_pr, i))
        degree += 1
      
    return res[distinguish:]

def polyval(p, x, pm):
    x_degrees = np.zeros((x.shape[0], p.shape[0]), dtype=int)
    oldest_degree = len(p) - 1
    x_degrees[:,oldest_degree] = 1
    
    for degree in range(1, oldest_degree+1):
        x_degrees[:, oldest_degree-degree] = prod(x_degrees[:, oldest_degree-degree+1], x, pm)
    
    res = prod(p, x_degrees, pm)
    return sum(res, axis=1)
        
        
def minpoly(x, pm):
    all_elems_belong = list()
    min_poly = np.array([1])
    for elem in x:
        if elem in all_elems_belong:
            continue
        else:
            elem_degree = elem.copy()
            cur_poly = np.array([1])
            
            while elem_degree not in all_elems_belong:
                cur_poly = polyprod(cur_poly, np.array([1, elem_degree]), pm)
                all_elems_belong.append(elem_degree)
                elem_degree = prod(np.array(elem_degree), np.array(elem_degree), pm)
                
        min_poly = polyprod(min_poly, cur_poly, pm)
        
    return  (min_poly, np.sort(np.array(all_elems_belong)))
    
def clear_begin_zeros(poly):
    num_zeros = 0
    for elem in poly:
        if elem==0:
            num_zeros += 1
        else:
            break
    return poly[num_zeros:], num_zeros

def polydivmod(p1, p2, pm):
    dist = len(p1) - len(p2)
    res = np.zeros((dist+1,), dtype=int)
    
    while dist >= 0:
        
        shift_poly = np.zeros((dist+1,), dtype=int)
        coef = divide(p1[0], p2[0], pm)
        shift_poly[0] = coef
        p2_shift = polyprod(p2, shift_poly, pm)         
        p1 = add(p1, p2_shift)
        res[dist] = coef
        p1, num_zeros = clear_begin_zeros(p1)
        dist -= num_zeros
        
    
    return (res, clear_begin_zeros(p1)[0])
        
  

class BCH:
    def __init__(self, n, t):
        with open('primpoly.txt','r') as f:
            for line in f:
                for word in line.split():
                    prim_poly = int(word[:-1])
                    if prim_poly > n:
                        break
        
        self.pm = gen_pow_matrix(prim_poly)
        self.n = n
        
        
        need_roots = np.zeros(shape=(2*t,), dtype=int)
        prime_elem = 2
        cur_elem = 2
        for i in range(0, 2*t):
            need_roots[i] = cur_elem
            cur_elem = prod(np.array(cur_elem), np.array(prime_elem), self.pm)
        
        self.g, self.R = minpoly(need_roots, self.pm)
        self.m = len(self.g)
        self.k = self.n - self.m
    def encode(self, U):
        assert self.k == U.shape[1]
        encoded = np.zeros(shape=(U.shape[0], self.n) ,dtype=int)
        encoded[:,0:self.k] = U
        for i, u_1 in enumerate(encoded):
            rest = polydivmod(u_1, self.g, self.pm)[1]
            encoded[i,self.k:] = rest
        return encoded
        
    def dist(self):
        dist = self.n
        for num in range(1, 2**self.k):
            print(num, 2**self.k)
            poly = np.zeros(shape=(self.k,), dtype=int)
            for i, let in enumerate(np.binary_repr(num, width=self.k)):
                poly[i] = let
            
            cur_dist = polyprod(poly, self.g, pm).sum()
            print(cur_dist)
            if cur_dist < dist:
                dist = cur_dist
        return dist        
        
 
