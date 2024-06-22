import os
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy import sparse
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser


def get_segs(winding, params, slope):
    segs = []
    
    breakpts = [0]+list(params)+[len(winding)]
    for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):
        a = int(a)
        b = int(b)
        seg = np.array(winding[a:b])
        if ii%2:
            try:
                seg -= slope*np.arange(a, b)
            except:
                print(a, b, params, slope)
                print(seg)
                print((slope*np.arange(a, b)))
                raise Exception()
        seg -= np.mean(seg)
        segs.append(seg)

    return segs

#loss function for 4-breakpoint regression
def loss_multi(winding, params, slope, penalties):
    segs = get_segs(winding, params,slope)
    return np.sum([penalties[ii%2]*np.sum(seg**2) for ii,seg in enumerate(segs)])

# regression with 4 breakpoints
def multi_regression(preX, l, r):
    winding, s, c, q, dx = get_winding(preX)

    m, scores = median_slope(winding, 150, 250)
    pre, mid, post = get_premidpost(winding, (l, r), m)

    
    start = np.where(np.diff(np.sign(pre)))[0][-1]
    if preX.shape[0] - r:
        end = r+np.where(np.diff(np.sign(post)))[0][0]
    else:
        end = preX.shape[0]
    m, scores = median_slope(winding[start:end], 20, 30)

    n = len(winding)
    l = n // 2
    r = (3 * n) // 4
    parameters = np.array([l,l+(r-l)/3,l+2*(r-l)/3, r ])

    penalties = [1, 1.5]
    epsilon = 0.01
    gradient = np.zeros(4)
    delta = [*np.identity(4)]
    prev_grad = np.array(gradient)
    thresh = .3

    for i in range(10000):
        present = loss_multi(winding, parameters, m, penalties)
        if np.linalg.norm(gradient - prev_grad)< thresh and i > 0:
            break
        gradient = np.array([loss_multi(winding, parameters + d, m, penalties) - present for d in delta])
        parameters = parameters - epsilon * gradient
    return winding, m, parameters

#plot regression with 4 breakpoints
def plot_regression_multi(winding, params, slope, save = False, filename = ''):
    segs = get_segs(winding, params, slope)

    plt.plot(winding)
    breakpts = [0]+list(params)+[len(winding)]
    for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):    
        a = int(a)
        b = int(b)        
        g = winding[a:b]
        plt.plot(range(a, b), g - segs[ii])
    if save:
        plt.savefig(filename + '.png')
        plt.close()
    else:
        plt.show()

#return winding and swl2 from two eigenvectors
def get_winding_swl2(s, c):
    ds = gaussian_filter(s, 1, order = 1)
    dc = gaussian_filter(c, 1, order = 1)
    r2 = s ** 2 + c ** 2
    summand = (c * ds - s * dc) / r2    
    winding = np.cumsum(summand) / (2 * np.pi)
    winding *= np.sign(winding[-1] - winding[0])

    N = len(winding)
    slope = mode_slope(winding, 20, 30)
    # slope = median_slope(winding, 20, 30)[0]

    M = 10
    swl2 = []
    for ii in range(len(winding) - M-1):
        res = winding[ii:ii+M]-slope*np.arange(ii,ii+M)
        res -= np.mean(res)
        swl2.append(np.linalg.norm(res))

    return winding, swl2, slope

#remaining functions written by Chris for graph laplacian calculation

def get_curv_vectors(X, MaxOrder, sigma, loop = False, m = 'nearest'):

    from scipy.ndimage import gaussian_filter1d as gf1d
    if loop:
        m = 'wrap'
    XSmooth = gf1d(X, sigma, axis=0, order = 0, mode = m)
    Vel = gf1d(X, sigma, axis=0, order = 1, mode = m)
    VelNorm = np.sqrt(np.sum(Vel**2, 1))
    VelNorm[VelNorm == 0] = 1
    Curvs = [XSmooth, Vel]
    for order in range(2, MaxOrder+1):
        Tors = gf1d(X, sigma, axis=0, order=order, mode = m)
        for j in range(1, order):
            #Project away other components
            NormsDenom = np.sum(Curvs[j]**2, 1)
            NormsDenom[NormsDenom == 0] = 1
            Norms = np.sum(Tors*Curvs[j], 1)/NormsDenom
            Tors = Tors - Curvs[j]*Norms[:, None]
        Tors = Tors/(VelNorm[:, None]**order)
        Curvs.append(Tors)
    return Curvs


def sliding_window(dist, win):
    N = dist.shape[0]
    dist_stack = np.zeros((N-win+1, N-win+1))
    for i in range(0, win):
        dist_stack += dist[i:i+N-win+1, i:i+N-win+1]
    for i in range(N-win+1):
        dist_stack[i, i] = 0
    return dist_stack

def get_csm(X, Y):
    if len(X.shape) == 1:
        X = X[:, None]
    if len(Y.shape) == 1:
        Y = Y[:, None]
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)


def csm_to_binary(D, kappa):
    N = D.shape[0]
    M = D.shape[1]
    if kappa == 0:
        return np.ones_like(D)
    elif kappa < 1:
        NNeighbs = int(np.round(kappa*M))
    else:
        NNeighbs = kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M), dtype=np.uint8)
    return ret.toarray()

def csm_to_binary_mutual(D, kappa):
    return csm_to_binary(D, kappa)*(csm_to_binary(D.T, kappa).T)

def getUnweightedLaplacianEigsDense(W):
    D = sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = linalg.eigh(L)
    except:
        return np.zeros_like(W)
    return v

