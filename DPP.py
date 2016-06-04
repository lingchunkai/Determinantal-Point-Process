"""
Copyright (c) 2016 Ling Chun Kai


Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Code snippet provides rudimentary (brute force) sampling of a L-ensemble [1]. Also provides approximate MAP inference up to a 1/4 approximation [2]. Requires the usual numpy, scipy, matplotlib stack.

[1] Kulesza, Alex, and Ben Taskar. "Determinantal point processes for machine learning." arXiv preprint arXiv:1207.6083 (2012).
[2] Gillenwater, Jennifer, Alex Kulesza, and Ben Taskar. "Near-optimal MAP inference for determinantal point processes." Advances in Neural Information Processing Systems. 2012.
"""

import math
import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

class DPP:
    
    def __init__(self, L):
        self.L = L
        self.eVal=None
        self.eVec=None

    def ComputeEigenDecomposition(self):
        self.eVal, self.eVec=scipy.linalg.eigh(self.L)
        idx=self.eVal.argsort()[::-1]
        self.eVal=self.eVal[idx]
        self.eVec=self.eVec[:,idx]

    def Sample(self):
        # Sorted eigendecomposition
        if self.eVal==None or self.eVec==None:
            print "eigendecomposition not cached, computing..."
            self.ComputeEigenDecomposition() 
      
        eVal = self.eVal
        eVec = self.eVec
            
        ## Sample from binomial distributions 
        peV=[eV/(eV+1) if eV>0 else 0 for eV in eVal]
        drawneVals=np.random.binomial(1,p=peV)

        eBasis = eVec[:, [x for x in xrange(len(drawneVals)) if drawneVals[x]==1]]

        print "E[|V|]=", sum(peV)
        print "|V|=", eBasis.shape[1]
    
        ## Iteratively draw vectors
        chosen = []
        while eBasis.shape[1] > 0:
            probs = np.sum(np.square(eBasis.T), 0)/eBasis.shape[1]
            
            # elementary vector chosen
            elem_chosen=np.random.multinomial(1, probs, size=1)
            chosen.append(np.squeeze(elem_chosen))
            
            if eBasis.shape[1] == 1: break
            
            # Projection of chosen element onto existing subspace
            proj = np.sum(np.diag(np.squeeze(eBasis.T.dot(elem_chosen.T))).dot(eBasis.T), 0)
            proj = np.expand_dims(proj/np.linalg.norm(proj), 1)  
            
            # Find orthogonal basis of subspace - projection
            residual = np.diag(np.squeeze(eBasis.T.dot(proj))).dot(np.tile(proj.T, (eBasis.shape[1],1)))
            eBasis = scipy.linalg.orth(eBasis-residual.T)
        
        return sum(chosen) 

    def ExpectedCardinality(self):
        if self.eVal==None or self.eVec==None:
            self.ComputeEigenDecomposition()           

        return sum([eV/(eV+1) for eV in self.eVal])

    def GetMAP(self):
        """
        Find MAP in O(N^3), up to convergence rate
        Polytope assumed to be [0,1]^N
        """        
        x = self.LocalOpt(np.ones([self.L.shape[0],1]))
        y = self.LocalOpt(optConstraints=np.reshape(np.rint(np.ones(x.shape)-x),-1)[...,np.newaxis])
        if self.SoftMax(x) > self.SoftMax(y): return x
        else: return y

    def LocalOpt(self, optConstraints, epsilon = 10**-5, max_it = 150): 
        """
        @param optConstraints: additonal optional inequality constraint (upper bound on x). Set to ones if we want restriction to hypercube
        """
        x = 0*np.ones([self.L.shape[0], 1])
        x = np.squeeze(x)
        for n in xrange(max_it):
            y = scipy.optimize.linprog(-np.squeeze(self.GradSoftMax(x)), np.concatenate((-np.eye(self.L.shape[0]), np.eye(self.L.shape[0])), axis=0), np.concatenate((np.zeros([self.L.shape[0], 1]), optConstraints )))
            y=y.x
            alpha, val, d = scipy.optimize.fmin_l_bfgs_b(lambda q: -self.SoftMax(q * x + (1.-q) * y), 0.5, approx_grad=True, fprime = lambda r: np.array(self.GradSoftMax(r*x+(1-r)*y).T.dot(x-y)), bounds = [(0.,1.)]) # TODO: Use proper gradient for optimization with l_bfgs
            x = alpha * x + (1-alpha) * y
            if np.all(np.abs(x-np.rint(x)) < epsilon): # Convergence
                break
        return x

    def SoftMax(self,x):
        """
        Compute F-tilde
        """
        return math.log(scipy.linalg.det(np.diag(np.squeeze(x)).dot(self.L-np.eye(x.size)) + np.eye(x.size)))

    def GradSoftMax(self,x):
        inv = scipy.linalg.inv(np.diag(np.squeeze(x)).dot(self.L-np.eye(x.size)) + np.eye(x.size)) # TODO: change to avoid explicit computation of inverse
        LmI = self.L - np.eye(self.L.shape[0])
        ret = np.zeros(x.shape)
        for k in xrange(x.size):
            ret[k] = LmI[k, :].dot(inv[:, k])
        return ret
 

if __name__ == "__main__":
    SIZE_GRID=15
    NUMELS=SIZE_GRID*SIZE_GRID
    SIGMA=25
    # Kernel defined to be non-circular, SE (RBF) 
    # SE kernel
    L=lambda a,b,sigma: 10*math.exp(-0.5*(np.linalg.norm(a-b)/sigma)**2)
    # Explicitly construct L matrix by iterating pairwise
    Ygrid,Xgrid=np.mgrid[0:SIZE_GRID,0:SIZE_GRID]
    points=np.concatenate((np.reshape(Ygrid, -1)[...,np.newaxis], np.reshape(Xgrid, -1)[...,np.newaxis]), 1)
    Lmat=np.zeros((points.shape[0],points.shape[0]))
    for i in xrange(points.shape[0]):
        for j in xrange(points.shape[0]):
            Lmat[i,j]=L(i,j,SIGMA)

    dpp = DPP(Lmat)

    dpp_sample_plot = plt.figure('Sampled from DPP')
    for k in xrange(9):
        ax = dpp_sample_plot.add_subplot(3,3,k+1)
        scatterplot = dpp.Sample()
        scatterplot = np.reshape(scatterplot, [SIZE_GRID,SIZE_GRID])
        ax.imshow(scatterplot, cmap='Greys',  interpolation='nearest')
    nExpectedCardinality = dpp.ExpectedCardinality()
    
    iid_sample_plot = plt.figure('Sampled independently')
    for k in xrange(9):
        ax = iid_sample_plot.add_subplot(3,3,k+1)
        scatterplot=np.random.binomial(1,p=[nExpectedCardinality/NUMELS]*NUMELS)
        scatterplot = np.reshape(scatterplot, [SIZE_GRID,SIZE_GRID])
        ax.imshow(scatterplot, cmap='Greys',  interpolation='nearest')

    dpp_MAP = dpp.GetMAP()
    dpp_MAP = np.rint(np.reshape(dpp_MAP, [SIZE_GRID, SIZE_GRID]))
    dpp_MAP_plot = plt.figure('Estimated MAP')
    ax = dpp_MAP_plot.add_subplot(111);
    ax.imshow(dpp_MAP, cmap='Greys', interpolation='nearest')
      
    plt.show()
