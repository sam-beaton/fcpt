import skfda
import numpy as np
import pandas as pd
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
import sys

from pathlib import Path

# Root of this project (the folder containing this notebook, or an ancestor of
# it, named 'FDA'). Walks up from the current working directory until it finds
# one. (Same pattern used throughout this project's notebooks.)
cwd = Path().resolve()
if cwd.name == 'FDA':
    projectRoot = cwd
else:
    for parent in cwd.parents:
        if parent.name == 'FDA':
            projectRoot = parent
            break
    else:
        raise FileNotFoundError("Could not find FDA project root")

sys.path.append(str(projectRoot / 'functions'))
import auxFuncChPt as auxF

##########################################################################################################################################

def changeFPCA(fdObj, dHat=0, numBootstrap=10000, h=0, plot=False, longRun=True, wildBinSeg=False, numIntervals=500, *args, **kwargs):
    
    #check fd input object is in the correct form
    if not(isinstance(fdObj, skfda.representation.basis._fdatabasis.FDataBasis)
           or isinstance(fdObj, skfda.representation.grid.FDataGrid)):
        #return error if not
        raise ValueError("Must use functional data in basis representation or grid form")
    
    if isinstance(fdObj, skfda.representation.basis._fdatabasis.FDataBasis):
        N = fdObj.coefficients[:, 0].shape[0] # First dimension = rows = number of functional objects
        D = fdObj.coefficients[0].shape[0] # Second dimension = columns = number of basis functions
    else: 
        #if FDataGrid
        N = fdObj.data_matrix.shape[0] # First dimension = rows = number of functional objects
        #num fPCs must be less than size of dataset and of number of sample points
        D = np.min([(N-1), (fdObj.data_matrix.shape[1])]) #set to min of these two values

    #set up PCA with full number of components
    #fPCA = FPCA(n_components=D) #centering = TRUE applied as default
    #find variance for each fPC
    #fPCAScores = fPCA.fit_transform(fdObj) #requires scores/comps to first be calculated

    # calculate fPCA  
    # check if specific number of PCs desired in reconstruction; find them using threshold (default 85%) if not
    if not(dHat):
        dHat = auxF.cumulPercVariance(fdObj)

    dHat = min(N, dHat)

     #set up PCA with required number of components
    fPCA = FPCA(n_components=dHat) #centering = TRUE applied as default
    #find variance for each fPC
    fPCAScores = fPCA.fit_transform(fdObj) #requires scores/comps to first be calculated

    # Auto-select h if not provided
    if longRun == True:
        if h == 0: #if lag not provided
            h = int(4 * (N / 100) ** (2 / 9))  # calculate using Newey-West rule, via. Andrews (1991)
        h = np.min([h, N-1]) #adjust lag if not enough functional data in this recursive iteration of binseg
    else: 
        h = 0 #just in case non-zero lag given when not using the L-R covariance

    #calculate (long-run) covariance
    sigmaHat = auxF.calcLRCovGrid(fPCAScores, d = dHat, h = h) 
    #invert it for partial sum calculation
    invSigmaHat = np.linalg.inv(sigmaHat).astype('float64')

    if wildBinSeg:
        # Wild Binary Segmentation approach
        intervals = auxF.generateRandomIntervals(N, numIntervals)
        maxT_NAll = []
        kStarAll = []
        
        for startIdx, endIdx in intervals:
            intervalLength = endIdx - startIdx
            T_N = np.empty(intervalLength)
            
            for k in range(intervalLength):
                L_Nk = auxF.computeIntervalPartialSumFPCA(k, fPCAScores, dHat, startIdx, endIdx)
                T_N[k] = (1/intervalLength) * (np.dot(np.transpose(L_Nk), (np.dot(invSigmaHat, L_Nk))))

            maxT_N = np.max(np.abs(T_N))
            kStar_interval = (np.where(np.abs(T_N) == maxT_N))[0][0]
            maxT_NAll.append((maxT_N, kStar_interval + startIdx))
        
        # Find interval with maximum test statistic
        maxT_N, kStar = max(maxT_NAll, key=lambda x: x[0])
        
    else:
        # Original approach
        T_N = np.empty(N)
        for k in range(N):
            L_Nk = auxF.computeIntervalPartialSumFPCA(k, fPCAScores, dHat)
            T_N[k] = (1/N) * (np.dot(np.transpose(L_Nk), (np.dot(invSigmaHat, L_Nk))))
        
        maxT_N = np.max(np.abs(T_N))
        kStar = (np.where(T_N == maxT_N))[0][0]

    # Bootstrap and significance calculation
    bridgeVals = auxF.brownianBridgeBootstrap(dHat, N, numPerms=numBootstrap)
    z = np.sum((maxT_N <= bridgeVals) == True)
    pValue = z / (bridgeVals.shape)[0]

    # Calculate means
    preChangeFuncs = (fdObj[:kStar])
    preChangeMean = skfda.exploratory.stats.mean(preChangeFuncs)
    postChangeFuncs = (fdObj[kStar:])
    postChangeMean = skfda.exploratory.stats.mean(postChangeFuncs)
    deltaMeans = preChangeMean - postChangeMean

    return kStar, pValue, maxT_N, preChangeMean, postChangeMean, deltaMeans, dHat

##########################################################################################################################################

def changeFF(fdObj, numBootstrap=10000, h=0, firstSamp=-20, lastSamp=182, numComponents=15, plot=False, longRun=True, wildBinSeg=False, numBinSegIntervals=500, *args, **kwargs):
    # Original input checks and setup remain the same
    if not(isinstance(fdObj, skfda.representation.basis._fdatabasis.FDataBasis)
           or isinstance(fdObj, skfda.representation.grid.FDataGrid)):
        raise ValueError("Must use functional data in basis representation or grid form")
    
    if isinstance(fdObj, skfda.representation.grid.FDataGrid):
        basis = BSplineBasis(n_basis=numComponents)
        fdObj = fdObj.to_basis(basis)
    
    coefs = fdObj.coefficients.astype('float64')
    N = coefs[:, 0].shape[0]
    D = coefs[0].shape[0]
    
    if wildBinSeg:
        intervals = auxF.generateRandomIntervals(N, numBinSegIntervals)
        allMaxPartialSum = []
        kStarAll = []
        
        for startIdx, endIdx in intervals:
            S_N = auxF.computeIntervalPartialSumFF(coefs, startIdx, endIdx)
            maxS_N = np.max(np.abs(S_N))
            kStar = (np.where(S_N == maxS_N))[0][0]
            allMaxPartialSum.append((maxS_N, kStar + startIdx, endIdx - startIdx))  
            
        maxS_N, kStar, segSize = max(allMaxPartialSum, key=lambda x: x[0])
        
    else:
        S_N = np.zeros([N])
        for i in range(2, N):
            S_N[i-1] = (np.sum((np.sum(coefs[0:i, :], axis=0) - 
                              (i/N) * np.sum(coefs, axis=0))**2)) / N
            
        maxS_N = np.max(np.abs(S_N))
        kStar = (np.where(np.abs(S_N) == maxS_N))[0][0]
        segSize = N  
    
    if longRun:
        if h == 0:
            h = int(4 * (N / 100) ** (2 / 9))
        h = np.min([h, N-1])
    else:
        h = 0
    
    covMat = auxF.calcLRCovBasis(fdObj, h=h)
    eigCovMat = np.linalg.eigh(covMat)
    eigValsCV = eigCovMat.eigenvalues
    basis = BSplineBasis(n_basis=D)
    
    bridgeVals = auxF.eigValBridgeBootstrap(D, segSize, eigValsCV, numPerms=numBootstrap)  # adjusted for segment size - important!
    
    z = np.sum((maxS_N <= bridgeVals) == True)
    pValue = z / (bridgeVals.shape)[0]
    
    preChangeFuncs = (fdObj[:kStar])
    preChangeMean = np.mean(preChangeFuncs)
    postChangeFuncs = (fdObj[kStar:])
    postChangeMean = np.mean(postChangeFuncs)
    deltaMeans = preChangeMean - postChangeMean
    
    return kStar, pValue, S_N, maxS_N, preChangeMean, postChangeMean, deltaMeans
