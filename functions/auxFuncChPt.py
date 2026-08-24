import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from statsmodels.stats.multitest import multipletests # type: ignore
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

# NOTE: this file does not actually call anything via `fNIRSData.` (confirmed -
# no fNIRSData.xxx() calls anywhere in this file's body), so unlike
# fNIRSData.py's own self-import, `import fNIRSData` is not needed here.
import funcChangePoint


def calcLRCovGrid(coefsMat, d = 0, h = 0, kernelType = "Bartlett", rowVar = True):

    ## Calculate long run covariance
    # As defined in Chapter 16 of Horváth and Kokoszka (2012) (CITE)
    # Adapted from code found in fChange R package (CITE)
    #
    # Inputs:
    #       data    :       a (Nxd) matrix (np.array) of values containing either (i) d fPC 
    #                       scores or (ii) d basis coefficients from a FD basis representation
    #                       for each of the N functional objects in the dataset
    #       d       :       number of fPC scores/basis functions to use (will result in a dHat x 
    #                       dHat sized Cov. matrix))
    #       h       :       bandwidth parameter (default is 0 i.e. data is i.i.d.)
    #                       default: 0 i.e. no lag
    #       kernelType:     Choice of kernel function for calculation
    #                       Options: Flat, Simple or Bartlett
    #                       default: Bartlett
    #       rowVar  :       If rowvar is True (default), then each row represents a variable,
    #                       with observations (fPC scores) in the columns. Otherwise, the 
    #                       relationship is transposed: each column represents a variable, 
    #                       while the rows contain observations.
    #
    # Outputs:
    #       covMat  :       covariance matrix for dataset fPC scores/basis coefs; using bandwidth h
    #                       results in a long-run covariance matrix
    #
    # SLB 21/11/23
    #
    # TO DO: 
    #       find eigenfunctions and -values of cov. matrix
    #       code the covariance plot (using eigendecomposition)

    ## -------------------- Initialise and define variables -----------------
    #check fd input object is in the correct form
    if not(isinstance(coefsMat, np.ndarray)):  
        #return error if not
        raise ValueError("Must provide fPC scores/basis coefficients in (numpy) matrix form. For functional objects in basis form use 'calcLRCovBasis'")
    
    # check for valid h value
    if h < 0:
        raise ValueError("Bandwidth cannot be negative")
    
    N = coefsMat.shape[0]  # First dimension = rows = number of functional objects
    D = coefsMat.shape[1]  # Second dimension = columns = number of basis functions

    #set dimensions of Cov matrix to max if none provided
    if d == 0:
        d = D

    covMat = np.zeros((d,d))

    for k in range(0,d):     #iterates over k=1:D where D = number of basis functions
        for r in range(k, d):   #iterates over r = k:D to fill the upper triangle of D_mat
            #for each combination of k & r, calculate the inner product of the kth and rth columns
            #of the of the centred functional object
            s = (np.inner(coefsMat[:, k], coefsMat[:, r])).astype('float64')
            if h > 0:       #if a positive lag has been specified
                for i in range(0, h):       #for each of the values up to and including lag h:
                    #take the inner product of the first (N-i) elements in the kth col by the last (N-i) elements in the rth col
                    a = np.inner(coefsMat[0:(N-(i+1)), k], coefsMat[(i+1):N, r]).astype('float64')
                    #add to the inner product of the first (N-i) elements in the rth col. & the last (N-i) elements in the kth col.
                    + np.inner(coefsMat[0:(N-(i+1)), r], coefsMat[(i+1):N, k]).astype('float64')
                    #multiply by kernel and add to s
                    s = (chooseKernel(kernelType, i, h) * a).astype('float64')
            covMat[k,r] = s
            covMat[r,k] = covMat[k,r]

    covMat = (covMat / N)

    ## ---------------find eigen values and functions of LR Cov mat ------------------
    eigDecomp = np.linalg.eig(covMat)
    ## -------------- produce covariance surface plot -------------------

    return covMat

#################################################

def calcLRCovBasis(fdObj, d = 0, h = 0, kernelType = "Bartlett", rowVar = True):

    ## Calculate long run covariance
    # As defined in Chapter 16 of Horváth and Kokoszka (2012) (CITE)
    # Adapted from code found in fChange R package (CITE)
    #
    # Inputs:
    #       data    :       a (Nxd) matrix (np.array) of values containing either (i) d fPC 
    #                       scores or (ii) d basis coefficients from a FD basis representation
    #                       for each of the N functional objects in the dataset
    #       d       :       number of fPC scores/basis functions to use (will result in a dHat x 
    #                       dHat sized Cov. matrix))
    #       h       :       bandwidth parameter (default is 0 i.e. data is i.i.d.)
    #                       default: 0 i.e. no lag
    #       kernelType:     Choice of kernel function for calculation
    #                       Options: Flat, Simple or Bartlett
    #                       default: Bartlett
    #       rowVar  :       If rowvar is True (default), then each row represents a variable,
    #                       with observations (fPC scores) in the columns. Otherwise, the 
    #                       relationship is transposed: each column represents a variable, 
    #                       while the rows contain observations.
    #
    # Outputs:
    #       covMat  :       covariance matrix for dataset fPC scores/basis coefs; using bandwidth h
    #                       results in a long-run covariance matrix
    #
    # SLB 21/11/23
    #
    # TO DO: 
    #       find eigenfunctions and -values of cov. matrix
    #       code the covariance plot (using eigendecomposition)

    ## -------------------- Initialise and define variables -----------------
    #check fd input object is in the correct form
    if not(isinstance(fdObj, skfda.representation.basis._fdatabasis.FDataBasis)):  
        #return error if not
        raise ValueError("Must provide fPC scores/basis coefficients in basis form. To calculate covariance on the coefficients only use 'calcLRCovGrid'")
    
    # check for valid h value
    if h < 0:
        raise ValueError("Bandwidth cannot be negative")
    
    N = fdObj.coefficients.shape[0]  # First dimension = rows = number of functional objects
    D = fdObj.coefficients.shape[1]  # Second dimension = columns = number of basis functions

    #set dimensions of Cov matrix to max if none provided
    #only recommended to use d!=0 when using with fPCs i.e. need to select a subset of fPCs for dim reduction
    if d == 0:
        d = D

    # assign basis variable for covariance plot - remove if not going to try to plot
    basisEigen = fdObj.basis
    coefsMat = fdObj.coefficients

    coefsMat = coefsMat - np.mean(coefsMat, axis = 0)
    fdObj.coefficients = coefsMat

    covMat = np.zeros((d,d))

    for k in range(0,d):     #iterates over k=1:D where D = number of basis functions
        for r in range(k, d):   #iterates over r = k:D to fill the upper triangle of D_mat
            #for each combination of k & r, calculate the inner product of the kth and rth columns
            #of the of the centred functional object
            s = (np.inner(coefsMat[:, k], coefsMat[:, r])).astype('float64')
            if h > 0:       #if a positive lag has been specified
                for i in range(0, h):       #for each of the values up to and including lag h:
                    #take the inner product of the first (N-i) elements in the kth col by the last (N-i) elements in the rth col
                    a = np.inner(coefsMat[0:(N-(i+1)), k], coefsMat[(i+1):N, r]).astype('float64')
                    #add to the inner product of the first (N-i) elements in the rth col. & the last (N-i) elements in the kth col.
                    + np.inner(coefsMat[0:(N-(i+1)), r], coefsMat[(i+1):N, k]).astype('float64')
                    #multiply by kernel and add to s
                    s = (chooseKernel(kernelType, i, h) * a).astype('float64')
            covMat[k,r] = s
            covMat[r,k] = covMat[k,r]

    covMat = (covMat / N)

    ## ---------------find eigen values and functions of LR Cov mat ------------------


    ## -------------- produce covariance surface plot -------------------

    return covMat

#################################################################################################################################

#assign kernel type based on input
# provide: i, h, kernelType
# out: kernel
def chooseKernel(kernelType, i, h):
    x = i/h
    if (kernelType).lower() == "flat":
        kernel = 0
    if (kernelType).lower() == "simple":
        kernel = 1
    if (kernelType).lower() == "bartlett":
        kernel = 1-x
    # ADD flat top and parzen!
    else:
        print("Kernel type not recognised: Bartlett kernel will be used")
        kernel = 1-x
    return kernel

##################################################################################################################################################

def brownianBridgeBootstrap(numBridges, lenBridgeDomain, numPerms = 1000, y0=0, y1=0, t0=0, t1=1): # M: number of B.Bridges; N: number of steps/time points in each B.Bridge
    bridgeMat = np.empty((numBridges, lenBridgeDomain), dtype=np.float32)
    bridgeVals = np.empty([numPerms])
    for iThresh in range(numPerms): 
        for iBridge in range(numBridges): 
            BB = np.array(BBridge(y0, y1, t0, t1, lenBridgeDomain))
            bridgeMat[iBridge, :] = BB**2
        bridgeVals[iThresh] = max(np.sum(bridgeMat, axis = 0))             
    bridgeVals = np.sort(bridgeVals)

    return bridgeVals

###################################################################################################################################################

def eigValBridgeBootstrap(numBridges, lenBridgeDomain, eigValCV, numPerms = 1000, y0=0, y1=0, t0=0, t1=1): # M: number of B.Bridges; N: number of steps/time points in each B.Bridge
    bridgeMat = np.empty((numBridges, lenBridgeDomain), dtype=np.float32)
    bridgeVals = np.empty([numPerms])
    for iThresh in range(numPerms): 
        for iBridge in range(numBridges): 
            BB = np.array(BBridge(y0, y1, t0, t1, lenBridgeDomain))
            bridgeMat[iBridge, :] = eigValCV[iBridge] * (BB**2) 
        bridgeVals[iThresh] = max(np.sum(bridgeMat, axis = 0))             
    bridgeVals = np.sort(bridgeVals)

    return bridgeVals

##################################################################################################################################################

def BBridge(y0=0, y1=0, t0=0, t1=1, N=100):

    """
    adapted from R code in sde package (sde > R/DM.R, found at https://rdrr.io/cran/sde/src/R/BM.R)
    only calculates one Brownian Bridge at a time i.e. is essentially the univariate version of genBrownianBridges
    """

    dt = (t1 - t0) / N #define steps in domain/x-axis
    T = np.linspace(t0, t1, N) #define linear domain according to the boundaries t0 & T, and the step calculated above
    randomSteps = np.random.normal(0, np.sqrt(dt), N-1) #draw N values from the normal distribution with mean =0  and STDev = np.sqrt(dt)
    BM = np.concatenate(([0], np.cumsum(randomSteps)), axis = None) #cumulative sum N randomly generated numbers, with 0 at start of vector
    BB = y0 + BM - ((T - t0)/(t1 - t0)) * (BM[N-1] - y1 + y0)
    X = pd.Series(BB, index=T)

    return X

######################################################################################################################################################

def cumulPercVariance(fdObj, varThresh = 0.85):
    
    if isinstance(fdObj, skfda.representation.basis._fdatabasis.FDataBasis):
        N = fdObj.coefficients[:, 0].shape[0]  # First dimension = rows = number of functional objects
        D = fdObj.coefficients[0].shape[0]  # Second dimension = columns = number of basis functions
    else: 
        #if FDataGrid
        N = fdObj.data_matrix.shape[0] # First dimension = rows = number of functional objects
        #num fPCs must be less than size of dataset and of number of sample points
        D = np.min([(N-1), (fdObj.data_matrix.shape[1])]) #set to min. of these two values
    
    fPCA = FPCA(n_components=D) #centering = TRUE applied as default

    #find variance for each fPC
    fPCAScores = fPCA.fit_transform(fdObj) #requires scores/comps to first be calculated

    fPCAVarPerc = fPCA.explained_variance_ratio_ #find variance %s
    # find number of fPCs needed to explain desired amount of variance as a % - default is 85%
    varPerc = 0
    dHat = 0
    for iVar in fPCAVarPerc:
        if varPerc < varThresh: #default must be expressed as a decimal
            varPerc += iVar
            dHat += 1 

    return dHat

###################################################################################################################################################

def imputeMissingTrialsFPCA(fdObj, trialList, firstTrial, lastTrial,
                             minObservedTrials=8, varThresh=0.85, maxComponents=None):
    """
    Reconstructs whole missing trials for a single infant/region/chromophore using
    functional PCA computed on that infant's own observed trials.

    The observed trial curves (as B-spline basis coefficients) are decomposed via FPCA
    into a mean curve plus a small number of principal component scores. Each
    component's score is modelled as a function of trial number using linear
    interpolation across the observed trials (with constant extrapolation at the
    edges), and the missing trial's score is predicted from that model. The missing
    trial's curve is then reconstructed as the mean curve plus the weighted sum of
    principal component curves using the predicted scores.

    This is a WITHIN-INFANT, WITHIN-CHROMOPHORE imputation: it borrows structure only
    from the same infant's other trials for the same measure, never from other infants
    or other chromophores.

    NOTE: because the model used to impute a trial is fit on the same trials that will
    subsequently be tested for a change point, imputed trials are not independent of
    the model used to detect change -- see the caveats discussed alongside this
    function's use in processIndividualRegionInfant. Imputed trials should be flagged
    downstream (e.g. via the returned `imputedTrials` list) so that sensitivity
    analyses excluding them can be run alongside the primary analysis.

    Parameters
    ----------
    fdObj : skfda.representation.basis.FDataBasis
        Functional data object containing only the OBSERVED (non-missing) trials for a
        single infant/region/chromophore, in the same row order as `trialList`.
    trialList : list of int
        Trial numbers corresponding to each row of `fdObj`, same length and order.
    firstTrial, lastTrial : int
        The full range of trial numbers that should be present after imputation.
    minObservedTrials : int, default 8
        Minimum number of observed trials required before imputation is attempted.
        Below this threshold, imputation is skipped and the input is returned
        unchanged, so the existing trial-count exclusion criteria still apply.
    varThresh : float, default 0.85
        Proportion of variance to retain when choosing the number of FPCA components,
        consistent with `cumulPercVariance`.
    maxComponents : int or None
        Optional hard cap on the number of FPCA components used, regardless of
        `varThresh`. Useful to prevent overfitting when few trials are observed.

    Returns
    -------
    fullFdObj : skfda.representation.basis.FDataBasis
        Functional data object with one row per trial number in
        range(firstTrial, lastTrial + 1), in ascending trial order. Observed rows are
        copied through unchanged; missing rows are FPCA-reconstructed.
    fullTrialList : list of int
        range(firstTrial, lastTrial + 1) as a list -- the trial numbers corresponding
        to each row of `fullFdObj`.
    imputedTrials : list of int
        Trial numbers that were imputed. Empty if nothing was missing, or if
        imputation was skipped due to insufficient observed trials.
    """
    basis = fdObj.basis
    coefs = fdObj.coefficients.astype('float64')
    nObs, nBasisFuncs = coefs.shape

    fullTrialRange = list(range(firstTrial, lastTrial + 1))
    missingTrials = sorted(set(fullTrialRange) - set(trialList))

    # nothing missing -- return input unchanged
    if not missingTrials:
        return fdObj, list(trialList), []

    # not enough real data to model trial-to-trial structure -- leave as-is
    # (downstream exclusion criteria, e.g. minimum trial count, will still apply)
    if nObs < minObservedTrials:
        return fdObj, list(trialList), []

    # choose number of FPCA components, capped so it stays well identified with few trials
    dHat = cumulPercVariance(fdObj, varThresh=varThresh)
    dHat = max(1, min(dHat, nObs - 1))
    if maxComponents is not None:
        dHat = min(dHat, maxComponents)

    fPCA = FPCA(n_components=dHat)
    scores = fPCA.fit_transform(fdObj)                # shape (nObs, dHat)
    meanCoef = fPCA.mean_.coefficients[0]              # shape (nBasisFuncs,)
    componentCoefs = fPCA.components_.coefficients     # shape (dHat, nBasisFuncs)

    sortedIdx = np.argsort(trialList)
    xObs = np.array(trialList)[sortedIdx]

    # model each PC score as a function of trial number; predict missing trials
    predictedScores = np.zeros((len(missingTrials), dHat))
    for k in range(dHat):
        yObs = scores[sortedIdx, k]
        interpFn = interp1d(
            xObs, yObs, kind='linear',
            bounds_error=False,
            fill_value=(yObs[0], yObs[-1])  # constant extrapolation at the edges
        )
        predictedScores[:, k] = interpFn(missingTrials)

    reconstructedCoefs = meanCoef[np.newaxis, :] + predictedScores @ componentCoefs

    # assemble the full dense coefficient matrix in trial-number order
    fullCoefs = np.zeros((len(fullTrialRange), nBasisFuncs))
    obsLookup = {t: i for i, t in enumerate(trialList)}
    missLookup = {t: i for i, t in enumerate(missingTrials)}
    for row, t in enumerate(fullTrialRange):
        if t in obsLookup:
            fullCoefs[row] = coefs[obsLookup[t]]
        else:
            fullCoefs[row] = reconstructedCoefs[missLookup[t]]

    fullFdObj = skfda.representation.basis.FDataBasis(basis, fullCoefs)

    return fullFdObj, fullTrialRange, missingTrials

###################################################################################################################################################

def computeAutocorrelation(dataMatrix, maxLag=10):
    """
    Compute lagged correlation between trials.
    
    Args:
        dataMatrix (np.array): Shape (n_trials, n_timepoints), functional data.
        maxLag (int): Maximum lag to check.
    
    Returns:
        avgCorrelation (float): Mean absolute correlation over all lags.
    """
    nTrials = dataMatrix.shape[0]
    correlations = []

    for lag in range(1, maxLag + 1):
        validTrials = nTrials - lag
        laggedCorrs = [
            pearsonr(dataMatrix[i], dataMatrix[i + lag])[0] for i in range(validTrials)
        ]
        correlations.extend(laggedCorrs)
    
    return np.abs(correlations)  # average absolute correlation across all lags

###################################################################################################################################################

def generateRandomIntervals(dataLength, numInts):
    intervals = []
    minLength = 2
    
    for _ in range(numInts):

        start = np.random.randint(0, dataLength - minLength)
        end = np.random.randint(start + minLength, dataLength)
        intervals.append((start, end))
        
    return intervals
    
##########################################################################################################################################

def computeIntervalPartialSumFF(coefs, startIdx, endIdx):
    intervalLength = endIdx - startIdx
    S_N = np.zeros([intervalLength])
    
    for i in range(2, intervalLength):
        S_N[i-1] = (np.sum((np.sum(coefs[startIdx:startIdx+i, :], axis=0) - 
                            (i/intervalLength) * np.sum(coefs[startIdx:endIdx, :], axis=0))**2)) / intervalLength
        
    return S_N

##########################################################################################################################################

# Define L_N function for an interval
def computeIntervalPartialSumFPCA(k, fPCAScores, dHat, start, end):
    intervalScores = fPCAScores[start:end]
    intervalN = end - start
    if dHat == 1:
        out = np.array(np.sum(intervalScores[:k]) - (k/intervalN)*(np.sum(intervalScores)).astype('float64'))
    else:
        out = np.sum(intervalScores[:k, :dHat], axis=0) - (k/intervalN)*(np.sum(intervalScores[:,:dHat], axis=0)).astype('float64')
    return out

##########################################################################################################################################

def filterSignificantChanges(df, alphaLevel, firstTrial, lastTrial, correctionMethod = 'h'):
    """
    Iteratively filters a dataframe based on adjusted p-values for increasing subsets of Recursion levels.
    
    Parameters:
    df (pd.DataFrame): input dataframe with columns "Recursion Level", "p-value", "First Trial", "Last Trial", "Change Point" (and possibly others)
    alphaLevel (float): significance threshold for filtering.
    firstTrial (float or int): the first trial reference value.
    lastTrial (float or int): the last trial reference value.
    correctionMethod (string): method for mutliple comparison correction - default = Bonferroni-Holm

    Returns:
    dict: A dictionary where keys are the max Chromophore included, and values are the corresponding filtered dataframes.
    """

    #return empty output if input df is empty
    if df.empty:
        return 

    df = df.copy()  # avoid modifying original dataframe
    uniqueRecursionLevels = sorted(df["Recursion Level"].unique())  # Get sorted unique recursion level values
    results = {}  # Store filtered dataframes at each step

    retainedIndexes = set(df.index) #start with all indexes

    for i in range(1, len(uniqueRecursionLevels) + 1):

        maxRecursionLevel = uniqueRecursionLevels[i-1]  # Expand subset incrementally

        # mask for the current subset of recursion level values
        subsetDf = df.loc[list(retainedIndexes)].copy() #use only indexes retained from previous loop
        mask = subsetDf["Recursion Level"] <= maxRecursionLevel #

        # first part: adjust p-values for the selected subset of recursion level values
        pValuesToCorrect = subsetDf.loc[mask, "p-value"].values

        if len(pValuesToCorrect) > 0:
            _, correctedPValues, _, _ = multipletests(pValuesToCorrect, method=correctionMethod)
            subsetDf.loc[mask, "Corrected p-value"] = correctedPValues
        # assign uncorrected p-values to recursion level > maxRecursionLevel rows
        subsetDf.loc[~mask, "Corrected p-value"] = subsetDf.loc[~mask, "p-value"]

        # second part: filter based on 'Corrected p-value' for the current 'recursionLevels' subset
        subsetDf = subsetDf[(subsetDf["Recursion Level"] > maxRecursionLevel) | (subsetDf["Corrected p-value"] <= alphaLevel)]

        # exit early if sub-dataframe empty
        if subsetDf.empty:
            break  

        # third part: remove descendants of non-significant breaks
        # empty case handled because both prevRowCount and length of DF will be 0
        prevRowCount = -1  # initialise row counter, starting at -1 to ensure runs at least once
        # continue until no more rows are removed i.e. keep checking to see if updated row removal
        while prevRowCount != len(subsetDf):  
            prevRowCount = len(subsetDf)  # store current number of rows
            
            # keep segments ending at true change point
            conditionPreChange = subsetDf.apply(lambda row: 
                any(
                    ((subsetDf["First Trial"] == row["First Trial"]) & 
                    (subsetDf["Change Point"] - 1 == row["Last Trial"])) &
                    (subsetDf.index != row.name)  # prevent row self-referencing
                ), axis=1)

            # keep segments starting at true change point
            conditionPostChange = subsetDf.apply(lambda row: 
                any(
                    ((subsetDf["Change Point"] == row["First Trial"]) & 
                    (subsetDf["Last Trial"] == row["Last Trial"])) & 
                    (subsetDf.index != row.name)  # prevents row self-referencing
                ), axis=1)

            # keep segments encompassing whole time series
            conditionFirstTest = subsetDf.apply(lambda row: 
                (firstTrial == row["First Trial"]) & 
                (lastTrial == row["Last Trial"])
            , axis=1)

            subsetDf = subsetDf[conditionPreChange | conditionPostChange | conditionFirstTest]  # filter

        retainedIndexes = set(subsetDf.index)

    return subsetDf
    
##########################################################################################################################################

def findCPtsFPCA(dataMatrix, firstFunc, lastFunc, alphaVal = 0.05, firstSamp = -20, lastSamp = 182, dHat = 0, chanMatCPts = None, pValues=None, h = 0, longRun = False, wildBinSeg = False, numBinSegIntervals=500, recursionLevel = 0):

    #if the first time calling the function
    if chanMatCPts is None:
        #define matrix (5 cols: first function, last function,  kStar + firstTrial i.e. CPt, maxT_N, pValue)
        chanMatCPts = (np.zeros([0, 6]))
        # initialise stroage for p-values so they can be corrected
        pValues = []
    
    if dataMatrix[firstFunc:lastFunc+1].shape[0] <= 2:
        return chanMatCPts, pValues

    #convert to FD
    fd = skfda.FDataGrid(
        data_matrix=dataMatrix[firstFunc:lastFunc+1],
        grid_points=np.arange(firstSamp, lastSamp, 1),
    )

    #run changeFPCA
    (kStar, pValue, maxT_N, preChangeMean, postChangeMean, deltaMeans, dHat) = funcChangePoint.changeFPCA(fd, dHat = dHat, numBootstrap = 100000, h = h, longRun = longRun, wildBinSeg=wildBinSeg, numBinSegIntervals=numBinSegIntervals)

    #add [kStar + firstTrial, maxT_N, pValue] to channel data matrix
    chanMatCPts = np.vstack([chanMatCPts, [firstFunc,
                                            lastFunc,
                                            (kStar + firstFunc),
                                            maxT_N,
                                            recursionLevel,
                                            pValue]])
    pValues.append(pValue) #collect for correction
    
    if pValue >= alphaVal:
        return chanMatCPts, pValues

    else:
        #run findAllCPts on 'first' functions i.e. 'firstTrial' -> 'kStar'
        #(remembering the 'true' change point is kStar+1 when considering Python's indexing rules)
        chanMatCPts, pValues = findCPtsFPCA(dataMatrix, firstFunc, firstFunc + (kStar - 1), alphaVal = alphaVal, firstSamp=firstSamp, lastSamp=lastSamp, dHat=dHat, chanMatCPts=chanMatCPts, pValues = pValues, h = h, longRun = longRun, wildBinSeg = wildBinSeg, numBinSegIntervals=numBinSegIntervals, recursionLevel=recursionLevel+1)

        #run findAllCPts on 'last' functions i.e. 'kStar' -> 'lastTrial'
        chanMatCPts, pValues = findCPtsFPCA(dataMatrix, kStar + firstFunc, lastFunc, alphaVal = alphaVal, firstSamp=firstSamp, lastSamp=lastSamp, dHat=dHat, chanMatCPts=chanMatCPts, pValues = pValues, h = h, longRun = longRun, wildBinSeg = wildBinSeg, numBinSegIntervals=numBinSegIntervals, recursionLevel=recursionLevel+1)
        
        return chanMatCPts, pValues
        
################################################################################################################################

def findCPtsFF(fdObj, firstFunc, lastFunc, alphaVal = 0.05, firstSamp = -20, lastSamp = 182, chanMatCPts = None, pValues=None, h = 0, longRun = False, wildBinSeg = False, numBinSegIntervals=500, recursionLevel = 0):

    #if the first time calling the function
    if chanMatCPts is None:
        #define matrix (5 cols: first function, last function,  kStar + firstTrial i.e. CPt, maxT_N, pValue)
        chanMatCPts = (np.zeros([0, 6]))
        # initialise stroage for p-values so they can be corrected
        pValues = []
    
    if fdObj.coefficients[firstFunc:lastFunc+1].shape[0] <= 2: #don't run if sub-series too short
        return chanMatCPts, pValues

    #run change point detection
    (kStar, pValue, S_N, maxS_N, preChangeMean, postChangeMean, deltaMeans) = funcChangePoint.changeFF(
        fdObj[firstFunc:lastFunc+1], numBootstrap = 10000, h = h, longRun = longRun, wildBinSeg=wildBinSeg, numBinSegIntervals=numBinSegIntervals
        )

    #add [kStar + firstTrial, maxT_N, pValue] to channel data matrix
    chanMatCPts = np.vstack([chanMatCPts, [firstFunc,
                                            lastFunc,
                                            (kStar + firstFunc),
                                            maxS_N,
                                            recursionLevel,
                                            pValue]])
    pValues.append(pValue) #collect for correction
    
    if pValue >= alphaVal:

        return chanMatCPts, pValues #stop recursion if not significant

    else:
        #run findAllCPts on 'first' functions i.e. 'firstTrial' -> 'kStar'
        #(remembering the 'true' change point is kStar+1 when considering Python's indexing rules)

        chanMatCPts, pValues   = findCPtsFF(fdObj, firstFunc, firstFunc + (kStar - 1), alphaVal = alphaVal, firstSamp=firstSamp, lastSamp=lastSamp, chanMatCPts=chanMatCPts, pValues = pValues, h = h, longRun = longRun, wildBinSeg = wildBinSeg, numBinSegIntervals=numBinSegIntervals, recursionLevel = recursionLevel+1)

        #run findAllCPts on 'last' functions i.e. 'kStar' -> 'lastTrial'

        chanMatCPts, pValues   = findCPtsFF(fdObj, kStar + firstFunc, lastFunc, alphaVal = alphaVal, firstSamp=firstSamp, lastSamp=lastSamp, chanMatCPts=chanMatCPts, pValues = pValues, h = h, longRun = longRun, wildBinSeg = wildBinSeg, numBinSegIntervals=numBinSegIntervals, recursionLevel=recursionLevel+1)
        
        return chanMatCPts, pValues 
        
################################################################################################################################

def calculateMeanCoefficients(trialCoefficients):
    """
    calculate mean coefficients for each trial.
    
    parameters:
    - trialCoefficients: dictionary with trial coefficients
    
    returns:
    - dictionary with mean coefficients per trial
    """
    meanCoefficients = {}
    for tsNum, coeffList in trialCoefficients.items():
        if coeffList:  # skips empty lists
            try:
                meanCoefficients[tsNum] = np.mean(np.stack(coeffList), axis=0)
            except ValueError as e:
                print(f"[WARNING] Skipping trial {tsNum} due to stacking error: {e}")
    
    return meanCoefficients

################################################################################################################################

def processChangePoints(fdObj, age, chrom, firstTrial, lastTrial, firstFunc, lastFunc, 
                        cptSignificance, lrCovLag, useLongRun, useWildBinSeg, regionType, regionValue):
    """
    process change points for a functional data object.
    
    parameters:
    - fdObj: functional data object
    - age, tile, chrom: metadata
    - chrom: chromophore number i.e. 1 = HbO, 2 = HbR, 3 = HbT, 4 = HbDiff
    - uniqueInfants: list of unique infant IDs for this age
    - firstTrial, lastTrial: trial range
    - firstFunc, lastFunc: function range
    - cptSignificance: significance level
    - lrCovLag: lag for long-run covariance
    - useLongRun, useWildBinSeg: algorithm parameters
    - regionType (str): either 'Tile' or 'Parcel'
    - regionValue (int): the value for the specific Tile or Parcel
    
    returns:
    - filteredTempRows: rows containing info about significant change points after filtering
    """
    regionMat, pValues = findCPtsFF(
        fdObj, firstFunc, lastFunc, 
        alphaVal=cptSignificance, h=lrCovLag, 
        longRun=useLongRun, wildBinSeg=useWildBinSeg
    )
    regionMat = regionMat.astype('float32')  # for downstream compatibility
    
    # initialise temporary parcel dataframe rows
    tempRows = []
    # populate temp. dataframe with change point results

    if regionType == 'SourceDetector':
        for i in range(regionMat.shape[0]):
            tempRows.append({
                'Age': age,
                'Source': regionValue[0],
                'Detector': regionValue[1],
                'Chromophore': chrom,
                'Recursion Level': (regionMat[i][4]),
                'First Trial': (regionMat[i][0] + 1),
                'Last Trial': (regionMat[i][1] + 1),
                'Change Point': (regionMat[i][2] + 1),
                'Max. T_N': (regionMat[i][3]),
                'p-value': (regionMat[i][5])
            })
    else:
        for i in range(regionMat.shape[0]):
            tempRows.append({
                'Age': age,
                regionType: regionValue,
                'Chromophore': chrom,
                'Recursion Level': (regionMat[i][4]),
                'First Trial': (regionMat[i][0] + 1),
                'Last Trial': (regionMat[i][1] + 1),
                'Change Point': (regionMat[i][2] + 1),
                'Max. T_N': (regionMat[i][3]),
                'p-value': (regionMat[i][5])
            })
    
    # create temp. dataframe
    tempDf = pd.DataFrame(tempRows)
    if not tempDf.empty:
        tempDf.index = range(0, len(tempDf))
    
    filteredTempDf = filterSignificantChanges(tempDf, cptSignificance, firstTrial, lastTrial)

    filteredTempRows = filteredTempDf.to_dict(orient='records')

    return filteredTempRows

################################################################################################################################

def saveExcludedLogs(excludedCombinedValidLog, excludedCombinedTrialsLog, excludedChromValidLog, excludedChromTrialsLog):
    
    # Combine all exclusion logs into one list
    combinedExcludedLog = []

    # Add entries from excludedCombinedValidLog
    for entry in excludedCombinedValidLog:
        entry['Log Type'] = 'Combined Valid'
        combinedExcludedLog.append(entry)

    # Add entries from excludedCombinedTrialsLog
    for entry in excludedCombinedTrialsLog:
        entry['Log Type'] = 'Combined Trials'
        combinedExcludedLog.append(entry)

    # Add entries from excludedChromValidLog
    for entry in excludedChromValidLog:
        entry['Log Type'] = 'Chrom Valid'
        combinedExcludedLog.append(entry)

    # Add entries from excludedChromTrialsLog
    for entry in excludedChromTrialsLog:
        entry['Log Type'] = 'Chrom Trials'
        combinedExcludedLog.append(entry)

    # Convert the combined log into a DataFrame
    combinedExcludedDf = pd.DataFrame(combinedExcludedLog)

    return combinedExcludedDf

################################################################################################################################

def saveImputationLog(imputedTrialsLog):
    """
    Converts the per-infant/region/chromophore imputed-trial log (a list of dicts,
    one row per imputed trial) into a DataFrame for saving alongside the change-point
    results, so imputed trials can be identified and, if desired, excluded in a
    sensitivity analysis.
    """
    return pd.DataFrame(imputedTrialsLog)

################################################################################################################################


def findFirstCPtsFF(fdObj, firstFunc, lastFunc, alphaVal = 0.05, firstSamp = -20, lastSamp = 182, chanMatCPts = None, pValues=None, h = 0, longRun = False, wildBinSeg = False, numBinSegIntervals=500, recursionLevel = 0):

    #if the first time calling the function
    if chanMatCPts is None:
        #define matrix (5 cols: first function, last function,  kStar + firstTrial i.e. CPt, maxT_N, pValue)
        chanMatCPts = (np.zeros([0, 6]))
        # initialise stroage for p-values so they can be corrected
        pValues = []
    
    if fdObj.coefficients[firstFunc:lastFunc+1].shape[0] <= 2: #don't run if sub-series too short
        return chanMatCPts, pValues

    #run change point detection
    (kStar, pValue, S_N, maxS_N, preChangeMean, postChangeMean, deltaMeans) = funcChangePoint.changeFF(
        fdObj[firstFunc:lastFunc+1], numBootstrap = 10000, h = h, longRun = longRun, wildBinSeg=wildBinSeg, numBinSegIntervals=numBinSegIntervals
        )

    #add [kStar + firstTrial, maxT_N, pValue] to channel data matrix
    chanMatCPts = np.vstack([chanMatCPts, [firstFunc,
                                            lastFunc,
                                            (kStar + firstFunc),
                                            maxS_N,
                                            recursionLevel,
                                            pValue]])
    pValues.append(pValue) #collect for correction

    return chanMatCPts, pValues 

################################################################################################################################

def findGroupRegionFirstChangePoints(fdObj, age, chrom, firstTrial, lastTrial, firstFunc, lastFunc, 
                        cptSignificance, lrCovLag, useLongRun, useWildBinSeg, regionType, regionValue):
    """
    find first change point for functional object
    
    parameters:
    - fdObj: functional data object
    - age, tile, chrom: metadata
    - chrom: chromophore number i.e. 1 = HbO, 2 = HbR, 3 = HbT, 4 = HbDiff
    - uniqueInfants: list of unique infant IDs for this age
    - firstTrial, lastTrial: trial range
    - firstFunc, lastFunc: function range
    - cptSignificance: significance level
    - lrCovLag: lag for long-run covariance
    - useLongRun, useWildBinSeg: algorithm parameters
    - regionType (str): either 'Tile' or 'Parcel'
    - regionValue (int): the value for the specific Tile or Parcel
    
    returns:
    - filteredTempRows: rows containing info about significant change points after filtering
    """
    regionMat, pValues = findFirstCPtsFF(
        fdObj, firstFunc, lastFunc, 
        alphaVal=cptSignificance, h=lrCovLag, 
        longRun=useLongRun, wildBinSeg=useWildBinSeg
    )
    regionMat = regionMat.astype('float32')  # for downstream compatibility
    
    # initialise temporary parcel dataframe rows
    tempRows = []
    # populate temp. dataframe with change point results

    if regionType == 'SourceDetector':
        for i in range(regionMat.shape[0]):
            tempRows.append({
                'Age': age,
                'Source': regionValue[0],
                'Detector': regionValue[1],
                'Chromophore': chrom,
                'Recursion Level': (regionMat[i][4]),
                'First Trial': (regionMat[i][0] + 1),
                'Last Trial': (regionMat[i][1] + 1),
                'Change Point': (regionMat[i][2] + 1),
                'Max. T_N': (regionMat[i][3]),
                'p-value': (regionMat[i][5])
            })
    else:
        for i in range(regionMat.shape[0]):
            tempRows.append({
                'Age': age,
                regionType: regionValue,
                'Chromophore': chrom,
                'Recursion Level': (regionMat[i][4]),
                'First Trial': (regionMat[i][0] + 1),
                'Last Trial': (regionMat[i][1] + 1),
                'Change Point': (regionMat[i][2] + 1),
                'Max. T_N': (regionMat[i][3]),
                'p-value': (regionMat[i][5])
            })
    
    # create temp. dataframe
    tempDf = pd.DataFrame(tempRows)
    if not tempDf.empty:
        tempDf.index = range(0, len(tempDf))

    tempRows = tempDf.to_dict(orient='records')

    return tempRows