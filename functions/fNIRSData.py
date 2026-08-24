import h5py 
import numpy as np
import os
import pandas as pd
import pickle
import scipy.io
import skfda

# NOTE: calculateMeanCoefficients was previously imported from BOTH auxFuncChPt
# and sharedUtils (the second import silently overwriting the first, since
# both modules define an identical, byte-for-byte-matching function). Kept
# here from sharedUtils only, since that was the version actually in effect;
# findGroupRegionFirstChangePoints (only defined in auxFuncChPt) is unaffected.
from auxFuncChPt import findGroupRegionFirstChangePoints
from collections import Counter
from sharedUtils import calculateMeanCoefficients
from scipy.stats import gamma
from skfda.representation.basis import BSplineBasis, FDataBasis
from statistics import mean
import sys
import warnings

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

# NOTE: this self-import looks unusual, but is genuinely needed - some
# functions in this file (e.g. processInfantTrials) call fNIRSData.xxx(...)
# without receiving fNIRSData as an explicit parameter, so they rely on this
# module-level import of itself. Functions that DO receive fNIRSData as an
# explicit parameter (e.g. processIndividualRegionInfant, processGroupRegion)
# use that parameter instead - this self-import is a fallback for the rest.
import fNIRSData

def storeChannelData(datapath, partID, chromNum, chanNum):

    ##create a dictionary for a specific participant, chrompohore and channel containing the timeseries for each trial
    # Inputs:
    # datapath: directory containing matlab matrix files output by the sb_getAllTrials matlab function
    # partID (int): the 4 digit participant number from the start of the filename
    # chromNum (int): the chromophore for which the the data is to be collected (1 = HbO, 2 = HbR, 3 = HbT)
    # chanNum (int): channel number for which the the data is to be collected
    #
    # Output:
    # 
    #
    #SB 3/3/23

    #find filenames in directory (excl. README file and hidden files)
    filenames = [f for f in os.listdir(datapath) if not (f.startswith('.') or f.startswith('R') or f.startswith('C'))]

    #iterate through files in data directory
    #assumes working directory has already been changed in parent script/function
    for file in filenames:
        
        #truncate file type
        filename = file[:-4]

        #find matching participant, chromophore and channel
        if int(filename[0:4]) == partID and int(filename[5:6]) == chromNum and int(filename[7:]) == chanNum:
            
            #load file
            matFile = scipy.io.loadmat(file)
            
            #go to TS data
            arr = matFile["tsMat"]
            len(arr)

            #create dictionary and fill it
            dictionary = dict()
            for i in range(0,len(arr)):
                dictionary[i+1] = arr[i] 
                
    return dictionary

##############################################################################

def canonicalHrf(t, double = 1, onset=0, duration=20, peakDelay=6, underDelay=16, puRatio=6, width=1, amp=1):
    """
    Generate a canonical hemodynamic response function (HRF) using the difference of two gamma functions, or a single gamme function.

    Returns:
        hrf (array): HRF values at each time point in `t`.
    """
    tSec = t - onset
    tSec[tSec < 0] = 0  # No response before onset

    # Gamma functions for peak and undershoot
    if double == 1:
        peak = gamma.pdf(tSec, peakDelay)
        undershoot = gamma.pdf(tSec, underDelay) / puRatio
        hrf = amp * (peak - undershoot)
    else:
        hrf = amp * gamma.pdf(tSec, a=peakDelay, scale=width)

    return hrf

##############################################################################

def getTrialTS(dataFrameName, chromNum, chanNum, trialNum):

    ##fill matrix with the time series across participants for the given chromophore, channel and trial number
    #
    # filename: pickle file 'filename.pkl' containing the nested dictionary as an output of storeChannelData
    # chromNum (int): the chromophore for which the the data is to be collected (1 = HbO, 2 = HbR, 3 = HbT)
    # chanNum (int): channel number for which the the data is to be collected
    # trialNum (int):   integer (usually 1-25) corresponding to the trial number for which the the data is to be collected
    #
    #SB 3/3/23
    #
    #SLB 24_6_24
    #Alter to accept pd dataframes in lieu of a filename to avoid the need to load the Dataframe every time calling the function
    #if already loaded elsewhere!
        
    if isinstance(dataFrameName, pd.DataFrame):
        dfFull = dataFrameName
    else:
        with open(dataFrameName, 'rb') as f:
            dfFull = pickle.load(f)

    #obtain necessary iterables
    partIDs = sorted(set(dfFull.loc[:,'ID']))
    trialNums = sorted(set(dfFull.loc[:,'Trial']))

    # slice DF to find relevant rows
    df = dfFull[(dfFull['Trial'] == trialNum) & (dfFull['Channel'] == chanNum) & (dfFull['Chrom'] == chromNum)]
    
    #create empty list to enter rows which will be removed
    dropList = []

    for partID in partIDs:
        tempRow = df[df.ID == partID]
        tempTS = tempRow['Time series'].item()

        #if timeseries contains dummy variables, not the index so it can be removed
        if (mean(tempTS) == -101 or mean(tempTS) == -102 or mean(tempTS) == -103 or mean(tempTS) == -104):
            i = df[((df.ID == partID))].index.item()
            dropList.append(i)

    #amend dataframe to remove missing data
    df = df.drop(dropList)
    
    return df

########################################################################

def getParcelTrialTS(dataFrameName, age, chromNum, parcNum, trialNum):

    ##fill matrix with the time series across participants for the given chromophore, channel and trial number
    #
    # filename: pickle file 'filename.pkl' containing the nested dictionary as an output of storeChannelData
    # chromNum (int): the chromophore for which the the data is to be collected (1 = HbO, 2 = HbR, 3 = HbT)
    # chanNum (int): channel number for which the the data is to be collected
    # trialNum (int):   integer (usually 1-25) corresponding to the trial number for which the the data is to be collected
    #
    #SB 3/3/23
    #
    #SLB 24_6_24
    #Alter to accept pd dataframes in lieu of a filename to avoid the need to load the Dataframe every time calling the function
    #if already loaded elsewhere!
        
    if isinstance(dataFrameName, pd.DataFrame):
        dfFull = dataFrameName
    else:
        with open(dataFrameName, 'rb') as f:
            dfFull = pickle.load(f)

    #obtain necessary iterables
    partIDs = sorted(set(dfFull.loc[:,'Infant']))
    trialNums = sorted(set(dfFull.loc[:,'Trial']))

    # slice DF to find relevant rows
    df = dfFull[(dfFull['Age'] == age) & (dfFull['Trial'] == trialNum) & (dfFull['Parcel'] == parcNum) & (dfFull['Chrom'] == chromNum)]
    
    # #create empty list to enter rows which will be removed
    # dropList = []

    # for partID in partIDs:
    #     tempRow = df[df.Infant == partID]

    #     #if timeseries contains dummy variables, note the index so it can be removed
    #     if tempRow.empty:
    #         i = df[((df.Infant == partID))].index.item()
    #         dropList.append(i)

    # #amend dataframe to remove missing data
    # df = df.drop(dropList)
    
    return df

########################################################################

def groupDataMatrix(dataFile,
                    numChan,
                    numChrom,
                    firstTrial = 1,
                    lastTrial = 25,
                    firstSamp = -20,
                    lastSamp = 182):

    gridPoints = np.arange(firstSamp, lastSamp, 1)  # Grid points of the curves
    dataMatrix = np.empty([lastTrial - (firstTrial-1), 202]) #softcode number of trials?

    for trial in range(firstTrial, lastTrial +1):
        trialDF = fNIRSData.getTrialTS(dataFile, numChrom, numChan, trial)
        numParts = len(trialDF)
        
        meanMat = np.empty([numParts, len(trialDF['Time series'].iloc[0])])
        for i in range(0, numParts):
            y = (trialDF['Time series'].iloc[i-1])
            coeff = np.polyfit(gridPoints, y, 1)
            yPoly = np.polyval(coeff, gridPoints)
            meanMat[i] = y - yPoly 

        meanNP = np.mean(meanMat, axis = 0)
        dataMatrix[trial - firstTrial -1] = meanNP

    for i in range(0, dataMatrix.shape[0]):
        iVal = dataMatrix[i, 21]
        dataMatrix[i, :] = dataMatrix[i, :] - iVal

    return dataMatrix

########################################################################

def groupParcelDataMatrix(dataFile,
                    age, 
                    numParc,
                    numChrom,
                    firstTrial = 1,
                    lastTrial = 25,
                    firstSamp = 0,
                    lastSamp = 181):

    for trial in range(firstTrial, lastTrial +1):
        trialDF = fNIRSData.getParcelTrialTS(dataFile, age, numChrom, numParc, trial)
        numParts = len(trialDF)

        gridPoints = np.arange(firstSamp, len(trialDF['Time series'].iloc[0]), 1)  # Grid points of the curves
        dataMatrix = np.empty([lastTrial - (firstTrial-1), len(gridPoints)]) 

        if numParts != 0: #shouldn't happen as check in main script applied, but just in case to stop errors
            meanMat = np.empty([numParts, len(trialDF['Time series'].iloc[0])])
            for i in range(0, numParts):
                y = (trialDF['Time series'].iloc[i-1])
                coeff = np.polyfit(gridPoints, y, 1)
                yPoly = np.polyval(coeff, gridPoints)
                meanMat[i] = y - yPoly 

            meanNP = np.nanmean(meanMat, axis = 0)
            dataMatrix[trial - firstTrial -1] = meanNP

    for i in range(0, dataMatrix.shape[0]):
        iVal = dataMatrix[i, 21]
        if np.isnan(iVal) or np.isinf(iVal):
            print(f"Invalid iVal at row {i}: {iVal}")
            continue  # or set dataMatrix[i, :] = np.nan
        if np.isnan(dataMatrix[i, :]).any() or np.isinf(dataMatrix[i, :]).any():
            print(f"Invalid data in row {i}")
            continue  # or clean it
        dataMatrix[i, :] = dataMatrix[i, :] - iVal

    return dataMatrix

#######################################################################################

def getROI(agePoint, cohort):

    """
    Defines channels consistent with TFCE-derived ROI channels for each age group.
    # TO DO: update UK values!!
    """

    if cohort == 'GM':
        if agePoint == 5:
            tfceListTotal = [3, 4, 5, 8, 9, 11, 12, 13, 17, 18, 19, 21, 28]
            tfceListHbo = [3, 4, 5, 8, 9, 11, 12, 13, 17, 18, 19]
            tfceListHbr = [8, 9, 12, 17, 18, 19, 21, 28]
            channels = {
                "Total": tfceListTotal,
                "HbO": tfceListHbo,
                "HbR": tfceListHbr,
                "Both": list(set(tfceListHbo) & set(tfceListHbr))
            }
        elif agePoint == 8:
            tfceListTotal = [3, 4, 8, 9, 12, 13, 16, 17, 18, 19]
            tfceListHbo = [3, 4, 8, 9, 12, 13, 17, 18, 19]
            tfceListHbr = [3, 4, 12, 13, 16, 18]
            channels = {
                "Total": tfceListTotal,
                "HbO": tfceListHbo,
                "HbR": tfceListHbr,
                "Both": list(set(tfceListHbo) & set(tfceListHbr))
            }
        elif agePoint == 12:
            tfceListTotal = [3, 4, 5, 8, 9, 10, 12, 13, 16, 17, 18, 19]
            tfceListHbo = [3, 4, 5, 8, 9, 10, 12, 17, 18, 19]
            tfceListHbr = [3, 4, 8, 9, 12, 13, 16, 18]
            channels = {
                "Total": tfceListTotal,
                "HbO": tfceListHbo,
                "HbR": tfceListHbr,
                "Both": list(set(tfceListHbo) & set(tfceListHbr))
            }
    elif cohort == 'UK':
        if agePoint == 5:
            channels = [3, 4, 8, 9, 12, 18]
        elif agePoint == 8:
            channels = [3, 4, 8, 9, 12, 18]
        elif agePoint == 12:
            channels = [3, 4, 8, 9, 12, 18]        
        
    return channels

########################################################################################

def getTileNumbers(padname):
    """ 
    Returns the source and detector tile mappings corresponding to a given pad name.
    Inputs:
              padname (string): The name of the pad or cap configuration 
              (e.g., 'GA00274', 'GA00438_NF', etc.).
      
    Outputs:
              sourceTile (dict): A dictionary where keys are tile numbers and values are 
              lists of source numbers grouped into the same tile.
    
              detTile (dict): A dictionary where keys are tile numbers and values are 
              lists of detector numbers grouped into the same tile.
    """

    sourceTile = {}
    detTile = {}

    if padname == 'GA00274':
        sourceTile = {
            1: [34, 35, 36],
            2: [31, 32, 33],
            3: [28, 29, 30],
            4: [19, 20, 21],
            5: [22, 23, 24],
            6: [25, 26, 27],
            7: [1, 2, 3],
            8: [4, 5, 6],
            9: [7, 8, 9],
            10: [16, 17, 18],
            11: [13, 14, 15],
            12: [10, 11, 12]
        }
        detTile = {
            1: [45, 46, 47, 48],
            2: [41, 42, 43, 44],
            3: [37, 38, 39, 40],
            4: [25, 26, 27, 28],
            5: [29, 30, 31, 32],
            6: [33, 34, 35, 36],
            7: [1, 2, 3, 4],
            8: [5, 6, 7, 8],
            9: [9, 10, 11, 12],
            10: [21, 22, 23, 24],
            11: [17, 18, 19, 20],
            12: [13, 14, 15, 16]
        }

    elif padname in {'GA00438_NF', 'GA00440_NF'}:
        sourceTile = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [34, 35, 36],
            5: [31, 32, 33],
            6: [28, 29, 30],
            7: [16, 17, 18],
            8: [13, 14, 15],
            9: [10, 11, 12],
            10: [19, 20, 21],
            11: [22, 23, 24],
            12: [25, 26, 27]
        }
        detTile = {
            1: [1, 2, 3, 4],
            2: [5, 6, 7, 8],
            3: [9, 10, 11, 12],
            4: [45, 46, 47, 48],
            5: [41, 42, 43, 44],
            6: [37, 38, 39, 40],
            7: [21, 22, 23, 24],
            8: [17, 18, 19, 20],
            9: [13, 14, 15, 16],
            10: [25, 26, 27, 28],
            11: [29, 30, 31, 32],
            12: [33, 34, 35, 36]
        }

    elif padname in {'GA00440', 'GA00438', 'GA00439'}:
        sourceTile = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [43, 44, 45],
            5: [40, 41, 42],
            6: [37, 38, 39],
            7: [22, 23, 24],
            8: [19, 20, 21],
            9: [16, 17, 18],
            10: [25, 26, 27],
            11: [28, 29, 30],
            12: [31, 32, 33],
            13: [10, 11, 12],
            14: [34, 35, 36],
            15: [13, 14, 15]
        }
        detTile = {
            1: [1, 2, 3, 4],
            2: [5, 6, 7, 8],
            3: [9, 10, 11, 12],
            4: [57, 58, 59, 60],
            5: [53, 54, 55, 56],
            6: [49, 50, 51, 52],
            7: [29, 30, 31, 32],
            8: [25, 26, 27, 28],
            9: [21, 22, 23, 24],
            10: [33, 34, 35, 36],
            11: [37, 38, 39, 40],
            12: [41, 42, 43, 44],
            13: [13, 14, 15, 16],
            14: [45, 46, 47, 48],
            15: [17, 18, 19, 20]
        }

    elif padname in {'GA00370', 'GA00369', 'GA00351'}:
        sourceTile = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [34, 35, 36],
            5: [31, 32, 33],
            6: [19, 20, 21],
            7: [16, 17, 18],
            8: [13, 14, 15],
            9: [22, 23, 24],
            10: [25, 26, 27],
            11: [10, 11, 12],
            12: [28, 29, 30]
        }
        detTile = {
            1: [1, 2, 3, 4],
            2: [5, 6, 7, 8],
            3: [9, 10, 11, 12],
            4: [45, 46, 47, 48],
            5: [41, 42, 43, 44],
            6: [25, 26, 27, 28],
            7: [21, 22, 23, 24],
            8: [17, 18, 19, 20],
            9: [29, 30, 31, 32],
            10: [33, 34, 35, 36],
            11: [13, 14, 15, 16],
            12: [37, 38, 39, 40]
        }

    return sourceTile, detTile

########################################################################################

def addTileColumn(metadataDf):

    """"
    Creates a column in metadataDf with the tile number, according to the information
    in the 'Age', 'Source', 'Detector', 'Cap name'
    """

    tileList = []
    for _, row in metadataDf.iterrows():
        capName = row['Cap name']
        sourceNum = row['Source']
        detectorNum = row['Detector']

        padName = fNIRSData.extractPadName(capName)
        sourceTile, detTile = fNIRSData.getTileNumbers(padName)

        tileFound = None
        for tileNum, sources in sourceTile.items():
            if sourceNum in sources:
                if detectorNum in detTile.get(tileNum, []):
                    tileFound = tileNum
                    break
        
        if tileFound is None:
            warnings.warn(f"No matching tile found for source {sourceNum} and detector {detectorNum}")
            tileFound = -1    

        tileList.append(tileFound)        

    tileList = np.array(tileList, dtype=np.int8)  # force int8

    metadataDf = metadataDf.copy()
    metadataDf.insert(metadataDf.columns.get_loc('Detector') + 1, 'Tile', tileList)
    return metadataDf

########################################################################################

def extractPadName(capName):

    """"
    Removes orientation suffixes from cap names if present
    Designed to be called by addTileColumn
    """

    for suffix in ['_RU', '_LU', '_R', '_L', '_U']:
        if capName.endswith(suffix):
            return capName.replace(suffix, '')
    return capName

########################################################################################

def prepareTimeSeries(ts, stimOnset):
    """
    apply preprocessing steps to a time series:
    1. linear detrending
    2. offset by value at stimulus onset
    3. slice from stimulus onset
    
    parameters:
    - ts: time series data array (n_trials x n_samples)
    - stimOnset: index of stimulus onset
    
    returns:
    - preprocessed time series
    """
    # linear detrending
    for i in range(ts.shape[0]):
        y = ts[i]
        coeff = np.polyfit(np.arange(0, len(y)), y, 1)
        yPoly = np.polyval(coeff, np.arange(0, len(y)))
        ts[i] = y - yPoly
    
    # offset by value at stimulus onset
    ts = ts - ts[:, stimOnset][:, np.newaxis]
    
    # slice from stimulus onset
    ts = ts[:, stimOnset:]
    
    return ts

########################################################################################

def determineSamplingParameters(minLen, firstSamp, lastSamp):
    """
    determine sampling frequency and stimulus onset sample based on data.
    
    parameters:
    - minLen: minimum length of time series
    - firstSamp: first sample time
    - lastSamp: last sample time
    
    returns:
    - fs: sampling frequency
    - stimOnset: index of stimulus onset
    """
    ratio = minLen / (abs(firstSamp) + abs(lastSamp)) * 10
    if np.isclose(ratio, 13.0, atol=0.2):
        fs = 12.5  # Hz
        stimOnset = 49  # 50th sample
    else:
        fs = 10
        stimOnset = 39  # 40th sample
    
    return fs, stimOnset

########################################################################################

def convertToFunctionalData(ts, gridPoints, numBasis):

    """
    convert time series data to functional data objects.
    
    parameters:
    - ts: time series data array (n_trials x n_samples)
    - gridPoints: grid points for functional data
    - numBasis: number of basis functions
    
    returns:
    - functional data object
    """
    fdataGrid = skfda.FDataGrid(data_matrix=ts, grid_points=gridPoints)
    basis = BSplineBasis(n_basis=numBasis)
    fdataObj = fdataGrid.to_basis(basis)
    
    return fdataObj

########################################################################################

def getCommonTrialsData(df1, df2):
    """
    find common trials between two dataframes and return filtered dataframes.
    
    parameters:
    - df1: first dataframe
    - df2: second dataframe
    
    returns:
    - df1Common: filtered first dataframe
    - df2Common: filtered second dataframe
    """
    df1Trials = df1['Trial']
    df2TrialsSet = set(df2['Trial'])
    commonTrials = df1Trials.isin(df2TrialsSet)
    df1Common = df1[commonTrials]
    
    commonTrialsSet = set(df1Common['Trial'])
    df2Common = df2[df2['Trial'].isin(commonTrialsSet)]
    
    return df1Common, df2Common

########################################################################################

def extractTrialData(tsDataset, df):
    """
    process trial data from a dataframe and extract time series.
    
    parameters:
    - tsDataset: time series dataset
    - df: dataframe with trial information
    - minLen: minimum length for time series
    
    outputs:
    - ts: time series data array
    - trials: list of trial numbers
    - valid: whether valid data was found
    """
    if df.empty:
        return None, [], False
    
    tsList = [tsDataset[tsIdx] for tsIdx in df['tsIDx']]
    trialList = df['Trial'].tolist()
    
    if len(tsList) == 0:
        return None, [], False
    
    minLength = min(len(ts) for ts in tsList)
    
    ts = np.array([ts[:minLength] for ts in tsList])
    
    # remove rows with any NaNs and corresponding trials
    validRows = ~np.isnan(ts).any(axis=1)
    ts = ts[validRows]
    trials = [trial for trial, valid in zip(trialList, validRows) if valid]
    
    if len(trials) == 0:
        return None, [], False
    
    return ts, trials, True

########################################################################################


def getIndicesForTrials(trialList, trialKeepCounts):
    """
    Return indices from trialList corresponding to trialKeepCounts,
    preserving order and respecting duplicate limits.
    
    Parameters:
        trialList (list): List of trial numbers (may contain duplicates).
        trialKeepCounts (dict): {trial_num: max_allowed_occurrences}.
        
    Returns:
        list: Indices of trials to keep.
    """
    trialSeen = Counter()
    indices = []
    for i, t in enumerate(trialList):
        if t in trialKeepCounts and trialSeen[t] < trialKeepCounts[t]:
            indices.append(i)
            trialSeen[t] += 1
    return indices

########################################################################################

def processInfantTrials(subsetDf, tsDataset, infant, chrom, firstSamp, lastSamp, numBasis, lastTrial, regionType, regionValue):

    if regionType == 'SourceDetector':
        comboDf = subsetDf[
            (subsetDf['Infant'] == infant) &
            (subsetDf['Source'] == regionValue[0]) &
            (subsetDf['Detector'] == regionValue[1]) &
            (subsetDf['Chrom'] == chrom) &
            (subsetDf['Trial'] <= lastTrial) 
        ]
    else:    
        comboDf = subsetDf[
            (subsetDf['Infant'] == infant) &
            (subsetDf[regionType] == regionValue) & 
            (subsetDf['Chrom'] == chrom) &
            (subsetDf['Trial'] <= lastTrial) 
        ]
    comboTs, trialList, valid = fNIRSData.extractTrialData(tsDataset, comboDf)

    if not valid:
        return None, None
    
    # find time series length for use downstream
    minLen = min(len(ts) for ts in comboTs)

    fs, stimOnset = fNIRSData.determineSamplingParameters(minLen, firstSamp, lastSamp)
    comboTs = fNIRSData.prepareTimeSeries(comboTs, stimOnset)

    gridPoints = np.arange(0, lastSamp+2, abs(lastSamp+2)/comboTs.shape[1])
    infTileChromObj = fNIRSData.convertToFunctionalData(comboTs, gridPoints, numBasis)

    return infTileChromObj, trialList

########################################################################################

def computeGroupCoefficients(uniqueInfants, subsetDf, tsDataset, chrom, 
                firstTrial, lastTrial, firstSamp, lastSamp, numBasis, regionType, regionValue):
    
    groupTrialCoefficients = {t: [] for t in range(firstTrial, lastTrial+1)}

    for infant in uniqueInfants:
        infObj, trialList = processInfantTrials(
            subsetDf, tsDataset, infant, chrom, firstSamp, lastSamp, numBasis, lastTrial, regionType, regionValue
        )
        if infObj is None:
            continue

        for i in range(len(trialList)):
            trialNum = trialList[i]
            groupTrialCoefficients[trialNum].append(infObj.coefficients[i])

    return groupTrialCoefficients
    
########################################################################################

def processIndividualRegionInfant(age, infant, subsetDf, tsDataset, uniqueChroms, basis,
                            firstTrial, lastTrial, firstSamp, lastSamp,
                            numBasis, firstFunc, lastFunc,
                            cptSignificance, lrCovLag, useLongRun, useWildBinSeg,
                            fNIRSData, auxFuncChPt, regionType, regionValue,
                            useTrialImputation=False, minObservedTrialsForImputation=8,
                            imputationVarThresh=0.85, maxImputationComponents=None,
                            restrictChroms=None):
    
    """
    runs tile-based change point detection for individual data; allows for parallelisation of the tile/infant loop
    to speed up processing.

    useTrialImputation : bool, default False
        If True, whole missing trials (individual level) are reconstructed via
        auxFuncChPt.imputeMissingTrialsFPCA before the change-point test, both for the
        combined HbT/HbDiff measures and for each chromophore analysed independently.
        Imputation is within-infant, within-chromophore, and is skipped for a given
        infant/region/chromophore if fewer than `minObservedTrialsForImputation` real
        trials are available (in which case behaviour falls back to the original
        complete-case pipeline). Default is False so existing results are reproduced
        exactly unless explicitly opted in.
    minObservedTrialsForImputation : int, default 8
        Passed through to auxFuncChPt.imputeMissingTrialsFPCA.
    imputationVarThresh, maxImputationComponents :
        Passed through to auxFuncChPt.imputeMissingTrialsFPCA.
    restrictChroms : iterable or None, default None
        If provided, restricts the per-chromophore loop (Chrom == 1, 2, ... analysed
        individually) to only these chromophore codes for this specific call, e.g. to
        subset to infant x parcel x age x chromophore combinations already flagged by
        an external analysis (such as TFCE). Does NOT affect the combined HbT/HbDiff
        analysis, which always uses both HbO and HbR. Pass None (default) to keep the
        original behaviour of looping over all of uniqueChroms.
    """
    
    excludedCombinedValidLog = []
    excludedCombinedTrialsLog = []
    excludedChromValidLog = []
    excludedChromTrialsLog = []
    imputedTrialsLog = []
    indChangepointRows = []

    # Get HbO and HbR data
    if regionType == 'SourceDetector':
        hboDf = subsetDf[(subsetDf['Infant'] == infant) & (subsetDf['Source'] == regionValue[0]) & (subsetDf['Detector'] == regionValue[1]) & (subsetDf['Chrom'] == 1)]
        hbrDf = subsetDf[(subsetDf['Infant'] == infant) & (subsetDf['Source'] == regionValue[0]) & (subsetDf['Detector'] == regionValue[1]) & (subsetDf['Chrom'] == 2)]
    else:
        hboDf = subsetDf[(subsetDf['Infant'] == infant) & (subsetDf[regionType] == regionValue) & (subsetDf['Chrom'] == 1)]
        hbrDf = subsetDf[(subsetDf['Infant'] == infant) & (subsetDf[regionType] == regionValue) & (subsetDf['Chrom'] == 2)]

    hboDfCommon, hbrDfCommon = fNIRSData.getCommonTrialsData(hboDf, hbrDf)
    hboTs, hboTrialList, hboValid = fNIRSData.extractTrialData(tsDataset, hboDfCommon)
    hbrTs, hbrTrialList, hbrValid = fNIRSData.extractTrialData(tsDataset, hbrDfCommon)

    if not (hboValid and hbrValid):
        print(f"Excluded Infant {infant}, Parcel {regionValue}: invalid HbO or HbR data")
        excludedCombinedValidLog.append({'Infant': infant, regionType: regionValue, 'Age': age})
        return indChangepointRows, excludedCombinedValidLog, excludedCombinedTrialsLog, excludedChromValidLog, excludedChromTrialsLog, imputedTrialsLog

    if (not hbrTrialList[0] in [1, 2]) or (not len(hbrTrialList) > 20):
        print(f"Excluded Infant {infant}, Parcel {regionValue}: insufficient trials (n={len(hbrTrialList)})")
        excludedCombinedTrialsLog.append({'Infant': infant, regionType: regionValue, 'Age': age})
        return indChangepointRows, excludedCombinedValidLog, excludedCombinedTrialsLog, excludedChromValidLog, excludedChromTrialsLog, imputedTrialsLog
    
    # Align by trial occurrence count
    hboCounts = Counter(hboTrialList)
    hbrCounts = Counter(hbrTrialList)
    commonTrials = set(hboCounts.keys()).intersection(hbrCounts.keys())
    trialKeepCounts = {trial: min(hboCounts[trial], hbrCounts[trial]) for trial in commonTrials}

    hboIdxSorted = getIndicesForTrials(hboTrialList, trialKeepCounts)
    hbrIdxSorted = getIndicesForTrials(hbrTrialList, trialKeepCounts)

    hboTs = hboTs[hboIdxSorted]
    hbrTs = hbrTs[hbrIdxSorted]
    hboTrialList = [hboTrialList[i] for i in hboIdxSorted]
    hbrTrialList = [hbrTrialList[i] for i in hbrIdxSorted]

    # final paranoia check...
    if hboTs.shape[0] != hbrTs.shape[0] or hboTrialList != hbrTrialList:
        print(f"[WARNING] Trial alignment mismatch after indexing — Infant: {infant}, Age: {age}, Region: {regionType} {regionValue}")
        # Optionally: truncate to min length here if absolutely necessary, but better to fix upstream
        minTrials = min(hboTs.shape[0], hbrTs.shape[0])
        hboTs = hboTs[:minTrials]
        hbrTs = hbrTs[:minTrials]
        hboTrialList = hboTrialList[:minTrials]
        hbrTrialList = hbrTrialList[:minTrials]

    minLen = min(len(ts) for ts in hboTs)
    fs, stimOnset = fNIRSData.determineSamplingParameters(minLen, firstSamp, lastSamp)

    hboTs = fNIRSData.prepareTimeSeries(hboTs, stimOnset)
    hbrTs = fNIRSData.prepareTimeSeries(hbrTs, stimOnset)

    hboGridPoints = np.arange(0, lastSamp+2, abs(lastSamp+2)/hboTs.shape[1])
    hbrGridPoints = np.arange(0, lastSamp+2, abs(lastSamp+2)/hbrTs.shape[1])

    hboInfRegionObj = fNIRSData.convertToFunctionalData(hboTs, hboGridPoints, numBasis)
    hbrInfRegionObj = fNIRSData.convertToFunctionalData(hbrTs, hbrGridPoints, numBasis)

    # Optionally reconstruct whole missing trials (individual level) via FPCA on this
    # infant's own observed trials, so downstream trial-index -> trial-number mapping
    # is dense over [firstTrial, lastTrial] and imputed trials are logged.
    # hboTrialList == hbrTrialList at this point (enforced by the paranoia check above),
    # but HbO and HbR are imputed independently since each has its own FPCA structure.
    if useTrialImputation:
        hboInfRegionObj, hboFullTrialList, hboImputed = auxFuncChPt.imputeMissingTrialsFPCA(
            hboInfRegionObj, hboTrialList, firstTrial, lastTrial,
            minObservedTrials=minObservedTrialsForImputation,
            varThresh=imputationVarThresh, maxComponents=maxImputationComponents
        )
        hbrInfRegionObj, hbrFullTrialList, hbrImputed = auxFuncChPt.imputeMissingTrialsFPCA(
            hbrInfRegionObj, hbrTrialList, firstTrial, lastTrial,
            minObservedTrials=minObservedTrialsForImputation,
            varThresh=imputationVarThresh, maxComponents=maxImputationComponents
        )
        # both chromophores were aligned to the same trial list beforehand, so the
        # reconstructed trial lists must match, or the two chromophores would no
        # longer correspond trial-for-trial in the HbT/HbDiff sum below
        if hboFullTrialList != hbrFullTrialList:
            print(f"[WARNING] Imputed trial lists diverged between HbO/HbR — Infant: {infant}, "
                  f"Age: {age}, Region: {regionType} {regionValue}. Skipping imputation for this combo.")
        else:
            for t in sorted(set(hboImputed) | set(hbrImputed)):
                imputedTrialsLog.append({
                    'Infant': infant, regionType: regionValue, 'Age': age,
                    'Chromophore': 'Combined', 'Trial': t
                })
            hboTrialList, hbrTrialList = hboFullTrialList, hbrFullTrialList

    totalInfantTrialCoefficients = []
    diffInfantTrialCoefficients = []
    for i in range(hboInfRegionObj.coefficients.shape[0]):
        totalInfantTrialCoefficients.append(hboInfRegionObj.coefficients[i] + hbrInfRegionObj.coefficients[i])
        diffInfantTrialCoefficients.append(hboInfRegionObj.coefficients[i] - hbrInfRegionObj.coefficients[i])

    totalInfantTrialChromObj = skfda.representation.basis.FDataBasis(basis, totalInfantTrialCoefficients)
    diffInfantTrialObj = skfda.representation.basis.FDataBasis(basis, diffInfantTrialCoefficients)

    filteredTotalRows = auxFuncChPt.processChangePoints(
                totalInfantTrialChromObj, age, 3, firstTrial, lastTrial,
                firstFunc, lastFunc, cptSignificance, lrCovLag, 
                useLongRun, useWildBinSeg, regionType, regionValue
            )   
    for row in filteredTotalRows:
        row['Infant'] = infant

    filteredDifferenceRows = auxFuncChPt.processChangePoints(
                diffInfantTrialObj, age, 4, firstTrial, lastTrial,
                firstFunc, lastFunc, cptSignificance, lrCovLag, 
                useLongRun, useWildBinSeg, regionType, regionValue
            )
    for row in filteredDifferenceRows:
        row['Infant'] = infant

    indChangepointRows.extend(filteredTotalRows)

    indChangepointRows.extend(filteredDifferenceRows)

    # Optionally restrict which chromophores are analysed individually for this
    # specific infant x region x age call, e.g. to subset to combinations already
    # flagged by an external analysis (TFCE). Combined HbT/HbDiff above is unaffected.
    chromsToRun = uniqueChroms if restrictChroms is None else [c for c in uniqueChroms if c in set(restrictChroms)]

    for chrom in chromsToRun:
        chromDf = subsetDf[(subsetDf['Infant'] == infant) & (subsetDf[regionType] == regionValue) & (subsetDf['Chrom'] == chrom)]
        chromTs, chromTrialList, chromValid = fNIRSData.extractTrialData(tsDataset, chromDf)

        if not chromValid:
            excludedChromValidLog.append({'Infant': infant, regionType: regionValue, 'Chrom': chrom, 'Age': age})
            continue

        if (not chromTrialList[0] in [1, 2]) or (not len(chromTrialList) > 20):
            excludedChromTrialsLog.append({'Infant': infant, regionType: regionValue, 'Chrom': chrom, 'Age': age})
            continue

        minLen = min(len(ts) for ts in chromTs)
        fs, stimOnset = fNIRSData.determineSamplingParameters(minLen, firstSamp, lastSamp)
        chromTs = fNIRSData.prepareTimeSeries(chromTs, stimOnset)
        gridPoints = np.arange(0, lastSamp+2, abs(lastSamp+2)/chromTs.shape[1])
        chromInfRegionObj = fNIRSData.convertToFunctionalData(chromTs, gridPoints, numBasis)

        if useTrialImputation:
            chromInfRegionObj, chromFullTrialList, chromImputed = auxFuncChPt.imputeMissingTrialsFPCA(
                chromInfRegionObj, chromTrialList, firstTrial, lastTrial,
                minObservedTrials=minObservedTrialsForImputation,
                varThresh=imputationVarThresh, maxComponents=maxImputationComponents
            )
            for t in chromImputed:
                imputedTrialsLog.append({
                    'Infant': infant, regionType: regionValue, 'Age': age,
                    'Chromophore': chrom, 'Trial': t
                })

        chromInfantTrialChromObj = skfda.representation.basis.FDataBasis(basis, chromInfRegionObj.coefficients)

        filteredChromRows = auxFuncChPt.processChangePoints(
            chromInfantTrialChromObj, age, chrom, firstTrial, lastTrial,
            firstFunc, lastFunc, cptSignificance, lrCovLag, 
            useLongRun, useWildBinSeg, regionType, regionValue
        )
        for row in filteredChromRows:
            row['Infant'] = infant

        indChangepointRows.extend(filteredChromRows)

    return indChangepointRows, excludedCombinedValidLog, excludedCombinedTrialsLog, excludedChromValidLog, excludedChromTrialsLog, imputedTrialsLog

########################################################################################

def processGroupRegion(age, uniqueInfants, uniqueChroms,
                        subsetDf, tsDataset, numBasis,
                        firstTrial, lastTrial, firstSamp, lastSamp,
                        firstFunc, lastFunc, cptSignificance, lrCovLag,
                        useLongRun, useWildBinSeg, fNIRSData, auxFuncChPt,
                        regionType, regionValue):
    """
    Runs region-based change point detection for group data (Channel, Tile or Parcel).
    Allows for parallelisation of the region/infant loop to speed up processing.
    
    Parameters:
        regionType (str): either 'Tile' or 'Parcel'
        regionValue (int): the value for the specific Tile or Parcel
    Returns:
        rows (list): list of detected change points
         chromCoeffs (dict):    Dictionary with keys from uniqueChroms (e.g., 1, 2),
                                each mapping to a NumPy array of coefficients (or None).
    """
    rows = []
    chromCoeffs = {} #save coefficients for construction again during statistical texting

    # Initialise dictionaries to hold per-trial coefficients
    totalGroupTrialCoefficients = {t: [] for t in range(firstTrial, lastTrial + 1)}
    differenceGroupTrialCoefficients = {t: [] for t in range(firstTrial, lastTrial + 1)}

    # Loop through all infants for this region
    for infant in uniqueInfants:

        if regionType == 'SourceDetector':
            hboDf = subsetDf[(subsetDf['Infant'] == infant) &
                            (subsetDf['Source'] == regionValue[0]) &
                            (subsetDf['Detector'] == regionValue[1]) &
                            (subsetDf['Chrom'] == 1) &
                            (subsetDf['Trial'] <= lastTrial)] 
            hbrDf = subsetDf[(subsetDf['Infant'] == infant) &
                            (subsetDf['Source'] == regionValue[0]) &
                            (subsetDf['Detector'] == regionValue[1]) &
                            (subsetDf['Chrom'] == 2) &
                            (subsetDf['Trial'] <= lastTrial)] 
        else:
            hboDf = subsetDf[(subsetDf['Infant'] == infant) &
                            (subsetDf[regionType] == regionValue) &
                            (subsetDf['Chrom'] == 1) &
                            (subsetDf['Trial'] <= lastTrial)] 
            hbrDf = subsetDf[(subsetDf['Infant'] == infant) &
                            (subsetDf[regionType] == regionValue) &
                            (subsetDf['Chrom'] == 2) &
                            (subsetDf['Trial'] <= lastTrial)] 

        hboDfCommon, hbrDfCommon = fNIRSData.getCommonTrialsData(hboDf, hbrDf)
        hboTs, hboTrialList, hboValid = fNIRSData.extractTrialData(tsDataset, hboDfCommon)
        hbrTs, hbrTrialList, hbrValid = fNIRSData.extractTrialData(tsDataset, hbrDfCommon)

        if not (hboValid and hbrValid):
            continue

        # Count trial occurrences
        hboCounts = Counter(hboTrialList)
        hbrCounts = Counter(hbrTrialList)

        commonTrials = set(hboCounts.keys()).intersection(hbrCounts.keys())
        trialKeepCounts = {trial: min(hboCounts[trial], hbrCounts[trial]) for trial in commonTrials}

        hboIdxSorted = getIndicesForTrials(hboTrialList, trialKeepCounts)
        hbrIdxSorted = getIndicesForTrials(hbrTrialList, trialKeepCounts)

        # Apply to data
        hboTs = hboTs[hboIdxSorted]
        hbrTs = hbrTs[hbrIdxSorted]
        hboTrialList = [hboTrialList[i] for i in hboIdxSorted]
        hbrTrialList = [hbrTrialList[i] for i in hbrIdxSorted]

        # final paranoia check...
        if hboTs.shape[0] != hbrTs.shape[0] or hboTrialList != hbrTrialList:
            print(f"[WARNING] Trial alignment mismatch after indexing — Infant: {infant}, Age: {age}, Region: {regionType} {regionValue}")
            # Optionally: truncate to min length here if absolutely necessary, but better to fix upstream
            minTrials = min(hboTs.shape[0], hbrTs.shape[0])
            hboTs = hboTs[:minTrials]
            hbrTs = hbrTs[:minTrials]
            hboTrialList = hboTrialList[:minTrials]
            hbrTrialList = hbrTrialList[:minTrials]

        minLen = min(len(ts) for ts in hboTs)
        fs, stimOnset = fNIRSData.determineSamplingParameters(minLen, firstSamp, lastSamp)

        hboTs = fNIRSData.prepareTimeSeries(hboTs, stimOnset)
        hbrTs = fNIRSData.prepareTimeSeries(hbrTs, stimOnset)

        gridPoints = np.arange(0, lastSamp + 2, abs(lastSamp + 2) / hboTs.shape[1])
        hboObj = fNIRSData.convertToFunctionalData(hboTs, gridPoints, numBasis)
        hbrObj = fNIRSData.convertToFunctionalData(hbrTs, gridPoints, numBasis)

        for i in range(hboTs.shape[0]):
            trialNum = hbrTrialList[i]
            totalGroupTrialCoefficients[trialNum].append(hboObj.coefficients[i] + hbrObj.coefficients[i])
            differenceGroupTrialCoefficients[trialNum].append(hboObj.coefficients[i] - hbrObj.coefficients[i])

    # Process total (HbT) and difference
    for coeffDict, chrom in zip([totalGroupTrialCoefficients, differenceGroupTrialCoefficients], [3, 4]):
        # for storing mean number of participants whose data was used for CPt calculation
        # for weighting in statistical analysis
        totalRows = sum(len(v) for v in coeffDict.values())
        numTrials = sum(1 for v in coeffDict.values() if len(v) > 0)

        meanCoeffs = auxFuncChPt.calculateMeanCoefficients(coeffDict)
        validTrials = [t for t in range(firstTrial, lastTrial + 1) if t in meanCoeffs]  # only include trials that have valid mean coefficients
        allCoeffs = [meanCoeffs[t] for t in validTrials]
        if not allCoeffs:
            print(f"[WARNING] Skipping chrom {chrom} for {regionType} {regionValue}, age {age}: no valid coefficients")
            continue
        basisGroup = BSplineBasis(n_basis=numBasis)
        combObj = FDataBasis(basisGroup, allCoeffs)
        filteredChromRows =auxFuncChPt.processChangePoints(
            combObj, age, chrom, firstTrial, lastTrial,
            firstFunc, lastFunc, cptSignificance, lrCovLag,
            useLongRun, useWildBinSeg, regionType, regionValue
        )
        for row in filteredChromRows:
            row['NumInfants'] = totalRows / numTrials if numTrials > 0 else 0
        rows.extend(filteredChromRows)

    # Process each chromophore independently (HbO and HbR)
    for chrom in uniqueChroms:
        groupCoeffs = fNIRSData.computeGroupCoefficients(
            uniqueInfants, subsetDf, tsDataset, chrom,
            firstTrial, lastTrial, firstSamp, lastSamp, numBasis,
            regionType, regionValue
        )
        # for storing mean number of participants whose data was used for CPt calculation
        # for weighting in statistical analysis
        totalRows = sum(len(v) for v in groupCoeffs.values())
        numTrials = sum(1 for v in groupCoeffs.values() if len(v) > 0)

        meanCoeffs = auxFuncChPt.calculateMeanCoefficients(groupCoeffs)
        validTrials = [t for t in range(firstTrial, lastTrial + 1) if t in meanCoeffs]  # only include trials that have valid mean coefficients
        allCoeffs = [meanCoeffs[t] for t in validTrials]
        if not allCoeffs:
            print(f"[WARNING] Skipping chrom {chrom} for {regionType} {regionValue}, age {age}: no valid coefficients")
            chromCoeffs[chrom] = None
            continue
        chromCoeffs[chrom] = np.array(allCoeffs)
        basisGroup = BSplineBasis(n_basis=numBasis)
        fdObj = FDataBasis(basisGroup, allCoeffs)
        filteredChromRows =auxFuncChPt.processChangePoints(
            fdObj, age, chrom, firstTrial, lastTrial,
            firstFunc, lastFunc, cptSignificance, lrCovLag,
            useLongRun, useWildBinSeg, regionType, regionValue
        )
        for row in filteredChromRows:
            row['NumInfants'] = totalRows / numTrials if numTrials > 0 else 0
        rows.extend(filteredChromRows)

    return rows, chromCoeffs

########################################################################################


def findCPtRegionsGroup(age, uniqueInfants, uniqueChroms,
                        subsetDf, tsDatasetFilepath, numBasis,
                        firstTrial, lastTrial, firstSamp, lastSamp,
                        firstFunc, lastFunc, cptSignificance, lrCovLag,
                        useLongRun, useWildBinSeg,
                        regionType, regionValue):
    """
    Looks for first change point in a region/chrom combination for group data (Channel, Tile or Parcel).
    Allows for parallelisation of the region/infant loop to speed up processing.
    
    Parameters:
        regionType (str): either 'Tile' or 'Parcel'
        regionValue (int): the value for the specific Tile or Parcel
    Returns:
        rows (list): list of first  detected change points per age/region/chrom combination

    """
    rows = []

    with h5py.File(tsDatasetFilepath, 'r') as h5f:
        tsDataset = h5f['timeSeries'][()].copy()

        # Process each chromophore independently (HbO and HbR)
        for chrom in uniqueChroms:
            groupCoeffs = computeGroupCoefficients(
                uniqueInfants, subsetDf, tsDataset, chrom,
                firstTrial, lastTrial, firstSamp, lastSamp, numBasis,
                regionType, regionValue
            )
            # for storing mean number of participants whose data was used for CPt calculation
            # for weighting in statistical analysis
            totalRows = sum(len(v) for v in groupCoeffs.values())
            numTrials = sum(1 for v in groupCoeffs.values() if len(v) > 0)

            meanCoeffs = calculateMeanCoefficients(groupCoeffs)
            validTrials = [t for t in range(firstTrial, lastTrial + 1) if t in meanCoeffs]  # only include trials that have valid mean coefficients
            allCoeffs = [meanCoeffs[t] for t in validTrials]
            basisGroup = BSplineBasis(n_basis=numBasis)
            fdObj = FDataBasis(basisGroup, allCoeffs)
            chromRows =findGroupRegionFirstChangePoints(
                fdObj, age, chrom, firstTrial, lastTrial,
                firstFunc, lastFunc, cptSignificance, lrCovLag,
                useLongRun, useWildBinSeg, regionType, regionValue
            )
            for row in chromRows:
                row['NumInfants'] = totalRows / numTrials if numTrials > 0 else 0
            rows.extend(chromRows)

    return rows

##################################################################################################################

def checkGroupCriteria(group_key, allCorrelations, allCovariances, 
                       correlationThreshold=0.6, covarianceThreshold=1.0):

    """
    Check if a group meets all the filtering criteria using ALL blocks
    Used when longitudinally analysing change point data
    """
    if group_key not in allCorrelations or group_key not in allCovariances:
        return False
    
    correlations = np.array(allCorrelations[group_key])
    covariances = np.array(allCovariances[group_key])
    
    # Remove NaN values (just in case)
    validCorr = correlations[~np.isnan(correlations)]
    validCov = covariances[~np.isnan(covariances)]
    
    if len(validCorr) == 0 or len(validCov) == 0:
        return False
    
    # Apply filtering criteria to all blocks
    highCorrelation = (np.abs(validCorr) > correlationThreshold).all()
    significantCovariance = (np.abs(validCov) > covarianceThreshold).all()
    consistentCorrDirection = len(np.unique(np.sign(validCorr))) == 1
    consistentCovDirection = len(np.unique(np.sign(validCov))) == 1
    
    return highCorrelation and significantCovariance 
    # and consistentCorrDirection and consistentCovDirection

#################################################################################################################

def checkGroupCriteriaTEST(group_key, allCorrelations, allCovariances, 
                       correlationThreshold=0.6, covarianceThreshold=1.0):
    """
    Check if a group meets all the filtering criteria using ALL blocks.
    Used when longitudinally analysing change point data.
    Returns True if at least one index satisfies BOTH correlation and covariance criteria.
    """
    if group_key not in allCorrelations or group_key not in allCovariances:
        return False

    correlations = np.array(allCorrelations[group_key])
    covariances = np.array(allCovariances[group_key])

    if len(correlations) != len(covariances):
        return False

    # Build boolean masks over the full arrays (NaN → False naturally via comparison)
    highCorrelation       = np.abs(correlations) > correlationThreshold
    significantCovariance = np.abs(covariances)  > covarianceThreshold

    # At least one index must satisfy BOTH criteria simultaneously
    return np.any(highCorrelation & significantCovariance)
    # and consistentCorrDirection and consistentCovDirection