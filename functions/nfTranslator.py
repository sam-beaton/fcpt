"""
nfTranslator.py

Functions for translating NF (No Frontal) cap Source and Detector values 
to regular cap configurations.
"""

import pandas as pd
import random
from typing import Tuple, Optional

# Define the tile mappings
nfSourceTile = {
    1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 
    4: [34, 35, 36], 5: [31, 32, 33], 6: [28, 29, 30], 
    7: [16, 17, 18], 8: [13, 14, 15], 9: [10, 11, 12], 
    10: [19, 20, 21], 11: [22, 23, 24], 12: [25, 26, 27]
}

nfDetTile = {
    1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12], 
    4: [45, 46, 47, 48], 5: [41, 42, 43, 44], 6: [37, 38, 39, 40], 
    7: [21, 22, 23, 24], 8: [17, 18, 19, 20], 9: [13, 14, 15, 16], 
    10: [25, 26, 27, 28], 11: [29, 30, 31, 32], 12: [33, 34, 35, 36]
}

regularSourceTile = {
    1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 
    4: [43, 44, 45], 5: [40, 41, 42], 6: [37, 38, 39], 
    7: [22, 23, 24], 8: [19, 20, 21], 9: [16, 17, 18], 
    10: [25, 26, 27], 11: [28, 29, 30], 12: [31, 32, 33], 
    13: [10, 11, 12], 14: [34, 35, 36], 15: [13, 14, 15]
}

regularDetTile = {
    1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12], 
    4: [57, 58, 59, 60], 5: [53, 54, 55, 56], 6: [49, 50, 51, 52], 
    7: [29, 30, 31, 32], 8: [25, 26, 27, 28], 9: [21, 22, 23, 24], 
    10: [33, 34, 35, 36], 11: [37, 38, 39, 40], 12: [41, 42, 43, 44], 
    13: [13, 14, 15, 16], 14: [45, 46, 47, 48], 15: [17, 18, 19, 20]
}


def findTileAndPosition(value, tileDict):

    """Find which tile a value belongs to and its position within that tile"""

    for tileNum, values in tileDict.items():
        if value in values:
            position = values.index(value)
            return tileNum, position
    return None, None


def translateNfValues(source, detector):

    
    # Translate NF source and detector values to regular values.
    
    # Inputss:
    #     source: NF source value
    #     detector: NF detector value
        
    # Returns: Tuple of (newSource, newDetector)

    # f tile and position for source
    sourceTile, sourcePos = findTileAndPosition(source, nfSourceTile)
    if sourceTile is None:
        raise ValueError(f"Source {source} not found in NF mapping")
    
    # Find tile and position for detector
    detTile, detPos = findTileAndPosition(detector, nfDetTile)
    if detTile is None:
        raise ValueError(f"Detector {detector} not found in NF mapping")
    
    # Get new values from regular mapping
    newSource = regularSourceTile[sourceTile][sourcePos]
    newDetector = regularDetTile[detTile][detPos]
    
    return newSource, newDetector


def translateNfDataframe(df, inPlace=False):

    """
    Translate NF values in a pandas DataFrame.
    
    Args:
        df: DataFrame containing the data
        inPlace: Whether to modify the original DataFrame (default: False)
        
    Returns:
        DataFrame with translated values (or None if inPlace=True)
    """
    if not inPlace:
        df = df.copy()
    
    # Find NF rows
    nfMask = df['Cap name'].str.endswith('_NF')
    nfCount = nfMask.sum() 
    
    if nfCount == 0:
        print("No NF caps")
        return df if not inPlace else None
    
    print(f"Found {nfCount} rows with _NF cap names")
    
    # apply translation
    for idx in df[nfMask].index:
        try:
            oldSource = df.loc[idx, 'Source']
            oldDetector = df.loc[idx, 'Detector']
            newSource, newDetector = translateNfValues(oldSource, oldDetector)
            df.loc[idx, 'Source'] = newSource
            df.loc[idx, 'Detector'] = newDetector
        except ValueError as e:
            print(f"Warning: Row {idx} - {e}")
    
    return df if not inPlace else None


def verifyNfTranslation(df, nSamples=10):
    """
    Show before/after comparison for random sample of NF rows.
    
    """
    nfMask = df['Cap name'].str.endswith('_NF')
    nfIndices = df[nfMask].index.tolist()
    
    if len(nfIndices) == 0:
        print("No NF rows found for verification")
        return pd.DataFrame()
    
    # sample
    random.seed(42)  # For reproducible results
    sampleSize = min(nSamples, len(nfIndices))
    sampleIndices = random.sample(nfIndices, sampleSize)
    sampleIndices.sort()
    
    # Get before values
    beforeData = df.loc[sampleIndices, ['Cap name', 'Source', 'Detector']].copy()
    
    # initialise
    afterSources = []
    afterDetectors = []
    
    #calculate after values
    for idx in sampleIndices:
        try:
            oldSource = df.loc[idx, 'Source']
            oldDetector = df.loc[idx, 'Detector']
            newSource, newDetector = translateNfValues(oldSource, oldDetector)
            afterSources.append(newSource)
            afterDetectors.append(newDetector)
        except ValueError as e:
            print(f"Warning: Row {idx} - {e}")
            afterSources.append(df.loc[idx, 'Source'])
            afterDetectors.append(df.loc[idx, 'Detector'])
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Index': sampleIndices,
        'CapName': beforeData['Cap name'].values,
        'SourceBefore': beforeData['Source'].values,
        'SourceAfter': afterSources,
        'DetectorBefore': beforeData['Detector'].values,
        'DetectorAfter': afterDetectors
    })
    
    print("Before/After Comparison:")
    print(comparison)
    return comparison