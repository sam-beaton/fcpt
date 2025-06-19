import numpy as np

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