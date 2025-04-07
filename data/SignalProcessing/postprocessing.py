from typing import List
import numpy as np
from scipy.signal import butter, lfilter, find_peaks, windows, savgol_coeffs
from scipy.ndimage import convolve1d
from .dataPreparation import spikeListToSpikeArray, getResponseRaw
from itertools import combinations
from functools import reduce

class Filter():
    """
    A filter to preprocess the raw data.
    Has implementations of the highpass/lowpass butterworth filter and the savgol polynomial fit.
    """
    def __init__(self, filterType: str = "highpass", device: str = 'cpu', cutOffFrequency: float = 200):
        sampleFrequency = 20000
        self.device = device
        if filterType == "highpass":
            filterOrder = 2
            cutOffDiscrete = cutOffFrequency / sampleFrequency * 2
            self.coeffB, self.coeffA = butter(filterOrder, cutOffDiscrete, btype="highpass")
            self.filterType = "butter"
        elif filterType == "lowpass":
            filterOrder = 2
            cutOffDiscrete = cutOffFrequency / sampleFrequency * 2
            self.coeffB, self.coeffA = butter(filterOrder, cutOffDiscrete, btype="low")
            self.filterType = "butter"
        elif filterType == "savgol":
            windowLength = 5
            polyOrder = 2
            self.kernel = savgol_coeffs(windowLength, polyOrder)
            self.filterType = "savgol"
        else:
            raise Exception("Filter type not supported.")

    def __call__(self, trace: np.ndarray):
        if self.filterType == "butter":
            return lfilter(self.coeffB,self.coeffA,trace.astype(float))
        elif self.filterType == "butter":
            return lfilter(self.coeffB,self.coeffA,trace.astype(float))
        elif self.filterType == "savgol":
            return convolve1d(trace.astype(float), self.kernel, mode="reflect")

class PeakDetection():
    """
    A class to do peak detection with.
    """
    def __init__(self, spikeDistance: int = 30, spikeThreshold: float = 5, useStd: bool = True, peakType: str = "abs"):
        self.spikeDistance = spikeDistance
        self.spikeThreshold = spikeThreshold
        self.useStd = useStd
        self.peakType = peakType
    def __call__(self, trace: np.ndarray):
        trace = np.squeeze(trace)
        if trace.ndim > 1:
            raise Exception("Function “detectSpikes“ only takes one dimensional input.")
        if self.useStd:
            height = self.spikeThreshold * np.std(trace)
        else:
            height = self.spikeThreshold
        if self.peakType == "abs":
            indices = find_peaks(
                np.abs(trace),
                height=height,
                distance=self.spikeDistance,
            )[0]
        elif self.peakType == "min":
            indices = find_peaks(
                -trace,
                height=height,
                distance=self.spikeDistance,
            )[0]
        else:
            raise Exception("Invalid peak type.")
        peaks = trace[indices]
        return indices, peaks

class SNEO():
    """
    Non-linear energy operator, which uses the “energy“ of the signal to detect spikes. Has optional peak correction.
    """
    def __init__(self, spikeDistance: int = 30, spikeThreshold: float = 20, k: int = 3, minimalThreshold: float = 5.0e-8,
                 peakCorrection: str = None, offsetCorrectionWidth: int = 14, medianEstimationLegnth: int = 10,
                 overlap: int = 3, dataSmooth = Filter(filterType="savgol")):
        self.spikeDistance = spikeDistance
        self.spikeThreshold = spikeThreshold
        self.k = k
        self.minimalThreshold = minimalThreshold
        self.kernel = windows.hamming(4*k+1)
        self.peakCorrection = peakCorrection
        self.offsetCorrectionWidth = offsetCorrectionWidth
        self.medianEstimationLegnth = medianEstimationLegnth
        self.overlap = overlap
        self.dataSmooth = dataSmooth
    def __call__(self, trace: np.ndarray, debug: bool=False):
        traceSNEO = np.copy(np.squeeze(trace))
        if self.dataSmooth is not None:
            traceSNEO = self.dataSmooth(traceSNEO)
        traceSNEO = np.pad(traceSNEO, pad_width=(self.k, self.k))
        traceSNEO = np.square(traceSNEO[self.k:-self.k]) - traceSNEO[..., 0:-2 * self.k] * traceSNEO[..., 2 * self.k:]
        traceSNEO = convolve1d(traceSNEO, self.kernel, mode="reflect")
        height = self.spikeThreshold * np.median(np.abs(traceSNEO))
        height = max(height, self.minimalThreshold)
        indices = find_peaks(
            traceSNEO,
            height=height,
            distance=self.spikeDistance,
        )[0]
        if self.peakCorrection == "min":
            indices += np.argmin(
                getResponseRaw(trace[None], indices, (-self.offsetCorrectionWidth, self.offsetCorrectionWidth))[:, 0],
                axis=1) - self.offsetCorrectionWidth
            indices = indices.clip(0, len(trace) - 1)
        elif self.peakCorrection == "abs":
            cutOuts = getResponseRaw(trace[None], indices, (-self.offsetCorrectionWidth - self.medianEstimationLength + self.overlap,
                                                            self.offsetCorrectionWidth + self.medianEstimationLength - self.overlap))
            mask = np.arange(self.medianEstimationLength * 2)
            mask[self.medianEstimationLength:] += cutOuts.shape[-1] - 2 * self.medianEstimationLength
            cutOuts = cutOuts - np.median(cutOuts[..., mask], axis=-1, keepdims=True)
            indices += np.argmax(
                np.abs(cutOuts[:, 0, self.medianEstimationLength - self.overlap:-self.medianEstimationLength + self.overlap]),
                axis=1) - self.offsetCorrectionWidth
            indices = indices.clip(0, len(trace) - 1)
        peaks = trace[indices]
        if debug:
            return indices, peaks, traceSNEO
        return indices, peaks

class UltimateArtefactBlanking():
    """
    A class to blank the artefact. It roughly proceeds the following way:
    blankTimings indicate where a stimulation happened. The algorithm finds the at maximum three peak shaped signal
    (which occurs after high pass filtering) around the stimulation timing. For this it uses the filtered average and
    does it for each electrode separately. The duration of said shape is then interpolated linearly in the raw signal.
    """
    def __init__(self,
        timeWindowThresh: float = 0.1,
        maxBlanking = (-5,30),
        blankingBuffer = (0,0),
        artefactPattern: (np.ndarray, np.ndarray) = (np.array([1, 0, -1, 0, 1, 0]), np.array([-1, 0, 1, 0, -1, 0])),
        artefactPatternMaxSteps: np.ndarray = np.array([8,15,2,15,2,15,2])
        #artefactPattern: (np.ndarray,np.ndarray) = (np.array([1,0,-1,0,1,0,-1,0]),np.array([-1,0,1,0,-1,0,1,0])), # This input would be for a 3 peak artefact
        #artefactPatternMaxSteps: np.ndarray = np.array([8,15,2,15,2,15,2,15,2])
    ):
        self.timeWindowThresh = timeWindowThresh
        self.maxBlanking = maxBlanking
        self.blankingBuffer = blankingBuffer
        self.artefactPattern = artefactPattern
        self.artefactPatternMaxSteps = artefactPatternMaxSteps
        self.filterHigh = Filter(cutOffFrequency=20)

    def __call__(self, traces: np.ndarray, blankTimings: List[np.ndarray], debug: bool = False) -> (np.ndarray, np.ndarray):
        if debug:
            outString = []

        maxLength = max([len(t) for t in blankTimings])
        blankTimingsArray = np.zeros((len(blankTimings),maxLength)) + np.nan
        for i,t in enumerate(blankTimings):
            blankTimingsArray[i,:len(t)] = t

        # We find how many stimulation pulses occured at a specific time simulatenously.
        # Depending on how many stimulations were active, the shape changes.
        # One entry in the list blankTimings contains the array of stimulation timings of one input
        allBlankTimesUnique, countsPerTiming = np.unique(blankTimingsArray, return_counts=True)
        keys = np.searchsorted(allBlankTimesUnique, blankTimingsArray)
        countsPerTiming = np.take(countsPerTiming, keys) # Here in the end have the number of stimuli occuring for each stimulation timing

        for e in range(len(traces)):
            artefactDetectSignal = self.filterHigh(traces[e])
            tracesNAN = np.zeros_like(traces[e])
            for k in range(1,len(blankTimings)+1):
                timingsOfInterest = [t[np.logical_and(countsPerTiming[i]==k,~np.isnan(t))].astype(int) for i,t in enumerate(blankTimingsArray)]
                for subset in combinations(timingsOfInterest, k):
                    if min(len(s) for s in subset)==0:
                        continue
                    currentTimings = reduce(np.intersect1d, subset)
                    margin = 20
                    averageSignal = np.median(getResponseRaw(artefactDetectSignal[None],currentTimings,(self.maxBlanking[0]-margin,self.maxBlanking[1]+margin)),axis=(0,1))
                    meanTrace = np.median(averageSignal)
                    averageSignal = averageSignal[margin:-margin]
                    threshold = np.median(np.absolute(averageSignal - meanTrace))*self.timeWindowThresh
                    threshold = (meanTrace+threshold,meanTrace-threshold)
                    averageSignalNew = np.zeros_like(averageSignal)
                    averageSignalNew[averageSignal > threshold[0]] = 1
                    averageSignalNew[averageSignal < threshold[1]] = -1
                    averageSignalNew[1:][np.abs(np.diff(averageSignalNew))>1] = 0 # To force "continuous" signal
                    # Check for artefact shape
                    argwhere_1 = np.argwhere(averageSignalNew == 1)
                    argwhere_minus1 = np.argwhere(averageSignalNew == -1)
                    min_index_1 = np.min(argwhere_1) if argwhere_1.size > 0 else np.inf
                    min_index_minus1 = np.min(argwhere_minus1) if argwhere_minus1.size > 0 else np.inf
                    arg = np.argmin([min_index_1, min_index_minus1])

                    subSequence = self.artefactPattern[arg]
                    blankingWindowTemp = np.asarray(self.findSplitSequenceToMax(averageSignalNew, subSequence, self.artefactPatternMaxSteps)) + self.maxBlanking[0]
                    blankingWindowTemp = blankingWindowTemp[~np.isnan(blankingWindowTemp)].astype(int)
                    blankingWindow = np.zeros(len(self.artefactPattern[0])+1,dtype=int) + np.max(blankingWindowTemp)
                    blankingWindow[:len(blankingWindowTemp)] = blankingWindowTemp

                    if len(blankingWindowTemp) > 1:
                        time_indices = np.arange(blankingWindow[0]+self.blankingBuffer[0], blankingWindow[-1]+self.blankingBuffer[1], 1) + np.atleast_2d(currentTimings).T
                        time_indices = time_indices[time_indices < traces.shape[1]]
                        time_indices = time_indices[time_indices > 0]
                        tracesNAN[np.unique(time_indices.flatten())] = np.nan



                        if debug:
                            outString.append(f"({e}:{blankingWindow})\n")
                    elif debug:
                        outString.append(f"({e}: None, {len(currentTimings)})\n")

            # Get the indices of the non-NaN values
            goodIndices = np.where(~np.isnan(tracesNAN))[0]
            # Get the non-NaN values
            goodValues = traces[e, goodIndices]
            # Interpolate the NaN values
            traces[e] = np.interp(np.arange(len(traces[e])), goodIndices, goodValues)
        if debug:
            print(''.join(outString))
        return traces


    def findSplitSequenceToMax(self, sequence: np.ndarray, subSequence: np.ndarray, subSequenceMaxSteps: np.ndarray):

        if np.isin(subSequence[0],sequence[:subSequenceMaxSteps[0]]):
            arg = np.argwhere(sequence==subSequence[0])[0,0]
            if len(subSequence) > 1:
                return [arg]+[arg+s for s in self.findSplitSequenceToMax(sequence[arg:],subSequence[1:],subSequenceMaxSteps[1:])]
            else:
                cond = sequence[arg:arg+subSequenceMaxSteps[1]] != subSequence[0]
                if np.any(cond):
                    argLast = np.min(np.argwhere(cond))
                else:
                    argLast = len(sequence[arg:arg+subSequenceMaxSteps[1]])-1
                return [arg,argLast+arg]
        else:
            return [len(sequence[:subSequenceMaxSteps[0]])-1]