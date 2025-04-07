from typing import List

import numpy as np
from copy import deepcopy
#from .cythonBuilds.median_isi import compute_median_isi

def averageFrequencyResponsePerElectrode(responses: List[np.ndarray], responseStarts: List[np.ndarray], mappedChannels:np.ndarray,
                                         window: int = 200) -> List[np.ndarray]:
    """
    For each response array of the responses list, the spike count is calculated.
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param window: The response window at which to look at.
    :return: Returns a list of numpy arrays. Each array contains for each response the frequency of an electrode in the
    order of the channel mapping. There is one array per sweep value.
    """

    counts = []

    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:, np.where(delayVectors[1] < window)[0]])
        countsResponse = np.zeros((len(responseStarts[n]), len(mappedChannels)))
        if validSpikes.shape[1] != 0:
            validSpikes[0] = np.searchsorted(mappedChannels,validSpikes[0])
            np.add.at(countsResponse, (validSpikes[3].astype(int), validSpikes[0].astype(int)), 1)
        counts.append(countsResponse)
    return counts

def averageDelayResponse(responses: List[np.ndarray], responseStarts: List[np.ndarray], mappedChannels: np.ndarray,
                         window: int = 200) -> List[np.ndarray]:
    """
    For each response array of the responses list, the delay of the first spike averaged over the electrodes is
    calculated. If no spike is detected, the value of an electrode is set to the window length.
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param window: The response window at which to look at.
    :return: Returns a list of numpy arrays. Each array contains for each response the TTFS of an electrode in the
    order of the channel mapping. There is one array per sweep value.
    """

    firstDelays = []
    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:, np.where(delayVectors[1] < window)[0]])
        firstDelay = np.zeros((len(responseStarts[n]),len(mappedChannels)))+window / 20
        if validSpikes.shape[1] != 0:
            validSpikes[3] = validSpikes[3] + len(responseStarts[n])*validSpikes[0]
            sorted_indices = np.lexsort((validSpikes[1],validSpikes[3]))
            validSpikes = validSpikes[:, sorted_indices]
            validSpikes = validSpikes[:,np.concatenate(([0],1+np.where(np.diff(validSpikes[3]) != 0)[0]))]
            validSpikes[3] = validSpikes[3] % len(responseStarts[n])
            validSpikes[0] = np.searchsorted(mappedChannels,validSpikes[0])
            firstDelay[validSpikes[3].astype(int),validSpikes[0].astype(int)] = validSpikes[1]/ 20
        firstDelays.append(firstDelay)
    return firstDelays

def meanISIResponse(responses: List[np.ndarray], responseStarts: List[np.ndarray], mappedChannels: np.ndarray,
                         window: int = 200) -> List[np.ndarray]:
    """
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param window: The response window at which to look at.
    :return: Returns a list of numpy arrays. Each array contains for each response the average isi of an electrode in the
    order of the channel mapping. There is one array per sweep value.
    """
    mean_isis = []

    def compute_mean_isi(isi: np.ndarray, valid_isi: np.ndarray, n_responses:int, n_channels: int):
        # Preallocate the median ISI array
        mean_isi = np.zeros((n_responses, n_channels)) + np.nan
        # Compute the median ISI for each response-channel pair
        for i in range(n_responses):
            for j in range(n_channels):
                if len(isi[valid_isi == i + n_responses * j]) != 0:
                    mean_isi[i, j] = np.mean(isi[valid_isi == i + n_responses * j])
        return mean_isi

    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:, np.where(delayVectors[1] < window)[0]])
        if validSpikes.shape[1] != 0:
            validSpikes[3] = validSpikes[3] + len(responseStarts[n])*np.searchsorted(mappedChannels,validSpikes[0])
            sorted_indices = np.lexsort((validSpikes[1],validSpikes[3]))
            validSpikes = validSpikes[:, sorted_indices]
            isi = np.diff(validSpikes[1])[validSpikes[3][1:] == validSpikes[3][:-1]]
            valid_isi = validSpikes[3,1:][validSpikes[3][1:] == validSpikes[3][:-1]]
            mean_isi = compute_mean_isi(isi,valid_isi.astype(int),len(responseStarts[n]),len(mappedChannels))
            mean_isi[np.isnan(mean_isi)] = window
        else:
            mean_isi = np.zeros((len(responseStarts[n]), len(mappedChannels))) + window
        mean_isis.append(mean_isi/20)
    return mean_isis

def meanPhaseResponse(responses: List[np.ndarray], responseStarts: List[np.ndarray], mappedChannels: np.ndarray,
                         window: int = 200, periodicity: int = 200) -> List[np.ndarray]:
    """
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param window: The response window at which to look at.
    :return: Returns a list of numpy arrays. Each array contains for each response the average phase (eq. to average delay)
    of an electrode in the order of the channel mapping. There is one array per sweep value.
    """
    mean_phases = []

    def compute_avg_phase(delay: np.ndarray, valid_delay: np.ndarray, n_responses:int, n_channels: int):
        # Preallocate the median ISI array
        mean_phase = np.zeros((n_responses, n_channels)) + np.nan
        # Compute the median ISI for each response-channel pair
        for i in range(n_responses):
            for j in range(n_channels):
                if len(delay[valid_delay == i + n_responses * j]) != 0:
                    phases = (delay[valid_delay == i + n_responses * j] % periodicity) / periodicity * 2 * np.pi
                    mean_phase[i, j] = np.arctan2(np.sum(np.sin(phases)), np.sum(np.cos(phases))) % (2 * np.pi)
        return mean_phase

    phases = ((periodicity-1) % periodicity) / periodicity * 2 * np.pi
    nan_value = np.arctan2(np.sum(np.sin(phases)), np.sum(np.cos(phases))) % (2 * np.pi)
    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:, np.where(delayVectors[1] < window)[0]])
        if validSpikes.shape[1] != 0:
            validSpikes[3] = validSpikes[3] + len(responseStarts[n])*np.searchsorted(mappedChannels,validSpikes[0])
            sorted_indices = np.lexsort((validSpikes[1],validSpikes[3]))
            validSpikes = validSpikes[:, sorted_indices]
            mean_phase = compute_avg_phase(validSpikes[1], validSpikes[3].astype(int), len(responseStarts[n]), len(mappedChannels))
            mean_phase[np.isnan(mean_phase)] = nan_value
        else:
            mean_phase = np.zeros((len(responseStarts[n]), len(mappedChannels))) + nan_value
        mean_phases.append(mean_phase)
    return mean_phases

def stack_arrays(arr_list: np.ndarray, fixed_n: int) -> np.ndarray:
    """
    Combines individual responses from the delay/frequency response function into one numpy array
    :param arr_list: List returned by the aforementioned functions
    :param fixed_n: Number of responses kept to adjust for different n_responses
    :return: Array of shape (cardinality_input, fixed_n, n_channels)
    """
    # Create an empty list to store the processed arrays
    processed_arrays = []
    for arr in arr_list:
        # If the array has more than 'fixed_n' indices, select a random subset
        if arr.shape[0] > fixed_n:
            selected_indices = np.sort(np.random.choice(arr.shape[0], fixed_n, replace=False))
            processed_arr = arr[selected_indices]
        # If the array has less than 'fixed_n' indices, pad it with zeros
        else:
            processed_arr = arr
        processed_arrays.append(processed_arr)
    # Stack the processed arrays along a new axis
    stacked_arr = np.stack(processed_arrays,axis=0)

    return stacked_arr


def responseDelayOverTime(
        responses: List[np.ndarray],
        responseStarts: List[np.ndarray],
        electrodeChannelMapping: np.ndarray,
        colourCoding,
        window:int=200,
        nrOfBinsDelay: int = None,
        nrOfBinsInput: int = None
):
    """
    Create an image of a raster plot as a numpy array of shape (H,W,3).
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param electrodeChannelMapping: A (2,n_electrodes) numpy array with the first dimension corresponding to (electrode,channel)
    :param colourCoding: A class that, when called, maps an array of electrodes to colours.
    :param window: The response window at which to look at.
    :param nrOfBins: In how many bins the responses are sorted. Default is the number of samples in the window.
    :return: A numpy array resembling an image of shape (H,W,3).
    """
    if nrOfBinsDelay is None:
        nrOfBinsDelay = window
    if nrOfBinsInput is None:
        nrOfBinsInput = sum([len(r) for r in responseStarts])
    else:
        responseStarts = [(np.copy(r) - np.min(r)) for r in responseStarts]
        binSize = np.ceil(sum(np.max(r) for r in responseStarts) / (nrOfBinsInput))
        responses = deepcopy(responses)
        for r in responses:
            r[3] = responseStarts[r[3]] // binSize


    image = np.zeros((nrOfBinsInput, nrOfBinsDelay, 3))
    avgToTake = np.zeros(image.shape)
    index = 0
    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:,np.where(delayVectors[1]<window)[0]])
        if validSpikes.shape[1] != 0:
            responseIndex = (validSpikes[3] + index).astype(int)
            delayIndex = np.round(validSpikes[1]*nrOfBinsDelay/window).astype(int)
            electrodes = electrodeChannelMapping[0, np.where(electrodeChannelMapping[1] == validSpikes[0][:, None])[1]]
            colours = colourCoding(electrodes)
            image[responseIndex,delayIndex] = colours[np.newaxis,:,:3]
            avgToTake[responseIndex, delayIndex] += 1
        index += len(responseStarts[n])
    image[np.where(avgToTake==0)] = 1
    #avgToTake[np.where(avgToTake != 0)] = np.max(avgToTake)
    avgToTake[np.where(avgToTake==0)] = 1
    image /= avgToTake
    image = np.flip(image, 0)

    return image

def unrollResponse(
        responses: List[np.ndarray],
        responseStarts: List[np.ndarray],
        channels: np.ndarray,
        window: int = 200,
        padZeroEnd: int = 50,
        nrOfBinsDelay: int = None,
        nrOfBinsInput: int = None
):
    """
    Stacks all responses into a 4D array with size (n_pulse,n_bins_input,n_bins_delay,n_electrodes).
    The electrodes are sorted like electrodeChannelMapping.
    :param responses: A List containing arrays of shape (4,n_spikes) with the first dimension corresponding to
    (channel, delay, amplitude, response). A list allows the processing of multiple experiments into one plot.
    :param responseStarts: A list containing the response starts for each array of responses.
    :param channels: A (n_electrodes) numpy array with all channels
    :param window: The response window at which to look at.
    :param nrOfBins: In how many bins the responses are sorted. Default is the number of samples in the window.
    :return: A numpy array resembling an image of shape (H,W,3).
    """
    if nrOfBinsDelay is None:
        nrOfBinsDelay = window
    if nrOfBinsInput is None:
        nrOfBinsInput = sum([len(r) for r in responseStarts])
    else:
        responseStarts = [(np.copy(r) - np.min(r)) for r in responseStarts]
        binSize = np.ceil(sum(np.max(r) for r in responseStarts) / (nrOfBinsInput))
        responses = deepcopy(responses)
        for r in responses:
            r[3] = responseStarts[r[3]] // binSize

    responseVector = np.zeros((nrOfBinsInput, nrOfBinsDelay+padZeroEnd, len(channels)))
    index = 0
    for n, delayVectors in enumerate(responses):
        validSpikes = np.copy(delayVectors[:,np.where(delayVectors[1]<window)[0]])
        if validSpikes.shape[1] != 0:
            responseIndex = (validSpikes[3] + index).astype(int)
            delayIndex = np.round(validSpikes[1]*nrOfBinsDelay/window).astype(int)
            indices = np.argwhere(channels == validSpikes[0][:, None])[:,1]
            responseVector[responseIndex,delayIndex,indices] = 1
        index += len(responseStarts[n])

    return responseVector


class PCA():
    """
    Class to do PCA with. Input is of shape (n_batch, ..., n_{nth_feature_dimension})
    Based on: Turk, Matthew, and Alex Pentland. "Eigenfaces for recognition." Journal of cognitive neuroscience 3.1 (1991): 71-86.
    """
    def __init__(self, initialSignals: np.ndarray = None, saveFile: str = None, loadFile: str = None):
        if loadFile is None:
            initialSignals = initialSignals.reshape((len(initialSignals),-1))
            self.mean = np.mean(initialSignals, axis=0, keepdims=True)
            covMatrix = np.matmul((initialSignals-self.mean).T, initialSignals-self.mean) / (len(initialSignals)-1)
            e, v = np.linalg.eigh(covMatrix)
            # Sorting in descending order to have eigenvectors of highest variance in the first indices
            self.eigenvectors = v[:,::-1]
            self.eigenvalues = e[::-1]
            if saveFile is not None:
                np.save(saveFile, self.eigenvectors)
        else:
            self.eigenvectors = np.load(loadFile)

    def __call__(self, signals: np.ndarray, nDim: int = 2, transformBack: bool = False):
        signalShape = signals.shape
        # Reshaping to 1D and transposing, s.t. notation is equal to non batch case: U^Tx = z
        signals = (signals.reshape((len(signals), -1))-self.mean).T
        z = np.matmul(self.eigenvectors.T,signals)
        if transformBack:
            z[nDim:,:] = 0
            signals = np.matmul(self.eigenvectors, z)
            # Transposing back s.t. shape is again (n_batch, n_{tot_features})
            signals = signals.T
            return (signals+self.mean).reshape(signalShape)
        else:
            return z[:nDim].T
    def transformZ(self,zVector: np.ndarray):
        z = np.concatenate([zVector,np.zeros((len(zVector),self.eigenvectors.shape[0]-zVector.shape[-1]))],axis=1).T
        signals = np.matmul(self.eigenvectors, z)
        # Transposing back s.t. shape is again (n_batch, n_{tot_features})
        signals = signals.T
        signalShape = np.concatenate((np.atleast_1d(len(zVector)),self.mean.shape[1:]))
        return (signals + self.mean).reshape(signalShape)
