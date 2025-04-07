import numpy as np
from typing import List

def getResponseRaw(traces: np.ndarray, responseStarts: np.ndarray, window: (int,int) = (0, 200))->np.ndarray:
    """
    Cuts out patches of a (n_channels,n_samples) trace numpy array.
    :param traces: Traces from which the cut is to be done.
    :param responseStarts: The center of the cut signal patch.
    :param window: Size of the patch surrounding the center.
    :return: A numpy array of shape (n_patches,n_channels,n_samples)
    """

    if window[1] <= window[0]:
        raise Exception("Inappropriate window.")

    channel_indices = np.arange(traces.shape[0])
    time_indices = np.arange(window[0],window[1],1) + np.atleast_2d(responseStarts).T
    # create a meshgrid of indices for each patch
    row_indices = np.expand_dims(channel_indices, axis=-1).repeat(int(window[1]-window[0]), axis=-1)
    col_indices = np.expand_dims(time_indices, axis=-2).repeat(traces.shape[0], axis=-2)
    # extract the patches using advanced indexing
    patches = np.pad(traces, pad_width=((0, 0), (-window[0],window[1])), mode='empty')[row_indices, col_indices-window[0]]
    return patches


def getResponse(spikeTimes: np.ndarray, amplitudes: np.ndarray,channels: np.ndarray, electrodeChannelMapping: np.ndarray,
                responseStarts: np.ndarray, window: (int,int) = (0, 200), consoleOut = False) -> (np.ndarray,np.ndarray):
    """
    Based on an array containing the start timings of a response, the spikes are labeled with the corresponding response.
    :param spikeTimes: A 1d numpy array containing the spike timings.
    :param amplitudes: A 1d numpy array containing the spike amplitudes.
    :param channels: A 1d numpy array containing the channels on which the spike has been recorded.
    :param usedChannels: Which channels where used for the recording. This is necessary, since there is no guarantee
    that all channels recorded spikes and therefore occur in the channels array.
    :param responseStarts: Start timings of the responses
    :param window: Window, in which a spike has to occur such that it belongs to a specific response start.
    :return: A numpy array of shape (5,n_filtered_spikes), where the first dimension corresponds to
    (channel, delay, amplitude, response, electrode) and the used response starts.
    """
    if len(responseStarts)<=0:
        print("Warning: Array with start timings is empty")
    elif np.min(responseStarts) < -window[0] or window[1] <= window[0]:
        raise Exception("Inappropriate window.")
    responses = np.zeros([5,spikeTimes.shape[0]])
    startIndex = 0
    lastStart = -np.inf
    nKickedResp = 0
    newResponseStarts = []
    for i, start in enumerate(responseStarts.astype(dtype=np.int64)):
        if start-lastStart > window[1]+window[0]:
            indices = np.where(np.logical_and(start+window[0]<=spikeTimes,spikeTimes<start+window[1]))[0]
            if len(indices) != 0:
                responses[0, startIndex:startIndex + len(indices)] = channels[indices]
                responses[1,startIndex:startIndex+len(indices)] = spikeTimes[indices] - start
                responses[2, startIndex:startIndex + len(indices)] = amplitudes[indices]
                responses[3, startIndex:startIndex + len(indices)] = len(newResponseStarts)
            startIndex += len(indices)
            lastStart = start
            newResponseStarts.append(start)
        else:
            nKickedResp += 1
    if nKickedResp != 0 and consoleOut:
        print(f"{nKickedResp} responses kicked due to being to close to previous.")

    responses = responses[:,:startIndex]
    responses[4] = electrodeChannelMapping[0, np.searchsorted(electrodeChannelMapping[1], responses[0])]
    return responses, np.asarray(newResponseStarts)



def spikeListToSpikeArray(spikeIndices: List[np.ndarray], amplitudes: List[np.ndarray], channels: List[int], ) -> np.ndarray:
    """
    Converts a list of size n_channels containing all detected spikes to a (n_spikes, 3) numpy array.
    :param spikeIndices: List containing the 1d numpy arrays of the detected spikes for each channel.
    :param amplitudes: List containing the 1d numpy arrays of the amplitudes of the detected spikes for each channel.
    :param channels: A list of channels which specifies the channels of the spikeIndices and amplitudes lists.
    :return: A (n_spikes,3) numpy array with the last dimension corresponding to (spike_timimg, amplitude, channel)
    """

    if len(spikeIndices) != len(amplitudes):
        raise Exception("Spike Indices and amplitudes do not have the same length.")
    elif len(channels) != len(amplitudes):
        raise Exception("Channels and amplitudes do not have the same length.")

    nrOfSpikes = sum([len(i) for i in spikeIndices])
    spikeMatrix = np.zeros((nrOfSpikes,3))

    index = 0
    for i, spikeTimes in enumerate(spikeIndices):
        spikeMatrix[index:index +len(spikeTimes),0] = spikeTimes
        spikeMatrix[index:index + len(spikeTimes), 1] = amplitudes[i]
        spikeMatrix[index:index + len(spikeTimes), 2] = channels[i]
        index += len(spikeTimes)

    return spikeMatrix[np.argsort(spikeMatrix[:,0])]
