import h5py
from typing import List
import numpy as np
import shutil, os
from tqdm import tqdm

def loadh5pySpikes(filepaths: List[str], channelsToReturn: np.ndarray = None)->np.ndarray:
    """
    Loads the spike matrix of a .h5 file.
    :param filepaths: List of the filepaths, that are to be loaded.
    :param channelsToReturn: The channels, that have to be extracted from the raw data.
    :return: Traces in the shape of (3, n_samples), with the first dimension being (spike_time, amplitude, channel)
    """

    rawDataFiles = []
    for file in filepaths:
        rawDataFiles.append(h5py.File(file, "r"))

    # Sort Files according to first spiketiming
    rawDataFilesBeginning = np.zeros(len(rawDataFiles),dtype=int)
    rawDataFileLength = 0
    usedIndices = []
    for i, rawData in enumerate(rawDataFiles):
        if channelsToReturn is None:
            indices = np.arange(0,len((rawData.get("proc0")["spikeTimes"])["frameno"]))
        else:
            channels = (rawData.get("proc0")["spikeTimes"])["channel"]
            indices = np.where(np.in1d(channels,channelsToReturn))[0]
        rawDataFilesBeginning[i] = np.min((rawData.get("proc0")["spikeTimes"])["frameno"])
        rawDataFileLength += len(indices)
        usedIndices.append(indices)
    rawDataFiles = [rawDataFiles[i] for i in np.argsort(rawDataFilesBeginning)]
    usedIndices = [usedIndices[i] for i in np.argsort(rawDataFilesBeginning)]
    # output with spiketimings/amplitudes/channels
    outputs = np.zeros([3,rawDataFileLength])

    startIndex = 0
    for i,rawData in enumerate(rawDataFiles):
        data = (rawData.get("proc0")["spikeTimes"])["frameno"][usedIndices[i]]
        outputs[0, startIndex:] = data
        outputs[1, startIndex:] = (rawData.get("proc0")["spikeTimes"])["amplitude"][usedIndices[i]]
        outputs[2, startIndex:] = (rawData.get("proc0")["spikeTimes"])["channel"][usedIndices[i]]
        startIndex += len(data)
    return outputs

def loadh5pyRaw(filepaths: List[str], channelsToReturn: np.ndarray = None)->np.ndarray:
    """
    Loads the raw data of multiple .h5 files.
    :param filepaths: List of the filepaths, that are to be loaded.
    :param channelsToReturn: The channels, that have to be extracted from the raw data.
    :return: Traces in the shape of (n_channels, n_samples)
    """

    rawDataFiles = []
    for file in filepaths:
        rawDataFiles.append(h5py.File(file, "r"))

    # Sort Files according to first spiketiming
    allRawData = []
    for i, rawData in enumerate(rawDataFiles):
        if channelsToReturn is None:
            allRawData.append(rawData["sig"])
        else:
            allRawData.append(rawData["sig"][np.atleast_1d(channelsToReturn)])
    return np.concatenate(allRawData)

def getMetaInfo(pathToFile: str) -> (np.ndarray,float,float):
    """
    Loads the meta info of a .h5 file.
    :param pathToFile: Filepath.
    :return: The electrode channel mapping as a (2,n_electrodes) matrix. First dimension is (electrodes,channels),
            gain, least significant bit
    """
    rawData = h5py.File(pathToFile, "r")
    gain = np.asarray(rawData["settings"]["gain"])[0]
    lsb = np.asarray(rawData["settings"]["lsb"])[0]
    electrodeInfo = np.asarray(rawData["mapping"]["channel", "electrode"])
    mask = [i["electrode"] != -1 for i in electrodeInfo]
    clean_abs_inds = np.asarray(
        [i[0]["electrode"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
    )
    clean_rel_inds = np.asarray(
        [i[0]["channel"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
    )

    # Kick out channels that are used by several electrodes
    _, unique_indices = np.unique(clean_rel_inds, return_index=True)
    if len(unique_indices) != len(clean_rel_inds):
        print(f"{len(clean_rel_inds)-len(unique_indices)} electrodes were removed since they were using the same channel.")
        clean_abs_inds = clean_abs_inds[unique_indices]
        clean_rel_inds = clean_rel_inds[unique_indices]

    electrodeChannelMapping = np.zeros(
        [2, clean_rel_inds.shape[0]], dtype=np.int32
    )

    # First index are the electrode numbers, second the channel where they are stored.
    electrodeChannelMapping[0, :] = np.squeeze(clean_abs_inds)
    electrodeChannelMapping[1, :] = np.squeeze(clean_rel_inds)

    return electrodeChannelMapping, gain, lsb

def compressFilesInDirectory(directory_path):
    def copy_file(input_file_path, output_file_path):
        shutil.copy2(input_file_path, output_file_path)
    def compress_dataset_in_h5_file(file_path):
        dataset_name = "sig"
        with h5py.File(file_path, 'a') as f:  # 'a' for read/write/create
            data = f[dataset_name][:]
            del f[dataset_name]
            f.create_dataset(dataset_name, data=data, compression='gzip', compression_opts=1)
    # Create a new directory for the compressed files
    parent_directory = os.path.dirname(directory_path)
    directory_name = os.path.basename(os.path.normpath(directory_path))
    compressed_directory_path = os.path.join(parent_directory, str(directory_name+'_compressed'))
    os.makedirs(compressed_directory_path, exist_ok=True)

    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.h5'):
            input_file_path = os.path.join(directory_path, filename)
            output_file_path = os.path.join(compressed_directory_path, filename)
            copy_file(input_file_path, output_file_path)
            compress_dataset_in_h5_file(output_file_path)
