import sys, os
import numpy as np
import h5py
from data.utils.h5pyUtils import getMetaInfo, loadh5pyRaw, loadh5pySpikes
from data.utils.parse_utils import parse_range
from data.SignalProcessing.postprocessing import Filter, PeakDetection
from data.SignalProcessing.dataPreparation import getResponse, spikeListToSpikeArray, getResponseRaw
import glob, os
import multiprocessing as mp
from tqdm import tqdm
import tarfile
from time import time
from datetime import datetime
import config
import argparse

def process(
    save_path: str,
    setting: dict,
    chipAndDIV:str,
    network:int,
    path:str,
    spikeDetection,
    voltageMapPath,
    blanking = None,
    postfix:str = "",
    save_spike_shapes:bool = False,
    counter: mp.Value = None,
    lock: mp.Lock = None
    ):
    selectionPath = os.path.join(path,f"selection_N{network}.npy")
    selectionPathAll = os.path.join(path,f"selection.npy")
    rate = 4

    inputArray = config.settings[setting]["inputArray"]
    expectedPeriodicity = config.settings[setting]["expectedPeriodicity"]
    pickMinimum = config.settings[setting]["pickMinimum"]
    responseWindow = config.settings[setting]["responseWindow"]
    blankingWindow = 30 # Used if no blanking function is given
    path = os.path.join(path,config.settings[setting]["path"])

    path = os.path.join(path, f"*_{inputArray[0, 0]}_{inputArray[0, 1]}.raw.h5")
    if len(glob.glob(path)) == 0:
        raise FileNotFoundError(path)
    else:
        path = glob.glob(path)[0]

    routingOg = np.load(selectionPath)
    routingAllOg = np.load(selectionPathAll)
    routing = routingOg.flatten()
    routingAll = routingAllOg.flatten()
    electrodes = np.where(routing==1)[0]
    triggersOne = np.where(np.logical_and(routingAll>1, routingAll%2==0))[0]
    triggersTwo = np.where(np.logical_and(routingAll > 1, routingAll % 2 == 1))[0]
    metaInfo = getMetaInfo(path)
    channelMapping = metaInfo[0]

    channels = channelMapping[1,np.where(np.in1d(channelMapping[0],electrodes))[0]]
    # prevLength = len(channels)
    # _, channels = filterSaturatedElectrodes(loadh5pyRaw([path], channels).astype(float) * metaInfo[2],channels)
    # del _
    # print(f"Removed {prevLength-len(channels)} channels")
    triggerChannelsOne = channelMapping[1,np.where(np.in1d(channelMapping[0],triggersOne))[0]]
    triggerChannelsTwo = channelMapping[1,np.where(np.in1d(channelMapping[0],triggersTwo))[0]]
    sortedSpikes = []

    folder = os.path.join(save_path, "processed")
    h5FileName = os.path.join(folder,f"{chipAndDIV}_N{network}_{setting}{postfix}.h5")
    #log_file = f"{h5FileName[:-3]}.txt"  # Unique log file per process
    #sys.stdout = open(log_file, 'w')
    #sys.stderr = open(log_file, 'w')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with h5py.File(h5FileName, 'w') as h5file:
        grp = h5file.create_group("Meta_Info")
        grp.create_dataset('channel_mapping', data=channelMapping)
        grp.create_dataset('gain', data=metaInfo[1])
        grp.create_dataset('lsb', data=metaInfo[2])
        grp.create_dataset('routing',data=routingOg)
        grp.create_dataset('routing_all', data=routingAllOg)
        grp.create_dataset('voltage_map',data=np.load(voltageMapPath))

    loadFileNames = [glob.glob(os.path.join(os.path.dirname(path), f"*_{amp[0]}_{amp[1]}.raw.h5"))[0] for amp in inputArray]
    pbar = tqdm(inputArray)
    for j,amp in enumerate(pbar):
        # Get spikedata of wanted electrodes
        file = loadFileNames[j]
        tic = time()
        skipBeginning = 0
        # Next, we have to get rid of low frequency fluctuations by filtering, each channel is filtered separately.
        filter = Filter()
        # Here we extract the raw signal of a file, loadh5pyRaw reads lists, however this is not often needed.
        rawData = loadh5pyRaw([file], channels).astype(np.float32)[:,skipBeginning:]*metaInfo[2]
        rawDataTriggerOne = loadh5pyRaw([file], triggerChannelsOne)[:,skipBeginning:].astype(np.float32)*metaInfo[2]
        rawDataTriggerTwo = loadh5pyRaw([file], triggerChannelsTwo)[:,skipBeginning:].astype(np.float32)*metaInfo[2]

        singleInputFlag = False
        responseStarts = None
        if amp[0] == 0 and amp[1] == 0:
            responseStarts = np.sort(np.arange(rawData.shape[1]-int(20000/rate)-25,0,-int(20000/rate)))
        elif amp[0] == 0 or amp[1] == 0:
            singleInputFlag = True

        timeLoading = time() - tic
        tic = time()
        for i, trace in enumerate(rawDataTriggerOne):
            rawDataTriggerOne[i] = filter(trace)
        for i, trace in enumerate(rawDataTriggerTwo):
            rawDataTriggerTwo[i] = filter(trace)
        # Get Artefacts
        timeFiltering = time() - tic
        tic = time()
        detectedartefactsOne = []
        detectSpikes = PeakDetection(spikeDistance=expectedPeriodicity[j,0], spikeThreshold=0.001,useStd=False)
        for blanktrace in rawDataTriggerOne:
            detectedartefactsOne.append(
                detectSpikes(blanktrace)[0])
        detectedartefactsOne = np.unique(np.concatenate(detectedartefactsOne))
        detectedartefactsTwo = []
        detectSpikes = PeakDetection(spikeDistance=expectedPeriodicity[j,1], spikeThreshold=0.001, useStd=False)
        for blanktrace in rawDataTriggerTwo:
            detectedartefactsTwo.append(
                detectSpikes(blanktrace)[0])
        detectedartefactsTwo = np.unique(np.concatenate(detectedartefactsTwo))
        for i,artefacts in enumerate([detectedartefactsOne,detectedartefactsTwo]):
            responseStability = [np.mean(np.diff(artefacts)), np.std(np.diff(artefacts)),
                                 np.median(np.diff(artefacts))]
            print(
                f"Distance between detected artifacts of {amp[i]}, #{len(artefacts)} per {rawData.shape[1]/20000}s: {responseStability[0]} +- {responseStability[1]}, median {responseStability[2]}")
        del rawDataTriggerOne, rawDataTriggerTwo
        minArtefactCount = 5

        if responseStarts is None:
            if singleInputFlag:
                if len(detectedartefactsOne)>=minArtefactCount and len(detectedartefactsTwo)<minArtefactCount:
                    responseStarts = np.unique(detectedartefactsOne[1:])
                elif len(detectedartefactsOne)<minArtefactCount and len(detectedartefactsTwo)>=minArtefactCount:
                    responseStarts = np.unique(detectedartefactsTwo[1:])
                else:
                    responseStarts = np.unique(np.concatenate([detectedartefactsOne[1:], detectedartefactsTwo[1:]]))
                    print("We are having an issue.")
            elif pickMinimum == 2:
                #pick only lower frequency
                responseStarts = np.unique([detectedartefactsOne[1:],detectedartefactsTwo[1:]][np.argmin(amp)])
            elif pickMinimum == 1:
                # pick only where close consecutive pulses appear, then pick later one
                responseStarts = np.unique(np.concatenate([detectedartefactsOne[1:], detectedartefactsTwo[1:]]))
                mask = np.diff(responseStarts) < 30 # Fixed delay for experiments
                # shift the mask to the right to mark the second element when the difference is less than n
                mask = np.append(False, mask)
                # create a new array excluding elements where mask is True
                responseStarts = responseStarts[mask]
            else:
                # pick everything
                responseStarts = np.unique(np.concatenate([detectedartefactsOne[1:],detectedartefactsTwo[1:]]))

            # If too close together, I only want last stimulus
            # find where the difference is less than n
            mask = np.diff(responseStarts) < responseWindow
            # shift the mask to the right to mark the second element when the difference is less than n
            mask = np.append(mask, False)
            # create a new array excluding elements where mask is True
            responseStarts = responseStarts[~mask]
        artefacts = np.sort(np.concatenate([detectedartefactsOne,detectedartefactsTwo]))
        # Remove offset
        if blanking is not None:
            rawData= blanking(traces=rawData, blankTimings=[detectedartefactsOne, detectedartefactsTwo])
        timeBlanking = time() - tic
        tic = time()
        for i, trace in enumerate(rawData):
            rawData[i] = filter(trace)
        timeFiltering += time() - tic
        tic = time()
        spikes = []
        amplitudes = []
        for i, trace in enumerate(rawData):
            spikes.append(spikeDetection(trace)[0][1:])
            amplitudes.append(rawData[i,spikes[i]])
        spikes = spikeListToSpikeArray(spikes,amplitudes,channels)
        timeSpikeDetection = time() - tic
        tic = time()
        sortedSpikesTemp, usedStarts = getResponse(spikes[:,0], spikes[:,1], spikes[:,2], channelMapping, responseStarts, window=(0,responseWindow))
        if blanking is None:
            sortedSpikesTemp = sortedSpikesTemp[:,sortedSpikesTemp[1]>blankingWindow]
        sortedSpikes.append(sortedSpikesTemp)
        timeResponseSorting = time() - tic
        timings = usedStarts[sortedSpikesTemp[3].astype(int)] + sortedSpikesTemp[1]
        if save_spike_shapes:
            spikeShapes = getResponseRaw(rawData, timings.astype(int), window=(-20, 20))
        with h5py.File(h5FileName, 'a') as h5file:
            grp = h5file.create_group(f"{amp[0]}_{amp[1]}")
            grp.create_dataset('spikes_all', data=spikes)
            grp.create_dataset('spikes', data=sortedSpikesTemp)
            grp.create_dataset('artifacts', data=artefacts)
            grp.create_dataset('starts', data=usedStarts)
            if save_spike_shapes:
                grp.create_dataset('spikeShapes', data=spikeShapes, compression='gzip', compression_opts=1)
        pbar.set_description(
            f"Load: {timeLoading:.2f}s, "
            f"Filter: {timeFiltering:.2f}s, "
            f"Blank: {timeBlanking:.2f}s, "
            f"Spikes: {timeSpikeDetection:.2f}, "
            f"Sorting: {timeResponseSorting:.2f}"
        )
    # Update the counter
    if lock is not None and counter is not None:
        with lock:
            counter.value += 1

""" # Function to start and manage processes
def start_processes(expSets, n_cores):
    with mp.Manager() as manager:
        # Create shared counter and lock via manager
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        tasks = []
        for expSet in expSets:
            nrOfNetworks = expSet[0]
            chipAndDIV = expSet[1]
            path = expSet[2]
            for n in range(nrOfNetworks * len(expSet[4])):
                args = (expSet[4][n // nrOfNetworks], chipAndDIV, n % nrOfNetworks, path, expSet[5], expSet[6], expSet[7], counter, lock)
                tasks.append(args)

        # Setup tqdm progress bar
        with tqdm(total=len(tasks)) as pbar:
            with mp.Pool(n_cores) as pool:
                # imap_unordered runs tasks asynchronously and unordered
                for _ in pool.imap_unordered(process_wrapper, tasks):
                    pbar.update(1)

# Wrapper function to allow passing args in imap_unordered
def process_wrapper(args):
    process(*args) """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description=
'''
This file takes the raw recordings produced by the Maxwell system and extracts spikes and heuristically removes artifacts.
It will produce a .h5 file for each sweep in a folder named "processed". This folder has to be created manually.

In the .h5, each value of the sweep has its own dataset consisting of the following:
- spikes: Contains timing, channel, delay, response index, and further info for each spike
- artifacts: Contains all timings of detected artifacts
- starts: Contains the start timing for each response index
- spikeShapes: For each spike, its shape is stored here

The overall meta info is stored as well. 
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("--exp_idx",
                        default="2", #"1,2,15-20"
                        type=parse_range,
                        help="Indices of experiment configs to be loaded, e.g., 1,2,4-6")
    args = parser.parse_args()
    expSets = [config.experiments[i] for i in args.exp_idx]

    for expSet in expSets:
        networks = expSet[0]
        chipAndDIV = expSet[1]
        path = expSet[2]
        voltageMapPath = expSet[3]
        for n in range(len(networks) * len(expSet[4])):
            process(expSet[8], expSet[4][n // len(networks)], chipAndDIV, networks[n % len(networks)], path, expSet[5], voltageMapPath, expSet[6], expSet[7])
