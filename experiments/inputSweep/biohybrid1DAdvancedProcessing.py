import sys, os, io
from data.SignalProcessing.evaluate import PCA, responseDelayOverTime, averageFrequencyResponsePerElectrode, averageDelayResponse, meanISIResponse, meanPhaseResponse, unrollResponse, stack_arrays
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.decomposition import FastICA
import numpy as np
from data.utils.h5pyUtils import getMetaInfo
from data.utils.parse_utils import parse_range
import config
from data.SignalProcessing.mapping import Linear_Colour_Coding, Spiral_Colour_Coding, TwoInOneOutCoding
import glob, os
from data.plotting import mapColourToElectrode, fourTileColourMap
import multiprocessing as mp
from tqdm import tqdm
from scipy.ndimage import convolve1d
from sklearn.manifold import MDS
from sklearn.cluster import OPTICS, HDBSCAN
import tarfile
import h5py
from data.SignalProcessing.analysisAlgorithms import blahut_arimoto, estimate_p_y_given_x, calculate_MI, calculate_MI_MAP
import argparse
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data
import contextlib


def process(
    path:str,
    setting:str,
    chipAndDIV:str,
    network:int,
    postfix: str,
    counter: mp.Value = None,
    lock: mp.Lock = None
    ):

    inputArray = config.settings[setting]["inputArray"]
    nrOfResponses = config.settings[setting]["nrOfResponses"]
    folder=os.path.join(path,"processed")
    fileSave = os.path.join(path, f"processedAdv/{chipAndDIV}_{setting}_{network}{postfix}.h5")
    #log_file = f"{fileSave[:-3]}.txt"  # Unique log file per process
    #sys.stdout = open(log_file, 'w')
    #sys.stderr = open(log_file, 'w')
    h5FileName = os.path.join(folder, f"{chipAndDIV}_N{network}_{setting}{postfix}.h5")
    with h5py.File(h5FileName, 'r') as h5file:
        routing = h5file["Meta_Info/routing"][:]
        channelMapping = h5file["Meta_Info/channel_mapping"][:]
        lsb = h5file["Meta_Info/lsb"][()]
        gain = h5file["Meta_Info/gain"][()]
        voltage_map = h5file["Meta_Info/voltage_map"][:]

    routing = routing.flatten()
    electrodes = np.where(routing == 1)[0]
    mappedElectrodes = channelMapping[0,np.where(np.in1d(channelMapping[0],electrodes))[0]]
    channels = channelMapping[1,np.where(np.in1d(channelMapping[0],electrodes))[0]]
    sortedSpikes = []
    sortedSpikesAll = []
    spikesAll = []
    responseStarts = []
    responseStartsAll = []
    spikeShapes = []
    # Open the HDF5 file and read the data
    with h5py.File(h5FileName, 'r') as h5file:
        for i, amp in enumerate(inputArray):
            grp = h5file[f"{amp}"]
            indicesToKeep = np.isin(grp['spikes'][0],channels)
            tempSpikes = grp['spikes'][:,indicesToKeep]
            sortedSpikesAll.append(np.copy(tempSpikes))
            responseStartsAll.append(np.copy(grp['starts']))
            tempSpikesAll = grp['spikes_all'][:]
            indicesToKeep = np.isin(tempSpikesAll[:,2],channels)
            tempSpikesAll = tempSpikesAll[indicesToKeep]
            spikesAll.append(np.copy(tempSpikesAll))

            indicesToKeep = np.arange(len(grp['starts']))
            if len(indicesToKeep) < nrOfResponses:
                print(f"Not enough responses, reduced quantity to {len(indicesToKeep)}")
                nrOfResponses = len(indicesToKeep)
            stepSize = len(indicesToKeep) // nrOfResponses
            skipSize = len(indicesToKeep) % nrOfResponses
            indicesToKeep = indicesToKeep[skipSize::stepSize]
            indicesToKeep = np.isin(tempSpikes[3, :], indicesToKeep)

            responseStarts.append(grp['starts'][skipSize::stepSize])
            tempSpikes = tempSpikes[:, indicesToKeep]
            if 'spikeShapes' in h5file:
                spikeShapes.append(grp['spikeShapes'][indicesToKeep])
            tempSpikes[3] = (tempSpikes[3]-skipSize) // stepSize
            sortedSpikes.append(np.concatenate([tempSpikes,np.ones((1, tempSpikes.shape[1]))*i],axis=0))

    try:
        # Try to open the file
        with h5py.File(fileSave, 'r') as file:
            pass  # File is fine, no action needed
    except:
        # If an OSError occurs, it could be due to a corrupted file
        with h5py.File(fileSave, 'w') as file:
            pass  # File is fine, no action needed

    with (h5py.File(fileSave, 'a') as h5file):
        if 'meta_info' in h5file:
            del h5file['meta_info']
        grp = h5file.create_group('meta_info')
        grp.create_dataset('routing', data=routing)
        grp.create_dataset('channel_mapping', data=np.vstack([mappedElectrodes,channels]))
        grp.create_dataset('voltage_map', data=voltage_map)
        grp.create_dataset('gain', data=gain)
        grp.create_dataset('lsb', data=lsb)

    with (h5py.File(fileSave, 'a') as h5file):
        if 'spike_detection' in h5file:
            del h5file['spike_detection']
        spikes = np.concatenate(sortedSpikes,axis=1)
        grp = h5file.create_group('spike_detection')
        grp.create_dataset('spikes', data=spikes)
        responseStartsArray = stack_arrays(responseStarts, nrOfResponses)
        grp.create_dataset('starts', data=responseStartsArray)

        grp_sub = grp.create_group('spikes_all')
        for i, spikesTemp in enumerate(spikesAll):
            amp = inputArray[i]
            grp_sub.create_dataset(f"{amp}", data=spikesTemp)



    if len(spikeShapes)!=0:
        with (h5py.File(fileSave, 'a') as h5file):
            if 'spike_shapes' in h5file:
                del h5file['spike_shapes']
            grp = h5file.create_group('spike_shapes')
            n_dim = 3
            spikeShapes = np.concatenate(spikeShapes,axis=0)
            spikeShapesShape = spikeShapes.shape
            spikes = np.concatenate(sortedSpikes,axis=1)
            avg_spike_shapes_el_temp = []
            top_components_el_temp = []
            cluster_labels_temp = []
            for i in range(len(channels)):
                spikes_el = spikeShapes[spikes[0]==channels[i],i]
                if len(spikes_el)==0:
                    avg_spike_shapes_el_temp.append(np.zeros((1,spikeShapes.shape[-1]))+np.nan)
                    top_components_el_temp.append(np.zeros((1,n_dim))+np.nan)
                    cluster_labels_temp.append(np.zeros(1)+np.nan)
                    continue
                elif len(spikes_el)==1:
                    avg_spike_shapes_el_temp.append(spikes_el)
                    top_components_el_temp.append(np.zeros((1,n_dim)))
                    cluster_labels_temp.append(np.array([-1]))
                    continue
                if len(spikes_el) < np.sum(spikeShapes.shape[1:]):
                    pca = PCA(spikes_el)
                    print(f"{np.sum(100 * pca.eigenvalues[:n_dim] / np.sum(pca.eigenvalues)):.2f}%")
                    top_components = pca(spikes_el, nDim=n_dim)
                else:
                    pca = PCA_sklearn(n_components=n_dim)
                    pca.fit(spikes_el.reshape(len(spikes_el), -1))
                    top_components = pca.transform(spikes_el.reshape(len(spikes_el), -1))
                    print(f"{100*np.sum(pca.explained_variance_ratio_):.2f}%")
                clusterer = HDBSCAN(min_cluster_size=min(25,len(top_components)),min_samples=min(5,len(top_components)),allow_single_cluster=True)
                clustering = clusterer.fit(top_components)
                labelsOpt = clustering.labels_
                average_spikes = np.zeros((np.max(labelsOpt) + 1, spikeShapes.shape[-1]))
                n_random_points = 500
                sampled_cluster_points = np.zeros(((np.max(labelsOpt) + 1)*n_random_points, n_dim)) + np.nan
                for j in range(np.max(labelsOpt) + 1):
                    n_points = len(top_components[labelsOpt==j])
                    sampled_cluster_points[j*n_random_points:j*n_random_points+min(n_random_points,n_points)] = top_components[labelsOpt==j][np.random.choice(n_points, min(n_random_points, n_points), replace=False)]
                dists = np.sqrt(np.sum((top_components[labelsOpt==-1][:, np.newaxis, :] - sampled_cluster_points[np.newaxis, :, :]) ** 2, axis=-1))
                labelsOpt[labelsOpt==-1] = np.nanargmin(dists,axis=1) // n_random_points
                for j in range(np.max(labelsOpt) + 1):
                    average_spikes[j] = np.mean(spikes_el[labelsOpt == j], axis=0)
                avg_spike_shapes_el_temp.append(average_spikes)
                top_components_el_temp.append(top_components)
                cluster_labels_temp.append(labelsOpt)
            del spikeShapes
            maxLength = max([len(cl_spike) for cl_spike in avg_spike_shapes_el_temp])
            avg_spike_shapes_el = np.zeros((maxLength,len(channels),spikeShapesShape[-1]),dtype=np.float32) + np.nan
            for i,cl_spike in enumerate(avg_spike_shapes_el_temp):
                avg_spike_shapes_el[:len(cl_spike),i] = cl_spike.astype(np.float32)
            maxLength = max([len(pca_feat) for pca_feat in top_components_el_temp])
            top_components_el = np.zeros((maxLength, len(channels), n_dim)) + np.nan
            labels_el = np.zeros((maxLength, len(channels), 1)) + np.nan
            for i, pca_feat in enumerate(top_components_el_temp):
                top_components_el[:len(pca_feat),i] = pca_feat
                labels_el[:len(pca_feat),i,0] = cluster_labels_temp[i]
            print("Saving Clustering...")
            grp.create_dataset('average_spike_shapes_single_el', data=avg_spike_shapes_el, compression='gzip', compression_opts=1)
            grp.create_dataset('top_components_el', data=top_components_el)
            grp.create_dataset('labels_el', data=labels_el)

            del avg_spike_shapes_el, labels_el, top_components_el, responseStartsArray, spikes
    
    with (h5py.File(fileSave, 'a') as h5file):
        window_list = np.arange(20, 201, 20)
        encodings = dict()
        if 'Rate' in h5file:
            del h5file['Rate']
        grp = h5file.create_group('Rate')
        encodings["Rate"] = dict()
        for w in window_list:
            countsAll = averageFrequencyResponsePerElectrode(sortedSpikes, responseStarts, mappedChannels=channels,window=w)
            countsAll = stack_arrays(countsAll, nrOfResponses)
            grp.create_dataset(f"w{w}", data=countsAll)
            countsAll = np.mean(countsAll, axis=-1)
            encodings["Rate"][f"w{w}"] = countsAll

        if 'Latency' in h5file:
            del h5file['Latency']
        grp = h5file.create_group('Latency')
        encodings["Latency"] = dict()
        for w in window_list:
            latencyAll = averageDelayResponse(sortedSpikes, responseStarts, mappedChannels=channels,window=w)
            latencyAll = stack_arrays(latencyAll, nrOfResponses)
            grp.create_dataset(f"w{w}", data=latencyAll)
            latencyAll = np.mean(latencyAll, axis=-1)
            encodings["Latency"][f"w{w}"] = latencyAll

        if 'ISI' in h5file:
            del h5file['ISI']
        grp = h5file.create_group('ISI')
        encodings["ISI"] = dict()
        for w in window_list:
            isiAll = meanISIResponse(sortedSpikes, responseStarts, mappedChannels=channels,window=w)
            isiAll = stack_arrays(isiAll, nrOfResponses)
            grp.create_dataset(f"w{w}", data=isiAll)
            isiAll = np.mean(isiAll, axis=-1)
            encodings["ISI"][f"w{w}"] = isiAll

        if 'Phase' in h5file:
            del h5file['Phase']
        grp = h5file.create_group('Phase')
        encodings["Phase"] = dict()
        for w in window_list:
            phaseAll = meanPhaseResponse(sortedSpikes, responseStarts, mappedChannels=channels,window=w, periodicity=w)
            phaseAll = stack_arrays(phaseAll, nrOfResponses)
            grp.create_dataset(f"w{w}", data=phaseAll)
            phaseAll = np.mean(phaseAll, axis=-1)
            encodings["Phase"][f"w{w}"] = phaseAll

    with (h5py.File(fileSave, 'a') as h5file):
        print("Started AIS Analysis")
        if 'AIS' in h5file:
            grp = h5file['AIS']  # Access the existing group
        else:
            grp = h5file.create_group("AIS")
        ais = ActiveInformationStorage()
        settings = {"cmi_estimator": "JidtKraskovCMI", "fdr_correction": False, "max_lag": 1, "verbose": True, "n_perm_min_stat": 200, "n_perm_max_stat": 200}
        pbar = tqdm(total=len(encodings) * len(window_list))
        for key, encoded_data_type in encodings.items():
            print(f"Started with {key}")
            if f'AIS/{key}' in h5file:
                del h5file[f'AIS/{key}']
            sub_group = grp.create_group(key)
            for key_window, encoded_data in encodings[key].items():
                p_ais = np.zeros(len(encoded_data)).astype(float)
                for i, t in enumerate(encoded_data):
                    with open('/dev/null', 'w') as f, contextlib.redirect_stdout(f):
                        # Prepare data for IDTxl
                        idtxl_data = Data(t[None, :, None], "psr", normalise=False)
                        result = ais.analyse_network(settings=settings, data=idtxl_data)

                        # Store results
                        p_ais[i] = result.get_single_process(0, fdr=False)["ais_pval"]
                pbar.update(1)
                sub_group.create_dataset(key_window + "_p_ais", data=p_ais)


    capacity_group_name="Capacity_Estimation"
    with (h5py.File(fileSave, 'a') as h5file):
        print("Started Capacity Analysis")
        if capacity_group_name in h5file:
            grp = h5file[capacity_group_name]  # Access the existing group
        else:
            grp = h5file.create_group(capacity_group_name)
            
        tries = 8
        pbar = tqdm(total=tries*len(encodings)*len(window_list))
        for key, encoded_data_type in encodings.items():
            print(f"Started with {key}")
            if f'{capacity_group_name}/{key}' in h5file:
                del h5file[f'{capacity_group_name}/{key}']
            sub_group = grp.create_group(key)
            for key_window, encoded_data in encodings[key].items():
                n_bins = 2 ** (np.arange(1, tries + 1))
                capacities = np.zeros_like(n_bins).astype(float)
                probabilities = np.zeros((len(n_bins), len(encoded_data)))
                P_Y_given_X = estimate_p_y_given_x(encoded_data)
    
                for b in range(len(n_bins)):
                    data_sorted = np.sort(encoded_data.flatten())
                    bin_edges = np.interp(np.linspace(0, len(data_sorted), n_bins[b]), np.arange(len(data_sorted)),
                                          data_sorted)
                    binned_array = np.clip(np.digitize(encoded_data, bin_edges), 0, n_bins[b]-1)
                    transitionMatrix = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_bins[b]), axis=1,
                                                           arr=binned_array) / nrOfResponses
                    c, p_x = blahut_arimoto(transitionMatrix)
                    probabilities[b] = p_x
                    capacities[b] = calculate_MI(p_x, P_Y_given_X)
                    pbar.update(1)
    
                sub_group.create_dataset(key_window+"_c", data=capacities)
                sub_group.create_dataset(key_window+"_p", data=probabilities)
                sub_group.create_dataset(key_window+"p_y", data=P_Y_given_X)

    capacity_group_name = "Capacity_Estimation_Small"
    with (h5py.File(fileSave, 'a') as h5file):
        print("Started Capacity Analysis")
        if capacity_group_name in h5file:
            grp = h5file[capacity_group_name]  # Access the existing group
        else:
            grp = h5file.create_group(capacity_group_name)
    
        tries = 8
        pbar = tqdm(total=tries * len(encodings) * sum([2,3,4]))
        for key, encoded_data_type in encodings.items():
            print(f"Started with {key}")
            if f'{capacity_group_name}/{key}' in h5file:
                del h5file[f'{capacity_group_name}/{key}']
            sub_group = grp.create_group(key)
    
            encoded_data = encodings[key][f"w{window_list[-1]}"]
    
            for n_split in [2,3,4]:
                split_indices = np.array_split(np.arange(encoded_data.shape[-1]), n_split)
                for n_set, indices in enumerate(split_indices):
                    subsample_data = encoded_data[..., indices]
                    n_bins = 2 ** (np.arange(1, tries + 1))
                    capacities = np.zeros_like(n_bins).astype(float)
                    #capacities_kernel = np.zeros_like(n_bins).astype(float)
                    probabilities = np.zeros((len(n_bins), len(subsample_data)))
                    P_Y_given_X = estimate_p_y_given_x(subsample_data)
    
                    for b in range(len(n_bins)):
                        data_sorted = np.sort(subsample_data.flatten())
                        bin_edges = np.interp(np.linspace(0, len(data_sorted), n_bins[b]),np.arange(len(data_sorted)),data_sorted)
                        binned_array = np.clip(np.digitize(subsample_data, bin_edges), 0, n_bins[b]-1)
                        transitionMatrix = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_bins[b]), axis=1,
                                                               arr=binned_array) / len(indices)
                        c, p_x = blahut_arimoto(transitionMatrix)
                        probabilities[b] = p_x
                        capacities[b] = calculate_MI(p_x,P_Y_given_X)
                        pbar.update(1)
    
                    sub_group.create_dataset(f"{n_split}_{n_set}_c", data=capacities)
                    sub_group.create_dataset(f"{n_split}_{n_set}_p", data=probabilities)
                    sub_group.create_dataset(f"{n_split}_{n_set}_p_y", data=P_Y_given_X)
    if lock is not None and counter is not None:
        with lock:
            counter.value += 1

""" def start_processes(expSets, n_cores):
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
                args = (expSet[4][n // nrOfNetworks], chipAndDIV, n % nrOfNetworks, path, expSet[7], counter, lock)
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
This file takes the .h5 files produced by biohybridProcess2D.py and calculates different output encodings and 
determines the channels capacity. Further, spikes are sorted per electrode. 

In the .h5, each value of the sweep has its own dataset consisting of the following:
- spikes: Contains infos and PCA of all spikes
- Rate, Latency, etc.: Consists of an array of shape (n_sweep_values,n_responses,n_channels)
- Capacity: Has the capacity calculation for each encoding for different bin and window sizes. 

On bin sizes: The channel has to be discretized to use the Blahut-Arimoto algorithm. The nr of bins and the value span
defines the size. For the final output, it is recommended to use the smallest bin size. 
On window sizes: Window size is usually specified after the “w“. This value specifies in number of samples, how much 
of the response following a stimulus is considered. 
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("--exp_idx",
                        default="255", #1,2,15-20
                        type=parse_range,
                        help="Indices of files to be loaded, e.g., 1,2,4-6")
    args = parser.parse_args()
    expSets = [config.experiments[i] for i in args.exp_idx]

    for expSet in expSets:
        networks = expSet[0]
        chipAndDIV = expSet[1]
        path = expSet[2]
        for n in range(len(networks) * len(expSet[4])):
            process(expSet[8], expSet[4][n // len(networks)], chipAndDIV, networks[n % len(networks)], expSet[7])
