import sys, os, io
from data.SignalProcessing.evaluate import responseDelayOverTime, averageFrequencyResponsePerElectrode, averageDelayResponse, stack_arrays
from data.SignalProcessing.analysisAlgorithms import blahut_arimoto
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.backends.backend_pdf
from matplotlib.patches import Patch
import numpy as np
from data.utils.h5pyUtils import getMetaInfo
from data.SignalProcessing.mapping import Linear_Colour_Coding, Spiral_Colour_Coding, TwoInOneOutCoding
import glob, os
from data.plotting import mapColourToElectrode, fourTileColourMap, violin_plot_with_map, plot_average_spikes_by_coordinates, plot_scatter_by_coordinates
import multiprocessing as mp
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize
from scipy.ndimage import convolve1d
from matplotlib.cm import ScalarMappable
import h5py
import seaborn as sns
import pandas as pd
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.neighbors import KernelDensity
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
import tarfile
from tqdm import tqdm
import config
import argparse
from data.utils.parse_utils import parse_range

def plot(
    path:str,
    setting:str,
    chipAndDIV:str,
    network:int,
    postfix: str,
    advancedPlots: bool = False,
    counter: mp.Value = None,
    lock: mp.Lock = None
    ):
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(os.path.join(path, "plots"),f"results_{chipAndDIV}_{setting}_{network}{postfix}.pdf"))
    filepath = os.path.join(path, f"processedAdv/{chipAndDIV}_{setting}_{network}{postfix}.h5")
    with h5py.File(filepath, 'r') as h5file:
        routing = h5file["meta_info/routing"][:]
        channelMapping = h5file["meta_info/channel_mapping"][:]
        voltageMap = h5file["meta_info/voltage_map"][:]
    plt.rcParams.update({'font.size': 16})

    max_rate = 2.5#1.9#2.4
    min_delay = 3#3#4
    min_phase = 2.5#3#2.5
    min_isi = 5

    unit = config.settings[setting]["unit"]
    legendLeft = config.settings[setting]["legendLeft"]
    legendRight = config.settings[setting]["legendRight"]
    inputArray = config.settings[setting]["inputArray"]
    nrOfResponses = config.settings[setting]["nrOfResponses"]
    responseWindow =config.settings[setting]["responseWindow"]
    imageColour = config.settings[setting]["imageColour"]
    plotArray = config.settings[setting]["plotArray"]

    routing = routing.flatten()
    electrodes = np.where(routing == 1)[0]
    triggerOne = np.where(routing == 2)[0]
    triggerTwo = np.where(routing == 3)[0]
    with h5py.File(filepath, 'r') as h5file:
        spikes = h5file['spike_detection/spikes'][:]
        if 'spike_shapes' in h5file:
            average_spikes_el = h5file['spike_shapes/average_spike_shapes_single_el'][:]
            top_components_el = h5file['spike_shapes/top_components_el'][:]
            labels_el = h5file['spike_shapes/labels_el'][:]
        else:
            average_spikes_el = None
            top_components_el = None
            labels_el = None
        responseStarts = h5file['spike_detection/starts'][:]
        countAllOG = h5file[f"Rate/w{responseWindow}"][:]
        delaysAllOG = h5file[f"Latency/w{responseWindow}"][:]
        isiAllOG = h5file[f"ISI/w{responseWindow}"][:]
        phaseAllOG = h5file[f"Phase/w{responseWindow}"][:]

        window_list = np.arange(20, 201, 20)
        encodings = ["Rate", "Latency", "ISI", "Phase"]
        if 'Capacity' in h5file:
            capacities = np.zeros((4,len(window_list)))
            for i, w in enumerate(window_list):
                capacities[0,i] = h5file[f"Capacity_Estimation/Rate/w{w}_c"][:][-1]
                capacities[1,i] = h5file[f"Capacity_Estimation/Latency/w{w}_c"][:][-1]
                capacities[2,i] = h5file[f"Capacity_Estimation/ISI/w{w}_c"][:][-1]
                capacities[3, i] = h5file[f"Capacity_Estimation/Phase/w{w}_c"][:][-1]
        else:
            capacities = None
        sortedSpikes = []
        for i in range(len(inputArray)):
            sortedSpikes.append(spikes[:,spikes[5]==i])


    mappedElectrodes = channelMapping[0,np.where(np.in1d(channelMapping[0],electrodes))[0]]
    channels = channelMapping[1,np.where(np.in1d(channelMapping[0],electrodes))[0]]
    mappedTriggersOne = triggerOne#channelMapping[0,np.where(np.in1d(channelMapping[0],triggerOne))[0]]
    triggerChannelsOne = channelMapping[1,np.where(np.in1d(channelMapping[0],triggerOne))[0]]
    mappedTriggersTwo = triggerTwo#channelMapping[0,np.where(np.in1d(channelMapping[0],triggerTwo))[0]]
    triggerChannelsTwo = channelMapping[1,np.where(np.in1d(channelMapping[0],triggerTwo))[0]]
    cmap = "gist_rainbow"
    colourCoding = TwoInOneOutCoding(cmap=cmap,electrodesToCode=electrodes)
    colours = colourCoding(mappedElectrodes)
    yLabelList = []
    numberOfTotalResponses = 0
    yLabel = 0# the labels we later add on the yaxis
    oneSideLen = np.sqrt(len(inputArray)).astype(int)
    minorTicks = []
    for i in range(oneSideLen):
        temp = 0
        for r in responseStarts[i*oneSideLen:(i+1)*oneSideLen]:
            numberOfTotalResponses += len(r)
            temp += len(r)
            minorTicks.append(yLabel+temp)
        yLabel += temp
        yLabelList.append(yLabel)
    # Open the HDF5 file and read the data

    fig,ax = plt.subplots(dpi=300)
    ax,handles = mapColourToElectrode(ax,voltageMap,mappedElectrodes,colours,legend="Recording Electrodes",stimElectrodes=np.concatenate((mappedTriggersOne,mappedTriggersTwo)),markerSize=10,markerSizeTitle=10)
    ax.set_title("Colour Coded Network")
    ax.legend(handles=handles, bbox_to_anchor=(1, -0.15), loc='lower right', ncol=2, prop={'size': 8})
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    image = responseDelayOverTime(sortedSpikes,responseStarts,channelMapping,colourCoding,window=responseWindow)
    ax.imshow(image,extent=[0, responseWindow / 20, 0, image.shape[0]], aspect='auto')
    for y in yLabelList:
        ax.axhline(y, c="black", lw=0.4)
    ax.set_ylabel(legendLeft)
    ax.set_xlabel("Latency [ms]")
    ax.yaxis.set_major_locator(mticker.FixedLocator(yLabelList))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(minorTicks))
    ax.set_yticks(yLabelList)
    ax.set_yticks(minorTicks, minor=True)
    #ax.set_yticklabels(yLabelList)
    ax2.set_yticks(np.array(yLabelList)-(oneSideLen *nrOfResponses)//2)
    ax2.set_ylabel(legendRight)
    ax2.set_ylim((0,numberOfTotalResponses))
    #ax2.tick_params(axis=u'both', which=u'both',length=0)
    idx_set = [f"{idx}" for idx in plotArray[::oneSideLen,1]]
    ax2.set_yticklabels(idx_set)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("")
    oneSideLen = np.sqrt(len(inputArray)).astype(int)
    countsAll = np.reshape(countAllOG,[oneSideLen,oneSideLen,countAllOG.shape[1],countAllOG.shape[2]])#.transpose((1,0,2,3))
    im = ax.imshow(np.mean(countsAll,axis=(2,3)),origin="lower",cmap=imageColour,extent=(-0.5, countsAll.shape[0]-0.5, -0.5, countsAll.shape[1]-0.5),vmin=0,vmax=max_rate)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg. Spike Count')
    ticks = np.linspace(cbar.vmin, cbar.vmax, 5)
    rounded_ticks = np.round(ticks, 1)
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(rounded_ticks)
    ax.yaxis.set_major_locator(mticker.FixedLocator(yLabelList))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    # Set the ticks to be at the centers of the pixels
    ax.set_xticks(np.arange(countsAll.shape[0]), minor=False)
    ax.set_yticks(np.arange(countsAll.shape[1]), minor=False)
    # Set the tick labels
    ax.set_xticklabels(plotArray[:oneSideLen,0].astype(int), minor=False, rotation=45)
    ax.set_yticklabels(plotArray[::oneSideLen,1].astype(int), minor=False)
    # Show grid on minor ticks (which are in the center of the pixels)
    ax.set_ylabel(legendRight)
    ax.set_xlabel(legendLeft)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("")
    delaysAll = np.reshape(delaysAllOG,[oneSideLen,oneSideLen,delaysAllOG.shape[1],delaysAllOG.shape[2]])#.transpose((1,0,2,3))
    im = ax.imshow(np.mean(delaysAll,axis=(2,3)),origin="lower",cmap=imageColour+"_r",extent=(-0.5, delaysAll.shape[0]-0.5, -0.5, delaysAll.shape[1]-0.5),vmin=min_delay,vmax=responseWindow/20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg. Time to First Spike [ms]')
    ticks = np.linspace(cbar.vmin, cbar.vmax, 7)
    rounded_ticks = np.round(ticks, 0).astype(int)
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(rounded_ticks)
    # Set the ticks to be at the centers of the pixels
    ax.set_xticks(np.arange(delaysAll.shape[0]), minor=False)
    ax.set_yticks(np.arange(delaysAll.shape[1]), minor=False)
    # Set the tick labels
    ax.set_xticklabels(plotArray[:oneSideLen,0], minor=False, rotation=45)
    ax.set_yticklabels(plotArray[::oneSideLen,1], minor=False)
    # Show grid on minor ticks (which are in the center of the pixels)
    ax.set_ylabel(legendRight)
    ax.set_xlabel(legendLeft)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()
    #
    fig, ax = plt.subplots()
    ax.set_title("")
    isiAll = np.reshape(isiAllOG,[oneSideLen,oneSideLen,isiAllOG.shape[1],isiAllOG.shape[2]])#.transpose((1,0,2,3))
    im = ax.imshow(np.mean(isiAll,axis=(2,3)),origin="lower",cmap=imageColour+"_r",extent=(-0.5, isiAll.shape[0]-0.5, -0.5, isiAll.shape[1]-0.5),vmin=min_isi,vmax=responseWindow/20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg. Inter-Spike Interval [ms]')
    ticks = np.linspace(cbar.vmin+0.5, cbar.vmax, 5)
    rounded_ticks = np.round(ticks, 0).astype(int)
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(rounded_ticks)
    # Set the ticks to be at the centers of the pixels
    ax.set_xticks(np.arange(isiAll.shape[0]), minor=False)
    ax.set_yticks(np.arange(isiAll.shape[1]), minor=False)
    # Set the tick labels
    ax.set_xticklabels(plotArray[:oneSideLen,0], minor=False, rotation=45)
    ax.set_yticklabels(plotArray[::oneSideLen,1], minor=False)
    # Show grid on minor ticks (which are in the center of the pixels)
    ax.set_ylabel(legendRight)
    ax.set_xlabel(legendLeft)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("")
    phaseAll = np.reshape(phaseAllOG,[oneSideLen,oneSideLen,phaseAllOG.shape[1],phaseAllOG.shape[2]])#.transpose((1,0,2,3))
    im = ax.imshow(np.mean(phaseAll,axis=(2,3)),origin="lower",cmap=imageColour+"_r",extent=(-0.5, phaseAll.shape[0]-0.5, -0.5, phaseAll.shape[1]-0.5),vmin=min_phase,vmax=2*np.pi)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg. Phase [rad]')
    ticks = np.linspace(cbar.vmin, cbar.vmax, 4)
    rounded_ticks = np.round(ticks, 1)
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(rounded_ticks)
    # Set the ticks to be at the centers of the pixels
    ax.set_xticks(np.arange(phaseAll.shape[0]), minor=False)
    ax.set_yticks(np.arange(phaseAll.shape[1]), minor=False)
    # Set the tick labels
    ax.set_xticklabels(plotArray[:oneSideLen,0], minor=False, rotation=45)
    ax.set_yticklabels(plotArray[::oneSideLen,1], minor=False)
    # Show grid on minor ticks (which are in the center of the pixels)
    ax.set_ylabel(legendRight)
    ax.set_xlabel(legendLeft)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()
    #
    if capacities is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, encoding in enumerate(encodings):
            # Create the figure and axis
            # Plot the curve
            ax.plot(window_list / 20, capacities[i], marker='o', linestyle='-', label=encoding)
            # Add labels and title
        ax.set_xlabel('window size [ms]')
        ax.set_ylabel('Capacity [bits/channel use]')
        ax.set_title(f'Capacity for different encodings')
        ax.legend()
        # Show grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        pdf.savefig(fig, transparent=True, bbox_inches='tight')
        plt.close()

    if labels_el is not None:
        scaling = np.nanmax(np.abs(average_spikes_el), axis=(2), keepdims=True)
        scaling[scaling == 0] = 1
        fig = plot_average_spikes_by_coordinates(average_spikes_el / scaling,
                                                 np.vstack([channelMapping[0] % 220, channelMapping[0] // 220]).T)
        
        pdf.savefig(fig, transparent=True, bbox_inches='tight')
        plt.close()
        
        fig = plot_scatter_by_coordinates(top_components_el[..., :2],
                                          np.vstack([channelMapping[0] % 220, channelMapping[0] // 220]).T,
                                          labels_el)
        pdf.savefig(fig, transparent=True, bbox_inches='tight')
        plt.close()
        
        for s in range(oneSideLen):
            fig, axs = plt.subplots(len(plotArray[s*oneSideLen:(1+s)*oneSideLen])//3+1, 3,dpi=300)
            allElectrodesCount = np.zeros((len(plotArray),len(mappedElectrodes)),dtype=float)
            countsAll = delaysAllOG[s*oneSideLen:(1+s)*oneSideLen]
            for i in range(len(plotArray[s*oneSideLen:(1+s)*oneSideLen])):
                allElectrodesCount[i] = np.mean(countsAll[i],axis=0)
            colourMap = cm.get_cmap("turbo")
            norm = Normalize(vmin=0,vmax=np.max(allElectrodesCount))
            colourMap.set_bad('white')
            for ax in axs.flatten():
                ax.axis("off")
            for i, amp in enumerate(plotArray[s*oneSideLen:(1+s)*oneSideLen]):
                axs.flatten()[i], handles = mapColourToElectrode(axs.flatten()[i],voltageMap, mappedElectrodes,colourMap(norm(allElectrodesCount[i])),markerSize=0.3,legend="Spike Count")
                axs.flatten()[i].set_title(str(amp)+unit)
            # Create a new axis for the colorbar
            cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
            # Create a scalar mappable and add the colorbar to the new axis
            sm = ScalarMappable(cmap="turbo", norm=Normalize(vmin=0, vmax=np.max(allElectrodesCount)))
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Delay [ms]', rotation=90, labelpad=5)
            #plt.legend(handles=handles,bbox_to_anchor=(-13, 0), loc='lower left', ncol=2)
            pdf.savefig(fig, transparent=True, bbox_inches='tight')
            plt.close()

    pdf.close()
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
            voltageMapPath = expSet[3]
            for n in range(nrOfNetworks * len(expSet[4])):
                args = (expSet[4][n // nrOfNetworks], chipAndDIV, n % nrOfNetworks, path, voltageMapPath, expSet[7], False, counter, lock)
                tasks.append(args)

        # Setup tqdm progress bar
        with tqdm(total=len(tasks)) as pbar:
            with mp.Pool(n_cores) as pool:
                # imap_unordered runs tasks asynchronously and unordered
                for _ in pool.imap_unordered(process_wrapper, tasks):
                    pbar.update(1)

# Wrapper function to allow passing args in imap_unordered
def process_wrapper(args):
    plot(*args) """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description=
'''
This file takes the .h5 files produced by biohybrid2DAdvancedProcessing.py and produces a PDF containing all releant plots. 
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("--exp_idx",
                        default="1-34", #"1,2,15-20"
                        type=parse_range,
                        help="Indices of files to be loaded, e.g., 1,2,4-6")
    args = parser.parse_args()
    expSets = [config.experiments[i] for i in args.exp_idx]

    for expSet in tqdm(expSets):
        networks = expSet[0]
        chipAndDIV = expSet[1]
        path = expSet[2]
        voltageMapPath = expSet[3]
        for n in range(len(networks) * len(expSet[4])):
            plot(expSet[8], expSet[4][n // len(networks)], chipAndDIV, networks[n % len(networks)], expSet[7])