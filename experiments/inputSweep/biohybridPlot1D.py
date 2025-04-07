import sys, os, io
from data.SignalProcessing.evaluate import responseDelayOverTime, averageFrequencyResponsePerElectrode, averageDelayResponse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.backends.backend_pdf
import numpy as np
from data.utils.h5pyUtils import getMetaInfo
from data.SignalProcessing.mapping import TwoInOneOutCoding, Spiral_Colour_Coding, Linear_Colour_Coding
from data.plotting import plot_average_spikes_by_coordinates,plot_scatter_by_coordinates
import glob, os
from data.plotting import mapColourToElectrode
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from matplotlib.colors import Normalize
import config
import argparse
from data.utils.parse_utils import parse_range
from tqdm import tqdm
import h5py
import multiprocessing as mp

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
    fitFunc = True

    unit = config.settings[setting]["unit"]
    legendRight = config.settings[setting]["legend"]
    inputArray = config.settings[setting]["inputArray"]
    nrOfResponses = config.settings[setting]["nrOfResponses"]
    responseWindow =config.settings[setting]["responseWindow"]
    pointColour = config.settings[setting]["pointColour"]
    lineColour = config.settings[setting]["lineColour"]
    plotArray = config.settings[setting]["plotArray"]
    nonLinearity = config.settings[setting]["nonLinearity"]
    nonLinearityFit = config.settings[setting]["nonLinearityFit"]

    if fitFunc:
        lineStyle = "none"
    else:
        lineStyle = "solid"

    routing = routing.flatten()
    electrodes = np.where(routing == 1)[0]
    triggers = np.where(routing > 1)[0]
    mappedElectrodes = channelMapping[0, np.where(np.in1d(channelMapping[0], electrodes))[0]]
    cmap = "gist_rainbow"
    colourCoding = TwoInOneOutCoding(cmap=cmap,electrodesToCode=electrodes)
    colours = colourCoding(mappedElectrodes)

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
                capacities[0,i] = h5file[f"Capacity/Rate/w{w}_c"][:][-1]
                capacities[1,i] = h5file[f"Capacity/Latency/w{w}_c"][:][-1]
                capacities[2,i] = h5file[f"Capacity/ISI/w{w}_c"][:][-1]
                capacities[3, i] = h5file[f"Capacity/Phase/w{w}_c"][:][-1]
        else:
            capacities = None
        sortedSpikes = []
        for i in range(len(inputArray)):
            sortedSpikes.append(spikes[:,spikes[5]==i])

    fig,ax = plt.subplots(dpi=300)
    ax,handles = mapColourToElectrode(ax,voltageMap,mappedElectrodes,colours,legend="Recording Electrodes",stimElectrodes=triggers,markerSize=10,markerSizeTitle=10)
    ax.set_title("Colour Coded Network")
    ax.legend(handles=handles, bbox_to_anchor=(1, -0.15), loc='lower right', ncol=2, prop={'size': 8})
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    yLabelList = []
    numberOfTotalResponses = 0
    yLabel = 0# the labels we later add on the yaxis
    for r in responseStarts:
        numberOfTotalResponses += len(r)
        yLabel += len(r)
        yLabelList.append(yLabel)
    image = responseDelayOverTime(sortedSpikes,responseStarts,channelMapping,colourCoding,window=responseWindow)
    ax.imshow(image,extent=[0, responseWindow / 20, 0, image.shape[0]], aspect='auto')
    for y in yLabelList:
        ax.axhline(y, c="black", lw=0.4)
    ax.set_ylabel("Stimulation Iteration")
    ax.set_xlabel("Latency [ms]")
    ax.yaxis.set_major_locator(mticker.FixedLocator(yLabelList))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_yticks(yLabelList)
    ax.set_yticklabels(yLabelList)
    ax2.set_yticks(np.array(yLabelList)-nrOfResponses//2)
    ax2.set_ylabel(legendRight)
    ax2.set_ylim((0,numberOfTotalResponses))
    #ax2.tick_params(axis=u'both', which=u'both',length=0)
    idx_set = [str(idx) for idx in plotArray]
    ax2.set_yticklabels(idx_set)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    cmapPoints = cm.get_cmap(pointColour)
    normalizePoints = Normalize(vmin=0, vmax=nrOfResponses - 1)
    ax.set_title("Activation Function")
    ax.plot(plotArray,np.mean(countAllOG,axis=(1,2)),color=lineColour,marker="x",linestyle=lineStyle, label="Mean", linewidth=3)
    if fitFunc:
        weights = nonLinearityFit(plotArray, np.mean(countAllOG,axis=(1,2))).x
        ax.plot(np.arange(plotArray[0],plotArray[-1]), nonLinearity(np.arange(plotArray[0], plotArray[-1]), *tuple(weights)),color=lineColour, label="Fit", linewidth=3)
    noiseSamplePlot = np.min(np.diff(plotArray))*0.1
    for i,amp in enumerate(plotArray):
        counts = np.mean(countAllOG[i],axis=-1)
        colors = [cmapPoints(normalizePoints(j)) for j in range(nrOfResponses)]
        sc = ax.scatter(amp*np.ones(nrOfResponses)+np.random.randn(nrOfResponses)*noiseSamplePlot,counts,color=colors,cmap=pointColour,s=3)
    sm = ScalarMappable(cmap=pointColour, norm=Normalize(vmin=0, vmax=nrOfResponses - 1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Response Index')
    ax.set_ylabel("Avg. #Spikes per Electrode")
    ax.set_xlabel(legendRight)
    plt.legend()
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    #TODO: Recode for new average Functions
    fig, axs = plt.subplots(len(plotArray)//4+1, 4,dpi=300)
    allElectrodesCount = np.mean(countAllOG,axis=1)
    colourMap = cm.get_cmap("autumn")
    norm = Normalize(vmin=0,vmax=np.max(allElectrodesCount))
    colourMap.set_bad('white')
    for ax in axs.flatten():
        ax.axis("off")
    for i, amp in enumerate(plotArray):
        axs.flatten()[i], handles = mapColourToElectrode(axs.flatten()[i],voltageMap, mappedElectrodes,colourMap(norm(allElectrodesCount[i])),markerSize=0.3,legend="Spike Count",stimElectrodes=triggers)
        axs.flatten()[i].set_title(str(amp)+unit)
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    # Create a scalar mappable and add the colorbar to the new axis
    sm = ScalarMappable(cmap="autumn", norm=Normalize(vmin=0, vmax=np.max(allElectrodesCount)))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Avg. Spike Count', rotation=90, labelpad=5)
    plt.legend(handles=handles,bbox_to_anchor=(-13, 0), loc='lower left', ncol=2)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(len(plotArray)//4+1, 4,dpi=300)
    colourMap = cm.get_cmap("gist_rainbow")
    norm = Normalize(vmin=0,vmax=responseWindow/20)
    colourMap.set_bad('white')
    for ax in axs.flatten():
        ax.axis("off")
    for i, amp in enumerate(plotArray):
        axs.flatten()[i], handles = mapColourToElectrode(axs.flatten()[i],voltageMap, mappedElectrodes, colourMap(norm(np.mean(delaysAllOG[i],axis=0))),markerSize=0.3,legend="Delay of First Spike",stimElectrodes=triggers)
        axs.flatten()[i].set_title(str(amp)+unit)
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
    # Create a scalar mappable and add the colorbar to the new axis
    sm = ScalarMappable(cmap="gist_rainbow", norm=Normalize(vmin=0, vmax=responseWindow/20))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Delay [ms]', rotation=90, labelpad=5)
    plt.legend(handles=handles,bbox_to_anchor=(-13, 0), loc='lower left', ncol=2)
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    cmapPoints = cm.get_cmap(pointColour)
    normalizePoints = Normalize(vmin=0, vmax=nrOfResponses - 1)
    ax.set_title("Activation Function")
    ax.plot(plotArray,np.mean(delaysAllOG,axis=(1,2)),color=lineColour,marker="x",linestyle=lineStyle, label="Mean", linewidth=3)
    if fitFunc:
        weights = nonLinearityFit(plotArray, np.mean(delaysAllOG,axis=(1,2))).x
        ax.plot(np.arange(plotArray[0],plotArray[-1]), nonLinearity(np.arange(plotArray[0], plotArray[-1]), *tuple(weights)),color=lineColour, label="Fit", linewidth=3)
    noiseSamplePlot = np.min(np.diff(plotArray))*0.1
    for i,amp in enumerate(plotArray):
        counts = np.mean(delaysAllOG[i],axis=-1)
        colors = [cmapPoints(normalizePoints(j)) for j in range(nrOfResponses)]
        sc = ax.scatter(amp*np.ones(nrOfResponses)+np.random.randn(nrOfResponses)*noiseSamplePlot,counts,color=colors,cmap=pointColour,s=3)
    sm = ScalarMappable(cmap=pointColour, norm=Normalize(vmin=0, vmax=nrOfResponses - 1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Response Index')
    ax.set_ylabel("Avg. TTFS per Electrode")
    ax.set_xlabel(legendRight)
    plt.legend()
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    cmapPoints = cm.get_cmap(pointColour)
    normalizePoints = Normalize(vmin=0, vmax=nrOfResponses - 1)
    ax.set_title("Activation Function")
    ax.plot(plotArray,np.mean(isiAllOG,axis=(1,2)),color=lineColour,marker="x",linestyle=lineStyle, label="Mean", linewidth=3)
    if fitFunc:
        weights = nonLinearityFit(plotArray, np.mean(isiAllOG,axis=(1,2))).x
        ax.plot(np.arange(plotArray[0],plotArray[-1]), nonLinearity(np.arange(plotArray[0], plotArray[-1]), *tuple(weights)),color=lineColour, label="Fit", linewidth=3)
    noiseSamplePlot = np.min(np.diff(plotArray))*0.1
    for i,amp in enumerate(plotArray):
        counts = np.mean(isiAllOG[i],axis=-1)
        colors = [cmapPoints(normalizePoints(j)) for j in range(nrOfResponses)]
        sc = ax.scatter(amp*np.ones(nrOfResponses)+np.random.randn(nrOfResponses)*noiseSamplePlot,counts,color=colors,cmap=pointColour,s=3)
    sm = ScalarMappable(cmap=pointColour, norm=Normalize(vmin=0, vmax=nrOfResponses - 1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Response Index')
    ax.set_ylabel("Avg. ISI per Electrode")
    ax.set_xlabel(legendRight)
    plt.legend()
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    cmapPoints = cm.get_cmap(pointColour)
    normalizePoints = Normalize(vmin=0, vmax=nrOfResponses - 1)
    ax.set_title("Activation Function")
    ax.plot(plotArray,np.mean(phaseAllOG,axis=(1,2)),color=lineColour,marker="x",linestyle=lineStyle, label="Mean", linewidth=3)
    if fitFunc:
        weights = nonLinearityFit(plotArray, np.mean(phaseAllOG,axis=(1,2))).x
        ax.plot(np.arange(plotArray[0],plotArray[-1]), nonLinearity(np.arange(plotArray[0], plotArray[-1]), *tuple(weights)),color=lineColour, label="Fit", linewidth=3)
    noiseSamplePlot = np.min(np.diff(plotArray))*0.1
    for i,amp in enumerate(plotArray):
        counts = np.mean(phaseAllOG[i],axis=-1)
        colors = [cmapPoints(normalizePoints(j)) for j in range(nrOfResponses)]
        sc = ax.scatter(amp*np.ones(nrOfResponses)+np.random.randn(nrOfResponses)*noiseSamplePlot,counts,color=colors,cmap=pointColour,s=3)
    sm = ScalarMappable(cmap=pointColour, norm=Normalize(vmin=0, vmax=nrOfResponses - 1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Response Index')
    ax.set_ylabel("Avg. Phase per Electrode")
    ax.set_xlabel(legendRight)
    plt.legend()
    pdf.savefig(fig, transparent=True, bbox_inches='tight')
    plt.close()

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
                args = (expSet[4][n // nrOfNetworks], chipAndDIV, n % nrOfNetworks, path, voltageMapPath, expSet[7], True, counter, lock)
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
This file takes the .h5 files produced by biohybrid1DAdvancedProcessing.py and produces a PDF containing all releant plots. 
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parser.add_argument("--exp_idx",
                        default="201-276", #"1,2,15-20"
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