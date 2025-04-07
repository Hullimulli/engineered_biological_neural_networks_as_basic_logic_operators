import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import List
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize

def violin_plot_with_map(ax, sample_points, densities, max_aposteriori, data, labels, normalization, colourmap):

    for l in range(len(densities)):
        data_segment = data[labels==l]
        median = np.median(data_segment)
        q25, q75 = np.percentile(data_segment, [25, 75])
        #whisker_low = np.min(data_segment[data_segment > q25 - 1.5 * (q75 - q25)])
        #whisker_high = np.max(data_segment[data_segment < q75 + 1.5 * (q75 - q25)])

        # Define transitions in labels
        transitions = np.argwhere(np.diff(np.concatenate(([-1], max_aposteriori))) != 0)[:,0]
        transitions = np.concatenate((transitions,[len(sample_points)]))
        scaleFactor = 2.1*np.max(densities)
        eps = np.percentile(densities[l] / scaleFactor,q=5)
        for i in range(len(transitions)-1):
            # Define color for the current violin
            colour = colourmap(normalization(max_aposteriori[transitions[i]]))
            x_seg = sample_points[transitions[i]:transitions[i+1]]
            dens = (densities[l,transitions[i]:transitions[i+1]] / scaleFactor)
            # Plot the violin
            ax.fill_betweenx(x_seg, l - dens, l, facecolor=colour, alpha=0.6)
            ax.fill_betweenx(x_seg, l, l + dens, facecolor=colour, alpha=0.6)
            ax.plot(l - dens[dens>eps], x_seg[dens>eps], color="black", linewidth=0.75)
            ax.plot(l + dens[dens>eps], x_seg[dens>eps], color="black", linewidth=0.75)

        # Plot the global boxplot elements outside the loop to overlay on top of all violins
        # Central box
        box_x = [l - 0.05, l - 0.05, l + 0.05, l + 0.05, l - 0.05]
        box_y = [q25, q75, q75, q25, q25]
        ax.plot(box_x, box_y, color='black', linewidth=0.75)

        # Median line
        ax.plot([l - 0.05, l + 0.05], [median, median], color='black', linewidth=1.5)

        # Whiskers
        #ax.plot([l, l], [whisker_low, q25], color='black', linestyle='--')
        #ax.plot([l, l], [q75, whisker_high], color='black', linestyle='--')


def mapColourToElectrode(
        axis: Axes,
        voltageMap: np.ndarray,
        electrodes: np.ndarray,
        colours: np.ndarray,
        legend: str = "",
        stimElectrodes: np.ndarray = None,
        nrBoundaryElectrodesX: int = 5,
        nrBoundaryElectrodesY: int = 5,
        horizontal: bool = True,
        markerSize: float = 5,
        markerSizeTitle: float = 10
)-> (Axes,List[Line2D]):
    """
    Generates a plot of the voltage map, which is binarized into open/covered electrodes with 1/0 respectively.
    The colours are put ontop of the specified electrodes.
    :param axis: Axis of a matplotlib subplot.
    :param voltageMapBinary: The voltage map, where covered electrodes have a value <= 0.
    :param electrodes: Electrodes, for which a blob should be plotted.
    :param colours: Colour of the respective blob.
    :param legend: Legend for the blobs.
    :param stimElectrodes: An x is scattered on top of the stimlation electrodes.
    :param nrBoundaryElectrodesX: The plot is automatically cropped, specifies the padding on the x-axis.
    :param nrBoundaryElectrodesY: The plot is automatically cropped, specifies the padding on the y-axis.
    :param horizontal: If True, the plot is such that the longer axis is the x-axis.
    :param markerSize: Size of the blobs. Crucial for different figure sizes.
    :param markerSizeTitle: Size of the blobs in the legend.
    :return: axis, legend handels
    """

    normalization = Normalize(voltageMap.min(),voltageMap.max())
    voltageMapColourMap = plt.get_cmap("binary_r")
    coordsX = electrodes%voltageMap.shape[1]
    coordsY = electrodes//voltageMap.shape[1]
    coordsStimX, coordsStimY = [], []
    if stimElectrodes is not None:
        coordsStimX = stimElectrodes%voltageMap.shape[1]
        coordsStimY = stimElectrodes//voltageMap.shape[1]
    boundX = [max(int(np.min(np.concatenate((coordsX,coordsStimX)))-nrBoundaryElectrodesX),0),min(int(np.max(np.concatenate((coordsX,coordsStimX)))+nrBoundaryElectrodesX+1),voltageMap.shape[1])]
    boundY = [max(int(np.min(np.concatenate((coordsY,coordsStimY))) - nrBoundaryElectrodesY),0), min(int(np.max(np.concatenate((coordsY,coordsStimY))) + nrBoundaryElectrodesY+1),voltageMap.shape[0])]
    image = voltageMapColourMap(normalization(voltageMap[boundY[0]:boundY[1],boundX[0]:boundX[1]]))[:,:,:3]
    coordsX = coordsX-boundX[0]
    coordsY = coordsY - boundY[0]
    if stimElectrodes is not None:
        coordsStimX = coordsStimX-boundX[0]
        coordsStimY = coordsStimY - boundY[0]
    if horizontal and (image.shape[0]>image.shape[1]) or (not horizontal) and (image.shape[0]<image.shape[1]):
        image = np.transpose(image,[1,0,2])
        coordsX, coordsY = coordsY, coordsX
        coordsStimX, coordsStimY = coordsStimY, coordsStimX

    axis.imshow(image)
    axis.scatter(coordsX,coordsY,s=markerSize,c=colours)

    legend_elements = [Line2D([], [], color="black",markerfacecolor='black', marker='s', linestyle='None',
                          markersize=markerSizeTitle, label='PDMS Structure'),
                       Line2D([], [], color="black",markerfacecolor='white', marker='s', linestyle='None',
                              markersize=markerSizeTitle, label='Free Electrodes'),
                       Line2D([], [], color="black",markerfacecolor='black', marker='o', linestyle='None',
                              markersize=markerSizeTitle, label=legend),
                       ]
    if stimElectrodes is not None:
        legend_elements.append(Line2D([], [], color="black", marker='x', linestyle='None',
                              markersize=markerSizeTitle, label="Stimulation Electrode"))
        axis.scatter(coordsStimX, coordsStimY, c="black", s=markerSize, marker="x")
    axis.axis("off")
    return axis, legend_elements

class fourTileColourMap():
    def __init__(self,minValue,maxValue,exponent: float=0.8):
        self.norm = Normalize(minValue, maxValue)
        self.top_left_color = np.array([1, 0, 0, 1])  # Red
        self.top_right_color = np.array([1, 1, 0, 1])  # Yellow
        self.bottom_left_color = np.array([0, 0, 1, 1])  # Blue
        self.bottom_right_color = np.array([0, 1, 0, 1])  # Green
        self.exponent = 0.8
    def __call__(self, X, Y):
        X = self.norm(np.atleast_1d(X))
        Y = self.norm(np.atleast_1d(Y))

        weight_tl = ((1 - X) ** self.exponent) * ((1 - Y) ** self.exponent)
        weight_tr = (X ** self.exponent) * ((1 - Y) ** self.exponent)
        weight_bl = ((1 - X) ** self.exponent) * (Y ** self.exponent)
        weight_br = (X ** self.exponent) * (Y ** self.exponent)

        weight_sum = weight_tl + weight_tr + weight_bl + weight_br
        weight_tl /= weight_sum
        weight_tr /= weight_sum
        weight_bl /= weight_sum
        weight_br /= weight_sum

        # Calculate the color for each point in the grid
        Z = np.zeros((len(X), 4))
        for i in range(4):  # Iterate over RGB channels
            Z[..., i] = (weight_tl * self.top_left_color[i] +
                         weight_tr * self.top_right_color[i] +
                         weight_bl * self.bottom_left_color[i] +
                         weight_br * self.bottom_right_color[i])
        return np.clip(Z, 0, 1)


def plot_average_spikes_by_coordinates(average_spikes, coords, cmap: str = "gist_rainbow"):
    n_clusters, n_channels, n_samples = average_spikes.shape
    normalization = Normalize(vmin=0,vmax=n_clusters-1)
    colourMap = plt.get_cmap(cmap)
    # Determine grid layout from coordinates
    x_coords, y_coords = coords[:, 0].astype(int), coords[:, 1].astype(int)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1
    # Create a grid layout
    subplot_size = 3  # You can adjust this value to make subplots larger or smaller

    # Calculate total figure size
    fig_width = subplot_size * x_range
    fig_height = subplot_size * y_range

    # Create a grid layout
    fig, axes = plt.subplots(nrows=y_range, ncols=x_range,
                             figsize=(fig_width, fig_height),
                             sharey=True)
    axUsed = np.zeros((y_range, x_range), dtype=bool)
    for idx in range(n_channels):
        x, y = coords[idx]
        ax = axes[int(y - y_min), int(x - x_min)]
        axUsed[int(y - y_min), int(x - x_min)] = True
        for cl in range(n_clusters):
            if not np.isnan(average_spikes[cl, idx, :]).all():
                ax.plot(average_spikes[cl, idx, :], alpha=0.5, c = colourMap(normalization(cl)))  # Plot average spikes for the current channel
        ax.xaxis.set_visible(False)  # Hide x-axis labels
        ax.yaxis.set_visible(False)  # Hide y-axis labels
    # Flatten the 2D array of axes into 1D for easy iteration
    axes = axes.flatten()
    axUsed = axUsed.flatten()
    # Hide any unused subplots
    for idx in range(len(axes)):
        if not axUsed[idx]:
            fig.delaxes(axes[idx])
    # Adjust the layout to minimize white space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjusted spacing between plots

    # Use tight_layout to handle any remaining overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Set a global title
    fig.suptitle('Average Spike Shapes Across Channels', fontsize=16)
    return fig


def plot_scatter_by_coordinates(average_spikes, coords, labels):
    n_clusters, n_channels, n_samples = average_spikes.shape

    # Ensure n_samples is 2 for scatter plot
    assert n_samples == 2, "n_samples dimension must be 2 for scatter plot"

    # Determine grid layout from coordinates
    x_coords, y_coords = coords[:, 0].astype(int), coords[:, 1].astype(int)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    # Create a grid layout
    subplot_size = 3  # You can adjust this value to make subplots larger or smaller

    # Calculate total figure size
    fig_width = subplot_size * x_range
    fig_height = subplot_size * y_range

    # Create a grid layout
    fig, axes = plt.subplots(nrows=y_range, ncols=x_range,
                             figsize=(fig_width, fig_height),
                             sharey=True)
    axUsed = np.zeros((y_range, x_range), dtype=bool)

    # Create a color map for labels
    normalization = Normalize(vmin=-1, vmax=np.nanmax(labels))
    cmap = plt.cm.get_cmap("gist_rainbow")  # You can choose another colormap

    for idx in range(n_channels):
        x, y = coords[idx]
        ax = axes[int(y - y_min), int(x - x_min)]
        axUsed[int(y - y_min), int(x - x_min)] = True

        # Extract the label for color mapping
        label_loc = ~np.isnan(labels[:, idx, 0])
        # Extract the points for the scatter plot
        x_points = average_spikes[label_loc, idx, 0]
        y_points = average_spikes[label_loc, idx, 1]
        label = labels[label_loc, idx, 0]
        # Scatter plot with color based on label
        ax.scatter(x_points, y_points, color=cmap(normalization(label)), alpha=0.5)

        ax.xaxis.set_visible(False)  # Hide x-axis labels
        ax.yaxis.set_visible(False)  # Hide y-axis labels

    # Flatten the 2D array of axes into 1D for easy iteration
    axes = axes.flatten()
    axUsed = axUsed.flatten()

    # Hide any unused subplots
    for idx in range(len(axes)):
        if not axUsed[idx]:
            fig.delaxes(axes[idx])

    # Adjust the layout to minimize white space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjusted spacing between plots

    # Use tight_layout to handle any remaining overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
