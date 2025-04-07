import numpy as np
import matplotlib.cm as cm
from skimage.feature import canny
from skimage.filters import median
from scipy.signal import convolve2d

# Various colour codings used to give the electrodes some colour.
class Colour_Coding:
    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "plasma"):
        self.colourMap = cm.get_cmap(cmap)
        self.colourMap.set_bad('black')
        self.width = width
        self.electrodes = np.sort(np.unique(electrodesToCode))
        self.colours = np.zeros((len(self.electrodes),4))
    def __call__(self, electrodes):
        mask = np.in1d(electrodes,self.electrodes)
        colours = np.zeros((len(electrodes),4)) + np.nan
        # Use broadcasting to create an array of indices for each element in the first array
        colours[mask] = self.colours[np.searchsorted(self.electrodes, electrodes[mask])]
        return colours

class Linear_Colour_Coding(Colour_Coding):

    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "plasma"):
        super().__init__(electrodesToCode,width,cmap)
        coordsX = self.electrodes % self.width
        coordsY = self.electrodes // self.width
        boundX = [np.min(coordsX), np.max(coordsX)+1]
        boundY = [np.min(coordsY), np.max(coordsY)+1]
        yAxisBool = (boundY[1]-boundY[0]) - (boundX[1]-boundX[0]) > 0
        if yAxisBool:
            bound = boundY
            coords = coordsY - boundY[0]
        else:
            bound = boundX
            coords = coordsX - boundX[0]
        span = bound[1] - bound[0]
        phi = coords / span

        self.colours = self.colourMap(phi)


class Spiral_Colour_Coding(Colour_Coding):

    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "plasma"):
        super().__init__(electrodesToCode,width,cmap)

        coordsX = self.electrodes % self.width
        coordsY = self.electrodes // self.width
        centerX, centerY = np.mean(coordsX), np.mean(coordsY)  # center of the points

        # Calculate distance and angle from center for each point
        dists = np.sqrt((coordsX - centerX)**2 + (coordsY - centerY)**2)
        angles = np.arctan2(coordsY - centerY, coordsX - centerX)

        # Normalize distance and angle to [0, 1]
        dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
        angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))

        # Combine distance and angle to create a spiral effect
        phi = (dists + angles) % 1

        self.colours = self.colourMap(phi)


class TwoInOneOutCoding(Colour_Coding):

    def __init__(self, electrodesToCode: np.ndarray, width: int = 220, cmap: str = "plasma"):
        super().__init__(electrodesToCode,width,cmap)
        coordsX = self.electrodes % self.width
        coordsY = self.electrodes // self.width
        boundX = [np.min(coordsX), np.max(coordsX) + 1]
        boundY = [np.min(coordsY), np.max(coordsY) + 1]
        dummyImage = np.zeros((boundY[1]-boundY[0]+4,boundX[1]-boundX[0]+4))
        dummyImage[coordsY-boundY[0]+2,coordsX-boundX[0]+2] = 1
        edges = convolve2d(dummyImage,np.array([[1,1,1],[1,1,1],[1,1,1]]),mode="same")
        edges = (edges<8).astype(int)
        edges[dummyImage==0] = 0
        dummyImage = dummyImage + edges
        values = dummyImage[coordsY-boundY[0]+2, coordsX-boundX[0]+2]
        FirstSet = np.argwhere(values==1)
        SecondSet = np.argwhere(values==2)
        yAxisBool = (boundY[1] - boundY[0]) - (boundX[1] - boundX[0]) > 0
        centerX, centerY = np.median(coordsX), np.median(coordsY)  # center of the points
        if yAxisBool:
            bound = boundY
            coords = coordsY - boundY[0]
            sign = np.sign(coordsX[SecondSet] - centerX)
        else:
            bound = boundX
            coords = coordsX - boundX[0]
            sign = np.sign(coordsY[SecondSet] - centerY)

        span = bound[1] - bound[0]
        middlePhi = coords[FirstSet] / (2*span)

        self.colours[FirstSet] = self.colourMap(middlePhi)


        # Calculate distance and angle from center for each point
        angles = np.arctan2(coordsY[SecondSet] - centerY, coordsX[SecondSet] - centerX)
        angles = angles/2 + sign*np.pi/2
        # Normalize angle to [0, 1]
        angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))

        self.colours[SecondSet] = self.colourMap(angles/2 + 0.5)
