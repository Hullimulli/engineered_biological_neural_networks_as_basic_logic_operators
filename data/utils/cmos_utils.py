import os
import numpy as np
def getCoordinates(id: int, width: int = 220)->(int,int):
    return id // width, id % width

def getId(y, x, width: int = 220)->int:
    return y * width + x

def createDirectory(path: str):
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)

def createFakeVoltageMap(electrodeChannelMapping: np.ndarray)->np.ndarray:
    """
    Recreates a voltage map solely based on the electrode channel mapping.
    """
    voltageMap = np.zeros((120,220))
    coords = np.array([electrodeChannelMapping[0]%voltageMap.shape[1],electrodeChannelMapping[0]//voltageMap.shape[1]]).T
    voltageMap[coords] = 45
    return voltageMap
