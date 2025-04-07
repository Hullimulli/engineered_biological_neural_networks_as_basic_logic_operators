##################################################################################################
# This file contains all functions used in the stimulusLoop notebook
##################################################################################################

import numpy as np

import maxlab
import maxlab.system
import maxlab.chip

##################################################################################################
# voltageMapExtract extracts a window of [height, width] containing a single circuit of interest
# (defined by its centre XY location)
##################################################################################################

def voltageMapExtract(Xcentre, Ycentre, voltageMapInput, height, width):

    X = int(np.absolute(Xcentre))
    Y = int(np.absolute(Ycentre))
    mask = np.zeros(voltageMapInput.shape)

    # Preventing out of bounds exceptions
    if X >= voltageMapInput.shape[1] or Y >= voltageMapInput.shape[1]:
        return matrix

    # Defining dimensions for the mask (within the limits of the matrix size)
    y_lower = max(Y-int(height/2),0)
    y_upper = min(Y+int(height/2),voltageMapInput.shape[0])
    x_lower = max(X-int(width/2),0)
    x_upper = min(X+int(width/2), voltageMapInput.shape[1])

    # Assigning 1 to indices within the area of interest (all other indices have value 0)
    mask[y_lower:y_upper, x_lower:x_upper] += 1

    # Extracting the area of interest from the matrix
    cutMatrix = voltageMapInput * mask

    return cutMatrix
    
##################################################################################################
# voltageMapExtractByElectrodeSelection extracts a window given an electrode list from custom electrode selection
##################################################################################################

def voltageMapExtractByElectrodeSelection(xy_array, voltageMapInput):
	
    mask = np.zeros(voltageMapInput.shape)

    # Assigning 1 to indices within the area of interest (all other indices have value 0)
    mask[xy_array[:,1], xy_array[:,0]] = 1

    # Extracting the area of interest from the matrix
    cutMatrix = voltageMapInput * mask

    return cutMatrix



##################################################################################################
# voltageMapBinaryThreshold returns a binary window of [height, width] containing a single circuit
# of interest (defined by its centre XY location).
##################################################################################################

def voltageMapBinaryThreshold(Xcentre, Ycentre, cutVoltageMapInput, height, width, threshold):

    X = int(np.absolute(Xcentre))
    Y = int(np.absolute(Ycentre))
    mask = np.zeros(cutVoltageMapInput.shape)
    aboveThreshIndices = cutVoltageMapInput > threshold

    # Preventing out of bounds exceptions
    if X >= cutVoltageMapInput.shape[1] or Y >= cutVoltageMapInput.shape[1]:
        mask[aboveThreshIndices] = 1
        return mask

    mask[aboveThreshIndices] = 1

    return mask
    

##################################################################################################
# voltageMapBinarysubSelection returns a binary window 
##################################################################################################

def voltageMapBinarysubSelection(cutVoltageMapInput, Xcentre, Ycentre, height, width, threshhold, dense=False):

    X = int(np.absolute(Xcentre))
    Y = int(np.absolute(Ycentre))
    thr_mask = np.zeros(cutVoltageMapInput.shape)
    bound_mask = np.zeros(cutVoltageMapInput.shape)
    
    # Defining dimensions for the mask (within the limits of the matrix size)
    y_lower = max(Y-int(height/2),0)
    y_upper = min(Y+int(height/2), cutVoltageMapInput.shape[0])
    x_lower = max(X-int(width/2),0)
    x_upper = min(X+int(width/2), cutVoltageMapInput.shape[1])

    # Assigning 1 to indices within the area of interest (all other indices have value 0)
    bound_mask[y_lower:y_upper, x_lower:x_upper] += 1
    
    # Assigning 1 to indices for electrodes above the voltage threshold
    aboveThreshIndices = cutVoltageMapInput > threshhold
    thr_mask[aboveThreshIndices] = 1

    if dense:
        return bound_mask
    
    return bound_mask * thr_mask
    
    
##################################################################################################
# voltageMapSubsampling returns a binary window where the electrode selection has been subsampled
##################################################################################################

def voltageMapSubsampling(binaryVoltageMap, factor, mode='simple'):

    sub_mask = np.zeros(binaryVoltageMap.shape)
    xy = np.array(np.where(binaryVoltageMap == 1)).T
    
    if mode == 'simple':
    	for i in range(0, len(xy), factor):
    	    sub_mask[xy[i,0], xy[i,1]] = 1
    elif mode == 'random':
    	for i in range(len(xy)):
    	    sub_mask[xy[i,0], xy[i,1]] = int(np.random.rand() < (1/factor))
    	
    return sub_mask


##################################################################################################
# convertXYtoElectrodeNumber converts XY coordinates to electrode number
##################################################################################################

def convertXYtoElectrodeNumber(x, y, chipWidth=220):
    return y*chipWidth + x%chipWidth


##################################################################################################
# convertElectrodeNumberToXY converts electrode number to XY coordinates
##################################################################################################

def convertElectrodeNumberToXY(electrodeNumber, chipWidth=220):
    x = int(electrodeNumber/chipWidth)
    y = electrodeNumber % chipWidth
    return x, y


##################################################################################################
# convertVoltsToBits converts a voltage (mV) to bits (0 - 1024)
##################################################################################################
def convertVoltsToBits(voltage):
    return (voltage/2.9).astype(int)


##################################################################################################
# append_stimulation_pulse defines stimulation pulse and sequence
##################################################################################################

def append_stimulation_pulse(sequence, amplitude, pulse_duration, sampling_time):

    sample = int(pulse_duration/sampling_time/2)
    voltageBaseline = 512 # corresponds to 1.6 V


    sequence.append( maxlab.chip.DAC(0, voltageBaseline-amplitude) )
    sequence.append( maxlab.system.DelaySamples(sample) )
    sequence.append( maxlab.chip.DAC(0, voltageBaseline+amplitude) )
    sequence.append( maxlab.system.DelaySamples(sample) )
    sequence.append( maxlab.chip.DAC(0, voltageBaseline) )
    return sequence


##################################################################################################
# calculate_electrode_angle calculates the angle of the electrode location (in degrees) relative
# to the network centre
##################################################################################################
def calculate_electrode_angle(electrodes, chipWidth=220):

    # Defining initial variables
    boundX = [min(electrodes % chipWidth), max(electrodes % chipWidth) + 1]
    boundY = [int(min(electrodes / chipWidth)), int(max(electrodes / chipWidth)) + 1]
    rad = np.zeros(len(electrodes))
    phi = np.zeros(len(electrodes))

    for i, el in enumerate(electrodes):

        # Converting the electrode numbers to their corresponding XY coordinates
        x = electrodes[i] % chipWidth - (boundX[1] + boundX[0]) / 2
        y = electrodes[i] / chipWidth - (boundY[1] + boundY[0]) / 2

        # Calculating the radius of each electrode
        rad[i] = np.sqrt(x ** 2 + y ** 2)

        # Calculating the electrode angle
        if x > 0:
            phi[i] = np.arctan(y / x)
        elif x < 0 and y >= 0:
            phi[i] = np.arctan(y / x) + np.pi
        elif x < 0 and y < 0:
            phi[i] = np.arctan(y / x) - np.pi
        elif x == 0 and y > 0:
            phi[i] = np.pi / 2
        elif x == 0 and y < 0:
            phi[i] = -np.pi / 2

    return phi*(-180/np.pi)


##################################################################################################
# For a given stimuluation electrode, find all neighbouring electrodes within a given radius, which
# are not sufficiently covered by the PDMS mask
##################################################################################################

def stimulusNeighboursMap(stimElectrode, radius, cutVoltageMapInput, threshold, sparse=1):

    mask = np.zeros(cutVoltageMapInput.shape)
    suprElec = []
    # Ensure that stimulation electrode is within the boundaries
    stimCoords = np.asarray(convertElectrodeNumberToXY(stimElectrode))
    if stimCoords[0] >= cutVoltageMapInput.shape[0] or stimCoords[1] >= cutVoltageMapInput.shape[1]:
        print('Stimulation Electrode out of bounds')
        return mask, suprElec

    # Find all electrodes within a given radius from the stim. Electrode and set their mask 1 to
        # one if the exceed the voltage threshold

    for y in range(len(cutVoltageMapInput)):
        for x in range(len(cutVoltageMapInput[y])):
            if cutVoltageMapInput[y][x] >= threshold:
                r = np.linalg.norm(stimCoords-np.asarray([y,x]))
                if r <= radius and cutVoltageMapInput.shape[1] and y <= cutVoltageMapInput.shape[0]:
                    for ring in range(1,radius,sparse):
                        if r >= ring - 0.5 and r <= ring + 0.5:
                            electrode = convertXYtoElectrodeNumber(x, y)
                            if electrode != stimElectrode:
                                mask[y, x] = 1
                                suprElec.append(electrode)


    if 1 not in mask:
        print('No valid supression electrodes were found')
        return mask, suprElec


    suprElecMap = mask * cutVoltageMapInput

    return suprElecMap, suprElec
