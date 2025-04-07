import os, sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
# Here we import the function for auto-selection of the most active electrodes
from interface.selectElectrodes import getElectrodeListsWithSelection
from data.utils.cmos_utils import getCoordinates
from matplotlib import style
plt.style.use('classic')
# Importing MaxLab libraries
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving
import supplementary.stimulusLoopFunctions as stimulusLoopFunctions

# Setting the time and date for the configuration and log files
timeNow = datetime.datetime.now()
timeFormat = timeNow.strftime("%H_%M_%S")
dateFormat = timeNow.strftime("%Y_%m_%d")

voltagemapPath = ...
experimentName = dateFormat+"_"+"sweep1102"+"_"+timeFormat
experimentDirectory =...
loadExistingBool = False
DIV = 23

np.random.seed(0)

autoSelectElectrode = 1 # the number of auto-selected electrodes within each node
experimentPath = os.path.join(experimentDirectory,experimentName)
SETTINGS = []
SETTINGS += ["frequency_sweep"]
#SETTINGS += ["increase_50mV"]
#SETTINGS += ["frequency_sweep_det"]
#SETTINGS += ["increase_10mV"] # Whether to do an amplitude-modulated or frequency-modulated stimulation
outputFolderList = [os.path.join(experimentPath, f'recordings_{setting}/') for setting in SETTINGS]
expConfig = {
    "voltagemapPath":voltagemapPath,
    "experimentName": experimentName,
    "experimentDirectory": experimentDirectory,
    "DIV": DIV,
    "experimentPath": experimentPath,
    "outputFolderList": outputFolderList,
    "autoSelectElectrode": autoSelectElectrode
}
os.makedirs(experimentPath)

for outputFolder in outputFolderList:
    try:
        os.makedirs(outputFolder)
    except:
        pass
# Stim Parameters
pulse_duration = 0.4
sampling_time = 0.05 # ms
selectionFilename = "selection.npy"
voltagemap = np.load(voltagemapPath) # make sure voltage map is in input folder
recordingElectrodes, electrodesSource = getElectrodeListsWithSelection(voltagemap, expConfig["experimentPath"], loadFileBool=loadExistingBool, n_sample=1020, selection_threshold=15)
# Initialize MaxLab system into a defined state
maxlab.util.initialize()
# Correct the zero-mean offset in each electrode reading
maxlab.util.offset()
# Power on MaxLab system
status = maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
print("Power ON status: " + status)
# Loop through the binary voltage map to attain electrode numbers from the area of interest
electrodes = recordingElectrodes
print("Total number of electrodes in isolated area: " + str(len(electrodes)))
# check that this is less than available amplifiers (1024?)
assert len(electrodes) < 1024
stimulusElectrodes = electrodesSource

#############################################
# Visualize recording and stimulation electrodes
#############################################
binaryVoltageMap = np.zeros([120,220])
coords = np.asarray([getCoordinates(i) for i in recordingElectrodes])
binaryVoltageMap[coords[:,[0]],coords[:,[1]]] = 1
# Converting electrode number to XY coordinates to plot
coords = np.asarray([getCoordinates(i) for i in stimulusElectrodes])
# Plotting map of selected stimulation electrodes
plt.imshow(binaryVoltageMap, interpolation="none")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Binary Voltage Map of Selected Circuit\nStimulation Electrodes Plotted in Green", size=20)
if len(coords) != 0:
    plt.scatter(coords[:,1], coords[:,0], color = 'g', marker="x", s=50)

#############################################
# Initialising the chip configuration
#############################################
array = maxlab.chip.Array('stimulation') # 'stimulation' is the token ID to identify the instance on the server
array.reset()
array.clear_selected_electrodes()
# Selecting the electrodes to be routed
array.select_electrodes(electrodes)
# array.select_electrodes([11767, 13081, 11759, 12218])
array.select_stimulation_electrodes(stimulusElectrodes)
array.route()
# Download the prepared array configuration to the chip
array.download()
maxlab.util.offset()
array.save_config(os.path.join(experimentPath,"config.cfg"))
# TODO Validate whether config save is necessary
# Code modification for different order settings of voltage array
wells = range(1)
# Investigating the total number of electrodes routed
routed = []
for electrode in stimulusElectrodes:
    if array.query_amplifier_at_electrode(electrode):
        routed.append(electrode)
# Randomising the order of stimulation electrodes
random.shuffle(routed)
print('Number of stimulation electrons NOT routed: ' + str(len(stimulusElectrodes) - len(routed)))

waitTimeBetween = 0

Saver = maxlab.saving.Saving()
Saver.open_directory(outputFolderList[0])
Saver.set_legacy_format(True)
Saver.group_delete_all()

for well in wells:
    Saver.group_define(well, "routed")

Saver.start_file("Dummy")
Saver.start_recording(wells)

# Run for however many seconds possible in sequence --> send stimulation pulse for x min
time.sleep(1)

# Stop recording data
Saver.stop_recording()
Saver.stop_file()
Saver.group_delete_all()


##################################################################################################
# Function declaration: "stimulate" implements the stimulation sequence
# Located in this notebook for ease of access to variables: to be included in functions script
##################################################################################################
def stimulate_voltage(sequence, voltage, stimElectrodes):
    # Load the configuration saved above
    array.load_config(os.path.join(experimentPath,"config.cfg"))
    stimulation_units = []
    # Connect the electrode to stimulation
    for stimElec in stimElectrodes:
        array.connect_electrode_to_stimulation(stimElec)
        # Check if a stimulation channel can be connected to the electrode
        stimulation_units.append(array.query_stimulation_at_electrode(stimElec))
    # TODO validate whether download is necessary
    # Repeated download of configuration to the chip executed each time to prevent unusual chip behaviour
    array.download()
    maxlab.util.offset()

    if stimulation_units:
        print(f"Used Stim Units: {stimulation_units}")
        Saver = maxlab.saving.Saving()
        Saver.open_directory(outputFolderList[nr])
        Saver.set_legacy_format(True)
        Saver.group_delete_all()

        for well in wells:
            Saver.group_define(well, "routed")
        # Power OFF the stimulation
        for stimulation_unit in stimulation_units:
            stimOFF = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False).connect(False)
            maxlab.send(stimOFF)
        maxlab.send(maxlab.system.DelaySamples(100))

        # Power ON the stimulation
        for stimulation_unit in stimulation_units:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(0))

        filenameStimulus = "DIV_{:0>2}_{:0>5}_amp_{}".format(str(DIV), experimentName, str(voltage[0]))

        Saver.start_file(filenameStimulus)
        Saver.start_recording(wells)

        # Run for however many seconds possible in sequence --> send stimulation pulse for x min
        for _ in range(repetition):
            sequence.send()
            time.sleep(total_time)

        # Stop recording data
        Saver.stop_recording()
        Saver.stop_file()
        Saver.group_delete_all()

        # Turning OFF the stimulation
        for stimulation_unit in stimulation_units:
            stimOFF = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False).connect(False)
            maxlab.send(stimOFF)
        maxlab.send(maxlab.system.DelaySamples(100))
        # Disconnect the electrode from stimulation
        for stimElec in stimElectrodes:
            array.disconnect_electrode_from_stimulation(stimElec)

    else:
        print("\tNo stimulation channel can connect to electrode")
def stimulate_frequency(sequence, stimElectrodes, idx, freq):
    # Load the configuration saved above
    array.load_config(os.path.join(experimentPath,"config.cfg"))
    stimulation_units = []
    # Connect the electrode to stimulation
    for stimElec in stimElectrodes:
        array.connect_electrode_to_stimulation(stimElec)
        # Check if a stimulation channel can be connected to the electrode
        stimulation_units.append(array.query_stimulation_at_electrode(stimElec))
    # Repeated download of configuration to the chip executed each time to prevent unusual chip behaviour
    array.download()

    if stimulation_units:
        print(f"Used Stim Units: {stimulation_units}")
        Saver = maxlab.saving.Saving()
        Saver.open_directory(outputFolderList[nr])
        Saver.set_legacy_format(True)
        Saver.group_delete_all()

        for well in wells:
            Saver.group_define(well, "routed")

        # Power OFF the stimulation
        for stimulation_unit in stimulation_units:
            stimOFF = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False).connect(False)
            maxlab.send(stimOFF)
        maxlab.send(maxlab.system.DelaySamples(100))

        # Power ON the stimulation
        for stimulation_unit in stimulation_units:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(0))

        filenameStimulus = "DIV_{:0>2}_{:0>5}_Freq_{}".format(str(DIV), experimentName, str(int(freq*1000)))

        Saver.start_file(filenameStimulus)
        Saver.start_recording(wells)

        # run for however many seconds possible in sequence --> send stimulation pulse for x min
        for seqRep in range(repetition[idx]):
            sequence.send()
            time.sleep(waitTime[idx])

        # Stop recording data
        Saver.stop_recording()
        Saver.stop_file()
        Saver.group_delete_all()

        # Turning OFF the stimulation
        for stimulation_unit in stimulation_units:
            stimOFF = maxlab.chip.StimulationUnit(stimulation_unit).power_up(False).connect(False)
            maxlab.send(stimOFF)
        maxlab.send(maxlab.system.DelaySamples(100))

        # Disconnect the electrode from stimulation
        for stimElec in stimElectrodes:
            array.disconnect_electrode_from_stimulation(stimElec)

    else:
        print("\tNo stimulation channel can connect to electrode: " + str(stimElec) + "\n")

for nr, s in enumerate(SETTINGS):
    time.sleep(waitTimeBetween)
    waitTimeBetween = 300
    if s in ["increase_50mV","increase_10mV"]:
        if s=="increase_50mV":
            voltageArray = np.arange(100, 801, 50)   # set stim voltage in mV
            np.random.shuffle(voltageArray)
        if s=="increase_10mV":
            voltageArray = np.arange(100, 301, 10)
            np.random.shuffle(voltageArray)
        # Defining constant stimulation conditions
        total_time = 30  # in s
        delay_time = 250.0  # ms
        totalPulses = int((1000 / delay_time) * total_time)
        # Repetition: 3 commands per pulse, repeated 60 times for max of 200 commands per sequence
        repetition = int(np.ceil((total_time * 1000) / (delay_time * totalPulses)))
        interpulse_interval = delay_time - pulse_duration  # ms
        sample_amount = int(interpulse_interval / sampling_time)
    elif s in ["frequency_sweep","frequency_sweep_det"]:
        min_iter = 120
        total_it = 30  # constant number of iterations, converted to seconds for fix time
        fix_time = True
        if s == "frequency_sweep":
            delay_time = np.array([1000, 500, 200, 100, 50, 25, 12.5])  # the time between the onset of two pulses
            np.random.shuffle(delay_time)
        if s == "frequency_sweep_det":
            delay_time = np.array(
                [1000, 500, 400, 250, 200, 125, 100, 80, 50])  # the time between the onset of two pulses
            np.random.shuffle(delay_time)
        if fix_time:
            total_it = ((1000 / delay_time) * total_it).astype(int)
            total_it = [max(min_iter, val) for val in total_it]
        else:
            total_it = np.atleast_1d(total_it)
        simulateFreqAmplitude = int((300 / 2.9))  # the input amplitude for the frequency-modulated experiments
        repetition = np.zeros(len(delay_time),dtype=int)
        pulsePerSequence = np.zeros(len(delay_time),dtype=int)
        interpulse_interval = np.zeros(len(delay_time),dtype=float)
        sample_amount = np.zeros(len(delay_time),dtype=int)
        waitTime = np.zeros(len(delay_time),dtype=float)
        nrOfPulsesPerSequence = 240
        for idx, delay in enumerate(delay_time):
            repetition[idx] = int(total_it[idx%len(total_it)]/nrOfPulsesPerSequence) + 1
            pulsePerSequence[idx] = min(total_it[idx%len(total_it)],nrOfPulsesPerSequence)
            interpulse_interval[idx] = delay - pulse_duration  # ms
            sample_amount[idx] = int(interpulse_interval[idx] / sampling_time)
            waitTime[idx] = min(total_it[idx%len(total_it)],nrOfPulsesPerSequence)*delay/1000

    else:
        raise Exception("Invalid Setting")
    # Creating an instance of Saving object for data collection




    if s in ["increase_50mV","increase_10mV"]:
        voltageBitArray = (voltageArray/2.9).astype(int)
        sample = int(pulse_duration / sampling_time / 2)
        voltageBaseline = 512  # corresponds to 1.6 V

        for voltage in voltageBitArray:

            # Creating the sequence
            sequence = maxlab.Sequence()
            print(f"Now stimulating with {voltage} LSB")
            for rep in range(1, totalPulses):
                sequence.append(maxlab.chip.DAC(0, voltageBaseline - voltage))
                sequence.append(maxlab.system.DelaySamples(sample))
                sequence.append(maxlab.chip.DAC(0, voltageBaseline + voltage))
                sequence.append(maxlab.system.DelaySamples(sample))
                sequence.append(maxlab.chip.DAC(0, voltageBaseline))
                sequence.append(maxlab.system.DelaySamples(sample_amount))

            # Printing the voltage sequence to be applied
            voltageAmp = voltageArray[np.where(voltageBitArray == voltage)]

            # Looping through all routed electrodes and apply a sequence of pulses
            stimulate_voltage(sequence, voltageAmp, routed)

        print("Stimulation complete.")

    if s in ["frequency_sweep","frequency_sweep_det"]:
        for idx, delay in enumerate(delay_time):
            frequency = (1000 / delay).astype(int)
            print("Start stimulate for frequency", frequency)
            sequence = maxlab.Sequence()
            for rep in range(1, pulsePerSequence[idx]):
                stimulusLoopFunctions.append_stimulation_pulse(sequence, simulateFreqAmplitude, pulse_duration,
                                                               sampling_time)
                sequence.append(maxlab.system.DelaySamples(sample_amount[idx]))
            # Looping through all routed electrodes and apply a sequence of pulses
            stimulate_frequency(sequence, routed, idx, frequency)
        print("Stimulation complete.")






