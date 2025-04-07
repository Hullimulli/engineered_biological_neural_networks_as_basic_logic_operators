import os, sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
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
experimentName = dateFormat+"_"+"sweep"+"_"+timeFormat
experimentDirectory = ...
loadExistingBool = False
DIV = 23
autoSelectElectrode = 1 # the number of auto-selected electrodes within each node
experimentPath = os.path.join(experimentDirectory,experimentName)
SETTINGS = [
    "increase_50mV",
    #"increase_50mV_rand",
    "increase_50mV_delay",
    "increase_50mV_delay_sec",
    #"increase_50mV_delay_rand",
    "frequency_sweep",
    #"frequency_sweep_rand",
    "frequency_sweep_delay",
    "frequency_sweep_delay_sec",
    #"frequency_sweep_delay_rand"
    ]
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
recordingElectrodes, electrodesSource = getElectrodeListsWithSelection(voltagemap, expConfig["experimentPath"], loadFileBool=loadExistingBool, n_sample=1020, selection_threshold=15, multiInputBool=True)
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
stimulusElectrodesOne = [e[0] for e in electrodesSource]
stimulusElectrodesTwo = [e[1] for e in electrodesSource]
binaryVoltageMap = np.zeros([120,220])
coords = np.asarray([getCoordinates(i) for i in recordingElectrodes])
binaryVoltageMap[coords[:,[0]],coords[:,[1]]] = 1
# Converting electrode number to XY coordinates to plot
coordsOne = np.asarray([getCoordinates(i) for i in stimulusElectrodesOne])
coordsTwo = np.asarray([getCoordinates(i) for i in stimulusElectrodesTwo])
# Plotting map of selected stimulation electrodes
plt.imshow(binaryVoltageMap, interpolation="none")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Binary Voltage Map of Selected Circuit\nStimulation Electrodes Plotted in Green", size=20)
if len(coordsOne) != 0:
    plt.scatter(coordsOne[:,1], coordsOne[:,0], color = 'g', marker="x", s=50)
if len(coordsTwo) != 0:
    plt.scatter(coordsTwo[:,1], coordsTwo[:,0], color = 'g', marker="x", s=50)
#############################################
# Initialising the chip configuration
#############################################
array = maxlab.chip.Array('stimulation') # 'stimulation' is the token ID to identify the instance on the server
array.reset()
array.clear_selected_electrodes()
# Selecting the electrodes to be routed
array.select_electrodes(electrodes)
# array.select_electrodes([11767, 13081, 11759, 12218])
array.select_stimulation_electrodes(stimulusElectrodesOne+stimulusElectrodesTwo)
array.route()
# Download the prepared array configuration to the chip
array.download()
maxlab.util.offset()
array.save_config(os.path.join(experimentPath,"config.cfg"))
# TODO Validate whether config save is necessary
# Code modification for different order settings of voltage array
wells = range(1)
# Investigating the total number of electrodes routed
routedOne = []
routedTwo = []
for electrode in stimulusElectrodesOne:
    if array.query_amplifier_at_electrode(electrode):
        routedOne.append(electrode)
for electrode in stimulusElectrodesTwo:
    if array.query_amplifier_at_electrode(electrode):
        routedTwo.append(electrode)
# Randomising the order of stimulation electrodes
random.shuffle(routedOne)
random.shuffle(routedTwo)
print('Number of stimulation electrons NOT routed: ' + str(len(stimulusElectrodesOne+stimulusElectrodesTwo) - len(routedOne+routedTwo)))

waitTimeBetween = 0
np.random.seed(0)

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
def stimulate_voltage(sequence, volOne, volTwo, stimElectrodesOne, stimElectrodesTwo):
    # Load the configuration saved above
    array.load_config(os.path.join(experimentPath,"config.cfg"))
    stimulation_units = []
    stimulation_units_One = []
    stimulation_units_Two = []
    # Connect the electrode to stimulation
    for stimElec in stimElectrodesOne:
        array.connect_electrode_to_stimulation(stimElec)
        stimulation_units_One.append(array.query_stimulation_at_electrode(stimElec))
        stimulation_units.append(stimulation_units_One[-1])
    for stimElec in stimElectrodesTwo:
        array.connect_electrode_to_stimulation(stimElec)
        stimulation_units_Two.append(array.query_stimulation_at_electrode(stimElec))
        stimulation_units.append(stimulation_units_Two[-1])
    if not set(stimulation_units_One).isdisjoint(set(stimulation_units_Two)):
        print(f"Routing Problem, stim one {stimulation_units_One} and stim two {stimulation_units_Two}")
    # TODO validate whether download is necessary
    # Repeated download of configuration to the chip executed each time to prevent unusual chip behaviour
    array.download()
    maxlab.util.offset()
    if stimulation_units:
        print(stimulation_units)
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
        for stimulation_unit in stimulation_units_One:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(1))
        for stimulation_unit in stimulation_units_Two:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(2))

        time.sleep(0.1)
        filenameStimulus = f"DIV_{DIV}_{experimentName}_amp_{volOne}_{volTwo}"

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
        for stimElec in stimElectrodesOne:
            array.disconnect_electrode_from_stimulation(stimElec)
        for stimElec in stimElectrodesTwo:
            array.disconnect_electrode_from_stimulation(stimElec)

    else:
        print("\tNo stimulation channel can connect to electrode")
def stimulate_frequency(sequence, freqOne, freqTwo, stimElectrodesOne, stimElectrodesTwo):
    # Load the configuration saved above
    array.load_config(os.path.join(experimentPath,"config.cfg"))
    stimulation_units = []
    stimulation_units_One = []
    stimulation_units_Two = []
    # Connect the electrode to stimulation
    for stimElec in stimElectrodesOne:
        array.connect_electrode_to_stimulation(stimElec)
        stimulation_units_One.append(array.query_stimulation_at_electrode(stimElec))
        stimulation_units.append(stimulation_units_One[-1])
    for stimElec in stimElectrodesTwo:
        array.connect_electrode_to_stimulation(stimElec)
        stimulation_units_Two.append(array.query_stimulation_at_electrode(stimElec))
        stimulation_units.append(stimulation_units_Two[-1])
    if not set(stimulation_units_One).isdisjoint(set(stimulation_units_Two)):
        print(f"Routing Problem, stim one {stimulation_units_One} and stim two {stimulation_units_Two}")
    # TODO validate whether download is necessary
    # Repeated download of configuration to the chip executed each time to prevent unusual chip behaviour
    array.download()

    if stimulation_units:
        print(stimulation_units)
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
        for stimulation_unit in stimulation_units_One:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(1))
        for stimulation_unit in stimulation_units_Two:
            stimON = maxlab.chip.StimulationUnit(stimulation_unit).power_up(True).connect(True)
            maxlab.send(stimON.set_voltage_mode().dac_source(2))

        time.sleep(0.1)
        filenameStimulus = f"DIV_{DIV}_{experimentName}_Freq_{freqOne*1000}_{freqTwo*1000}"

        Saver.start_file(filenameStimulus)
        Saver.start_recording(wells)

        # run for however many seconds possible in sequence --> send stimulation pulse for x min
        for i in range(total_repeats):
            sequence.send()
            time.sleep(total_time-0.1)
        time.sleep(total_repeats*0.1)
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
        for stimElec in stimElectrodesOne:
            array.disconnect_electrode_from_stimulation(stimElec)
        for stimElec in stimElectrodesTwo:
            array.disconnect_electrode_from_stimulation(stimElec)

    else:
        print("\tNo stimulation channel can connect to some electrode: " + "\n")

for nr, s in enumerate(SETTINGS):
    time.sleep(waitTimeBetween)
    waitTimeBetween = 300
    if s in ["increase_50mV","increase_50mV_rand","increase_50mV_delay","increase_50mV_delay_sec","increase_50mV_delay_rand"]:
        voltageArrayOne = np.arange(0, 801, 100)   # set stim voltage in mV
        voltageArrayTwo = np.arange(0, 801, 100)   # set stim voltage in mV
        voltageArray = np.stack(np.meshgrid(voltageArrayOne, voltageArrayTwo), axis=-1)
        # Reshape the result to have shape (225, 2)
        voltageArray = voltageArray.reshape(-1, 2)
        np.random.shuffle(voltageArray)
        if s == "increase_50mV":
            stdRandomness = 0
            interDelay = 0
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "increase_50mV_rand":
            stdRandomness = 20
            interDelay = 0
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "increase_50mV_delay":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "increase_50mV_delay_sec":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = False
            setDAC = np.array([2,1])
        elif s == "increase_50mV_delay_rand":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = True
            setDAC = np.array([1,2])
        # Defining constant stimulation conditions
        total_time = 30  # in s
        delay_time = 250.0  # ms
        totalPulses = int((1000 / delay_time) * total_time)
        # Repetition: 3 commands per pulse, repeated 60 times for max of 200 commands per sequence
        repetition = int(np.ceil((total_time * 1000) / (delay_time * totalPulses)))
        interpulse_interval = delay_time - pulse_duration  # ms
        sample_amount = int(interpulse_interval / sampling_time)
    elif s in ["frequency_sweep","frequency_sweep_rand","frequency_sweep_delay","frequency_sweep_delay_sec","frequency_sweep_delay_rand"]:
        total_time = 15  # duration of single sequence
        total_repeats = 4
        delay_time_One = np.array([1000, 500, 200, 100, 50, 25, 12.5])  # the time between the onset of two pulses
        delay_time_Two = np.array([1000, 500, 200, 100, 50, 25, 12.5])  # the time between the onset of two pulses
        delay_time = np.stack(np.meshgrid(delay_time_One, delay_time_Two), axis=-1)
        # Reshape the result to have shape (225, 2)
        delay_time = delay_time.reshape(-1, 2)
        np.random.shuffle(delay_time)
        if s == "frequency_sweep":
            stdRandomness = 0
            interDelay = 0
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "frequency_sweep_rand":
            stdRandomness = 20
            interDelay = 0
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "frequency_sweep_delay":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = False
            setDAC = np.array([1,2])
        elif s == "frequency_sweep_delay_sec":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = False
            setDAC = np.array([2,1])
        elif s == "frequency_sweep_delay_rand":
            stdRandomness = 0
            interDelay = 1.2
            randomFirst = True
            setDAC = np.array([1,2])
        total_it = ((1000 / delay_time) * total_time).astype(int)
        simulateFreqAmplitude = int((400 / 2.9))  # the input amplitude for the frequency-modulated experiments
        # The code is based around 1 second sequences, we adapt our usual methodology to this
        combinedPatternAll = []
        for idx, delay in enumerate(delay_time):
            # In the following code, we will construct the list for stimulation at each time point
            # In general, we want to send the information to the routed electrodes like "now turn on your stimulation for _ seconds"
            # Since we have different input frequency on 2 sides, sometimes the electrodes on 2 sides will be simultaneously stimulated, and sometimes only 1 side
            # So the information we want to send is a set of pattern lists with each pattern as [patternIdx, time], where patternIdx denotes the current stimulation setup (i.e. stimulate on the left/right/both/middle), and time dentoes the duration of the pulses
            idx1 = round(1000 / delay[0])  # In 1 s, how many times will the left node being stimulated
            idx2 = round(1000 / delay[1])  # In 1 s, how many times will the right node being stimulated
            commonFactor = round(idx1 * idx2 / math.gcd(idx1,
                                                        idx2))  # In 1 s, at least how many pieces the time should be split to differentiate different stimulation condition
            point1 = round(
                commonFactor / idx1)  # based on the commonFactor, after how many time steps will the left node being stimulated
            point2 = round(
                commonFactor / idx2)  # based on the commonFactor, after how many time steps will the right node being stimulated
            # for the above code we use simple examples: if left and right stimulation frequency is 1 and 20 Hz, then idx1 = 1, idx2 = 20, commonFactor = 20, point1 = 20, point2 = 1; this means we split each 1 s into 20 (commonFactor) time pieces, after 20 (point1) pieces we will stimulate the left node once, and after 1 (point2) piece we will stimulate the right once
            # if left and right stimulation frequency is 2 and 5 Hz, then idx1 = 2, idx2 = 5, commonFactor = 10, point1 = 5, point2 = 2; this means we split each 1 s into 10 (commonFactor) time pieces, after 5 (point1) pieces we will stimulate the left node once, and after 2 (point2) pieces we will stimulate the right once
            sample = round(
                pulse_duration / sampling_time / 2)  # sample number for the whole stimulation pulses (2 pulses)
            interTimeNoStim = round(
                1000 / commonFactor / sampling_time)  # sample number for the interval without stimulation: sometimes we don't have stimulation on the current step, i.e. for the second example above, after splitting the 1 s into 10 (commonFactor) pieces, at the end of the first piece there will be no pulse, since the first stimulation on the left occurs after the 5th (point1) time piece and the first right stimulation occurs after the 2nd (point2) time piece
            interTimeWithStim = round((
                                                  1000 / commonFactor - pulse_duration) / sampling_time)  # sample number for the interval with stimulation
            pulseSample = round(pulse_duration / sampling_time)  # sample number for one pulse

            # create switch pattern
            switchPattern = []  # for the Indicators below: 0 = no stimulus; 1 = post-stimulus; 2 = stimulate the left; 3 = stimulate the right; 4 = stimulate both sides. In theory, after each stimulation, we will have a post-stimulus (Indicator 1) time which refers to the delay for interTimeWithStim.

            for switchIdx in range(commonFactor):
                if switchIdx % point1 == 0 and switchIdx % point2 == 0:  # here we use the % operation to find which sides to stimulation in the current time piece
                    switchPattern.append(4)
                    switchPattern.append(1)
                elif switchIdx % point1 == 0 and switchIdx % point2 != 0:
                    switchPattern.append(2)
                    switchPattern.append(1)
                elif switchIdx % point1 != 0 and switchIdx % point2 == 0:
                    switchPattern.append(3)
                    switchPattern.append(1)
                else:
                    switchPattern.append(0)
            combinedPattern = []
            for stimulusTimeIdx in range(total_time):

                  # To avoid too many fragments in the pattern list, we try to combine the pattern in the situation when after each stimulation, we have a post-stimulus time following a no-stimulus time. These 2 periods without any stimulation could be combined to Indicator 0 denoting the blanking time. For other indicators, its meaning is changed into: 1 = stimulate the left; 2 = stimulate the right; 3 = stimulate both sides. In each pattern, if the indicator is between 1 to 3, the duration will be set to 0; if the indicator is 0, we set the time duration as the actual time wait between 2 pulses
                patternIdx = 0

                while patternIdx < len(switchPattern):

                    if switchPattern[patternIdx] == 4:
                        combinedPattern.append([3, 0])
                        patternIdx += 1
                    elif switchPattern[patternIdx] == 3:
                        combinedPattern.append([2, 0])
                        patternIdx += 1
                    elif switchPattern[patternIdx] == 2:
                        combinedPattern.append([1, 0])
                        patternIdx += 1
                    else:
                        if switchPattern[patternIdx] == 1:
                            temCount = interTimeWithStim
                        else:
                            temCount = interTimeNoStim
                        patternIdx += 1
                        while patternIdx < len(switchPattern):
                            if switchPattern[patternIdx] == 4 or switchPattern[patternIdx] == 3 or switchPattern[
                                patternIdx] == 2:
                                break
                            elif switchPattern[patternIdx] == 1:
                                temCount += interTimeWithStim
                                patternIdx += 1
                            else:
                                temCount += interTimeNoStim
                                patternIdx += 1
                        combinedPattern.append([0, temCount])
                # if we introduce the noise: for the current pattern, we first check whether the duration of the next pattern is long enough to allow us to randomly pick a time between to deliver the stimulation (if interval - dataLength > 0:). If so, the 2 patterns will be spit as 3 patterns, where the first pattern denotes the waiting before the current stimulation, the second pattern denotes the current stimulation, and the third pattern denotes the waiting after the current stimulation. i.e. originally we have 2 patterns [1, 0] and [0, 500], 500 is enough to be split and also get long enough data. If the randomStart is 200, then we create 3 patterns [0, 200], [1, 0], [0, 300]
                if stdRandomness!=0:
                    dataLength=150
                    newCombinedPattern = []
                    idx = 0
                    while idx < len(combinedPattern):
                        interval = combinedPattern[idx + 1][1]
                        if interval - dataLength > 0:
                            if combinedPattern[idx][0] == 3:
                                randomStartOne = random.randint(0, interval - dataLength)
                                randomStartTwo = random.randint(0, interval - dataLength)
                                newCombinedPattern.append([0, min(randomStartOne,randomStartTwo)])
                                if randomStartOne<randomStartTwo:
                                    newCombinedPattern.append([1, 0])
                                    newCombinedPattern.append([0, randomStartTwo - randomStartOne])
                                    newCombinedPattern.append([2, 0])
                                    newCombinedPattern.append([0, interval - randomStartTwo])
                                else:
                                    newCombinedPattern.append([2, 0])
                                    newCombinedPattern.append([0, randomStartOne - randomStartTwo])
                                    newCombinedPattern.append([1, 0])
                                    newCombinedPattern.append([0, interval - randomStartOne])
                            else:
                                randomStart = random.randint(0, interval - dataLength)
                                newCombinedPattern.append([0, randomStart])
                                newCombinedPattern.append([combinedPattern[idx][0], 0])
                                newCombinedPattern.append([0, interval - randomStart])
                            idx = idx + 2
                        else:
                            newCombinedPattern.append(combinedPattern[idx])
                            newCombinedPattern.append(combinedPattern[idx + 1])
                            idx = idx + 2

                    # similar as before, to avoid too many fragments we redo the combination procedure.
                    combinedPattern = newCombinedPattern
                    newCombinedPattern = []
                    idx = 0
                    accumTem = 0
                    while idx < len(combinedPattern):
                        if combinedPattern[idx][0] == 0:
                            accumTem += combinedPattern[idx][1]
                            idx = idx + 1
                        else:
                            newCombinedPattern.append([0, accumTem])
                            newCombinedPattern.append(combinedPattern[idx])
                            accumTem = 0
                            idx = idx + 1
                    combinedPattern = newCombinedPattern
                # When we introduce the time delay, it will only affect the situation with an Indicator of 3. We split that pattern into multiple patterns with the differentiation of first and second stimulation
                if not interDelay==0:

                    newCombinedPattern = []
                    idx = 0
                    while idx < len(combinedPattern):
                        if combinedPattern[idx][0] == 3:
                            nextGap = combinedPattern[idx + 1][1]
                            if randomFirst:
                                firstIndicator = random.randint(1, 2)
                                if firstIndicator == 1:
                                    secondIndicator = 2
                                else:
                                    secondIndicator = 1
                            else:
                                firstIndicator = 1
                                secondIndicator = 2
                            newCombinedPattern.append([firstIndicator, 0])
                            newCombinedPattern.append([0, 3])
                            newCombinedPattern.append([secondIndicator, 0])
                            newCombinedPattern.append([0, nextGap - 3 - pulseSample])
                            idx = idx + 2
                        else:
                            newCombinedPattern.append(combinedPattern[idx])
                            idx = idx + 1
                    combinedPattern = newCombinedPattern
            combinedPatternAll.append(combinedPattern)

    else:
        raise Exception("Invalid Setting")
    # Creating an instance of Saving object for data collection




    if s in ["increase_50mV","increase_50mV_rand","increase_50mV_delay","increase_50mV_delay_sec","increase_50mV_delay_rand"]:
        voltageBitArray = (voltageArray/2.9).astype(int)
        n_samples = int(pulse_duration / sampling_time / 2)
        samplesInterDelay = int(interDelay / sampling_time)
        voltageBaseline = 512  # corresponds to 1.6 V
        for voltage in voltageBitArray:
            # Creating the sequence
            sequence = maxlab.Sequence()
            print(f"Stimulate with {voltage[0]}_{voltage[1]} LSB")
            if samplesInterDelay - 2*n_samples > 0:
                print("Delay applied")
            for rep in range(1, totalPulses):
                randomValue = np.round(np.random.randn(2)*stdRandomness/2.9)
                if randomFirst:
                    np.random.shuffle(setDAC)
                if samplesInterDelay-2*n_samples >= 0:
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 - round(voltage[0] + randomValue[0])))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 + round((voltage[0] + randomValue[0]))))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512))
                    sequence.append(maxlab.system.DelaySamples(max(0,samplesInterDelay-2*n_samples)))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 - round((voltage[1] + randomValue[1]))))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 + round((voltage[1] + randomValue[1]))))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512))
                    sequence.append(maxlab.system.DelaySamples(sample_amount-samplesInterDelay))
                else:
                    sequence.append(maxlab.chip.DAC(1, 512 - round((voltage[0] + randomValue[0]))))
                    sequence.append(maxlab.chip.DAC(2, 512 - round((voltage[1] + randomValue[1]))))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(1, 512 + round((voltage[0] + randomValue[0]))))
                    sequence.append(maxlab.chip.DAC(2, 512 + round((voltage[1] + randomValue[1]))))
                    sequence.append(maxlab.system.DelaySamples(n_samples))
                    sequence.append(maxlab.chip.DAC(1, 512))
                    sequence.append(maxlab.chip.DAC(2, 512))
                    sequence.append(maxlab.system.DelaySamples(sample_amount))

            # Printing the voltage sequence to be applied
            voltageAmpOne = voltageArray[np.where(voltageBitArray == voltage[0])][0]
            voltageAmpTwo = voltageArray[np.where(voltageBitArray == voltage[1])][0]

            # Looping through all routed electrodes and apply a sequence of pulses
            stimulate_voltage(sequence, voltageAmpOne, voltageAmpTwo,routedOne, routedTwo)

        print("Stimulation complete.")

    if s in ["frequency_sweep","frequency_sweep_rand","frequency_sweep_delay","frequency_sweep_delay_sec","frequency_sweep_delay_rand"]:
        for idx, delay in enumerate(delay_time):
            frequency = (1000 / delay).astype(int)
            print(f"Stimulate with {frequency[0]}_{frequency[1]}")
            # Based on the defined patterns, we deliver the stimulation command to the electrodes
            # Just to reiterate the definition for indicators: 0 = blanking; 1 = stimulate left; 2 = stimulate right; 3 = stimulate both; 4 = additional "0" learning signal; 5 = additional "1" learning signal
            sequence = maxlab.Sequence()
            for pattern in combinedPatternAll[idx]:
                if int((pattern[0])) == 3:
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 - simulateFreqAmplitude))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 - simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 + simulateFreqAmplitude))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 + simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512))
                elif int((pattern[0])) == 2:
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 - simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512 + simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[1], 512))
                elif int((pattern[0])) == 1:
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 - simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512 + simulateFreqAmplitude))
                    sequence.append(maxlab.system.DelaySamples(sample))
                    sequence.append(maxlab.chip.DAC(setDAC[0], 512))
                else:
                    sequence.append(maxlab.system.DelaySamples(pattern[1]))
            # Looping through all routed electrodes and apply a sequence of pulses
            stimulate_frequency(sequence, frequency[0],frequency[1],routedOne, routedTwo)
        print("Stimulation complete.")






