import numpy as np
from data.SignalProcessing.postprocessing import UltimateArtefactBlanking, PeakDetection, SNEO
from supplementary.fitFunction import elu,eluFit,sigmoidFit,sigmoid

inputArrayFrequency = np.stack(np.meshgrid(np.array([1000, 500, 200, 100, 50, 25, 12.5]), np.array([1000, 500, 200, 100, 50, 25, 12.5])), axis=-1).reshape(-1, 2)
expectedPeriodicityFrequency = inputArrayFrequency * 20 - 50
inputArrayFrequency = ((1000 ** 2) / inputArrayFrequency).astype(int)
inputArrayFrequencyOneD = np.array([1000, 500, 200, 100, 50, 25, 12.5])
expectedPeriodicityFrequencyOneD = inputArrayFrequencyOneD * 20 - 50
inputArrayFrequencyOneD = ((1000 ** 2) / inputArrayFrequencyOneD).astype(int)
inputArrayAmplitude = np.stack(np.meshgrid(np.arange(0, 801, 100), np.arange(0, 801, 100)), axis=-1).reshape(-1, 2)
expectedPeriodicityAmplitude = 5000*np.ones((len(inputArrayAmplitude),2))-500
inputArrayAmplitudeDelay = np.stack(np.meshgrid(np.arange(100, 801, 100), np.arange(100, 801, 100)), axis=-1).reshape(-1, 2)
expectedPeriodicityAmplitudeDelay = 5000*np.ones((len(inputArrayAmplitudeDelay),2))-500
sneo = SNEO(spikeThreshold=20)
sneoSensitive = SNEO(spikeThreshold=20)
peakDet = PeakDetection(spikeThreshold=3)
blanking = UltimateArtefactBlanking(blankingBuffer=(-40,0))
save_location = ...

settings = {
    "amplitude_1D": {
        "unit": "mV",
        "legend": f"Stimulation Amplitude In\u2081 mV",
        "inputArray": np.arange(100, 801, 50),
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6,1],
        "nonLinearity": sigmoid,
        "nonLinearityFit": sigmoidFit,
        "plotArray": np.arange(0, 801, 50),
        "mode": 0,
        "expectedPeriodicity": expectedPeriodicityAmplitude[...,0]
    },
    "amplitude_1D_Double": {
        "unit": "mV",
        "legend": f"Stimulation Amplitude In\u2081 mV",
        "inputArray": np.arange(0, 801, 100),
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6,1],
        "nonLinearity": sigmoid,
        "nonLinearityFit": sigmoidFit,
        "plotArray": np.arange(0, 801, 100),
        "mode": 1,
        "expectedPeriodicity": expectedPeriodicityAmplitude[...,0]
    },
    "amplitude_1D_Double_One": {
        "unit": "mV",
        "legend": f"Stimulation Amplitude In\u2081 mV",
        "inputArray": np.arange(0, 801, 100),
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6,1],
        "nonLinearity": sigmoid,
        "nonLinearityFit": sigmoidFit,
        "plotArray": np.arange(0, 801, 100),
        "mode": 2,
        "expectedPeriodicity": expectedPeriodicityAmplitude[...,0]
    },
    "amplitude_1D_Double_Two": {
        "unit": "mV",
        "legend": f"Stimulation Amplitude In\u2081 mV",
        "inputArray": np.arange(0, 801, 100),
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6,1],
        "nonLinearity": sigmoid,
        "nonLinearityFit": sigmoidFit,
        "plotArray": np.arange(0, 801, 100),
        "mode": 3,
        "expectedPeriodicity": expectedPeriodicityAmplitude[...,0]
    },
    "amplitude": {
        "unit": "mV",
        "legendLeft": f"Stimulation Amplitude In\u2081 mV",
        "legendRight": f"Stimulation Amplitude In\u2082 mV",
        "inputArray": inputArrayAmplitude,
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV",
        "imageColour": "RdPu",
        "plotArray": inputArrayAmplitude,
        "pickMinimum": 0,
        "expectedPeriodicity": expectedPeriodicityAmplitude
    },
    "amplitude_rand": {
        "unit": "mV",
        "legendLeft": f"Stimulation Amplitude In\u2081 mV",
        "legendRight": f"Stimulation Amplitude In\u2082 mV",
        "inputArray": inputArrayAmplitude,
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV_rand",
        "imageColour": "RdPu",
        "plotArray": inputArrayAmplitude,
        "pickMinimum": 0,
        "expectedPeriodicity": expectedPeriodicityAmplitude

    },
    "amplitude_delay": {
        "unit": "mV",
        "legendLeft": f"Stimulation Amplitude In\u2081 mV",
        "legendRight": f"Stimulation Amplitude In\u2082 mV",
        "inputArray": inputArrayAmplitudeDelay,
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV_delay",
        "imageColour": "RdPu",
        "plotArray": inputArrayAmplitudeDelay,
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityAmplitudeDelay
    },
    "amplitude_delay_sec": {
        "unit": "mV",
        "legendLeft": f"Stimulation Amplitude In\u2081 mV",
        "legendRight": f"Stimulation Amplitude In\u2082 mV",
        "inputArray": inputArrayAmplitudeDelay,
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV_delay_sec",
        "imageColour": "RdPu",
        "plotArray": inputArrayAmplitudeDelay,
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityAmplitudeDelay
    },
    "amplitude_delay_rand": {
        "unit": "mV",
        "legendLeft": f"Stimulation Amplitude In\u2081 mV",
        "legendRight": f"Stimulation Amplitude In\u2082 mV",
        "inputArray": inputArrayAmplitudeDelay,
        "nrOfResponses": 110,
        "responseWindow": 200,
        "path": "recordings_increase_50mV_delay_rand",
        "imageColour": "RdPu",
        "plotArray": inputArrayAmplitudeDelay,
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityAmplitudeDelay
    },
    "frequency_1D": {
        "unit": "Hz",
        "legend": f"Stimulation Frequency In\u2081 Hz",
        "inputArray": inputArrayFrequencyOneD,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6,1],
        "nonLinearity": elu,
        "nonLinearityFit": eluFit,
        "plotArray": (inputArrayFrequencyOneD / 1000).astype(int),
        "mode": 0,
        "expectedPeriodicity": expectedPeriodicityFrequencyOneD
    },
    "frequency_1D_Double": {
        "unit": "Hz",
        "legend": f"Stimulation Frequency In\u2081 Hz",
        "inputArray": inputArrayFrequencyOneD,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep",
        "pointColour": "cool",
        "lineColour": [1, 0.2, 0.6, 1],
        "nonLinearity": elu,
        "nonLinearityFit": eluFit,
        "plotArray": (inputArrayFrequencyOneD / 1000).astype(int),
        "mode": 1,
        "expectedPeriodicity": expectedPeriodicityFrequencyOneD
    },
    "frequency": {
        "unit": "Hz",
        "legendLeft": f"Stimulation Frequency In\u2081 Hz",
        "legendRight": f"Stimulation Frequency In\u2082 Hz",
        "inputArray": inputArrayFrequency,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep",
        "imageColour": "RdPu",
        "plotArray": (inputArrayFrequency / 1000).astype(int),
        "pickMinimum": 2,
        "expectedPeriodicity": expectedPeriodicityFrequency
    },
    "frequency_rand": {
        "unit": "Hz",
        "legendLeft": f"Stimulation Frequency In\u2081 Hz",
        "legendRight": f"Stimulation Frequency In\u2082 Hz",
        "inputArray": inputArrayFrequency,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep_rand",
        "imageColour": "RdPu",
        "plotArray": (inputArrayFrequency / 1000).astype(int),
        "pickMinimum": 0,
        "expectedPeriodicity": np.ones_like(inputArrayFrequency)+150
    },
    "frequency_delay": {
        "unit": "Hz",
        "legendLeft": f"Stimulation Frequency In\u2081 Hz",
        "legendRight": f"Stimulation Frequency In\u2082 Hz",
        "inputArray": inputArrayFrequency,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep_delay",
        "imageColour": "RdPu",
        "plotArray": (inputArrayFrequency / 1000).astype(int),
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityFrequency
    },
    "frequency_delay_rand": {
        "unit": "Hz",
        "legendLeft": f"Stimulation Frequency In\u2081 Hz",
        "legendRight": f"Stimulation Frequency In\u2082 Hz",
        "inputArray": inputArrayFrequency,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep_delay_rand",
        "imageColour": "RdPu",
        "plotArray": (inputArrayFrequency / 1000).astype(int),
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityFrequency
    },
    "frequency_delay_sec": {
        "unit": "Hz",
        "legendLeft": f"Stimulation Frequency In\u2081 Hz",
        "legendRight": f"Stimulation Frequency In\u2082 Hz",
        "inputArray": inputArrayFrequency,
        "nrOfResponses": 55,
        "responseWindow": 200,
        "path": "recordings_frequency_sweep_delay_sec",
        "imageColour": "RdPu",
        "plotArray": (inputArrayFrequency / 1000).astype(int),
        "pickMinimum": 1,
        "expectedPeriodicity": expectedPeriodicityFrequency[:,[1,0]]
    }
}

# These are my paths, adjust to your paths and setup.
experiments = {
    1: [
        [0],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    2: [
        [1],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude",
            #    "amplitude_delay",
            #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    3: [
        [0],
        "1649_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep2D_1649_11_42_30",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "_bs",
	save_location
    ],
    4: [
        [1],
        "1649_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep2D_1649_11_42_30",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    5: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency",
            "frequency_delay",
            "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    6: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency",
            "frequency_delay",
            "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    7: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency",
            "frequency_delay",
            "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    8: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency",
        #    "frequency_delay"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    9: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency",
        #    "frequency_delay"
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    10: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency",
            "frequency_delay"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    11: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
        #    "frequency_delay_sec",
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    12: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
        #    "frequency_delay_sec",
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    13: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_delay_sec",
            "amplitude",
            "amplitude_delay",
            "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    14: [
        [0],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    15: [
        [1],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    16: [
        [0],
        "1853_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep2D_1853_12_04_01",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    17: [
        [1],
        "1853_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep2D_1853_12_04_01",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    18: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
        #    "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
            "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    19: [
        [1],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
        #    "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
            "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    20: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
        #    "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    21: [
        [1],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
        #    "amplitude"
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    22: [
        [0],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
        #    "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    23: [
        [1],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
        #    "amplitude"
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    24: [
        [2],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
        #    "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    25: [
        [0],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
        #    "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
            "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    26: [
        [1],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
        #    "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
            "amplitude"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    27: [
        [2],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
        #    "frequency",
        #    "frequency_delay",
        #    "frequency_delay_sec",
            "amplitude"
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    28: [
        [0],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_14_57_35",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    29: [
        [1],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_14_57_35",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "amplitude",
        #    "amplitude_delay",
        #    "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    30: [
        [0],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_18_42_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "frequency",
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    31: [
        [1],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_18_42_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "frequency",
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    32: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude",
            "amplitude_delay",
            "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    33: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude",
            "amplitude_delay",
            "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    34: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude",
            "amplitude_delay",
            "amplitude_delay_sec"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    113: [
        [0,1],
        "1216_Killer",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/killer_experiment/2024_05_08_killer_experiment_1216_19_17_41",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/killer_experiment/voltageMapArrays/2024_05_08_1216_JKV.npy",
        [
            "frequency",
            "frequency_rand",
            "frequency_delay",
            "amplitude",
            "amplitude_delay",
            "amplitude_rand",
            "amplitude_delay_sec",
            "amplitude_delay_rand",
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    114: [
        [0,1,2,3],
        "1791_Killer",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/killer_experiment/2024_05_08_killer_experiment_1791_19_00_36",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/killer_experiment/voltageMapArrays/2024_05_08_1791_JKV.npy",
        [
            "frequency",
            "frequency_rand",
            "frequency_delay",
            "frequency_delay_rand",
            "frequency_delay_sec",
            "amplitude",
            "amplitude_delay",
            "amplitude_rand",
            "amplitude_delay_sec",
            "amplitude_delay_rand",
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    201: [
        [0],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    202: [
        [0],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    203: [
        [0],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    204: [
        [1],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    205: [
        [1],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    206: [
        [1],
        "1649_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV22/2024_02_08_sweep2D_1649_11_43_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    207: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    208: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    209: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        210: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        211: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        212: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        213: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        214: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        215: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_17_11_57",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    216: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    217: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    218: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    219: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    220: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
        ],
    221: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    222: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    223: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    224: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_17_05_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    225: [
        [0],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    226: [
        [0],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    227: [
        [0],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        228: [
        [1],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
        229: [
        [1],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    230: [
        [1],
        "1853_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV22/2024_02_08_sweep2D_1853_11_31_17",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    231: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "amplitude_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    232: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "amplitude_1D_Double_One"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    233: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    234: [
        [1],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    235: [
        [0],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    236: [
        [1],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    237: [
        [2],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    238: [
        [0],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_14_57_35",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    239: [
        [1],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_14_57_35",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "amplitude_1D_Double",
            "amplitude_1D_Double_One",
            "amplitude_1D_Double_Two"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    240: [
        [0],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_18_42_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "frequency_1D_Double",
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    276: [
        [1],
        "1742_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1742/2024_01_09_sweep2D_1742_18_42_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1742_JKV.npy",
        [
            "frequency_1D_Double",
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    241: [
        [0],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    242: [
        [1],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    243: [
        [2],
        "1746_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1746/2024_01_02_sweep2D_1746_16_36_24",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1746_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    244: [
        [0],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    245: [
        [1],
        "1838_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_231212/1838/2024_01_02_sweep2D_1838_16_35_02",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/231212_VoltageArrayMaps/2023_12_12_1838_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    246: [
        [0],
        "1853_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep2D_1853_12_04_01",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    247: [
        [1],
        "1853_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep2D_1853_12_04_01",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    248: [
        [0],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    249: [
        [1],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    250: [
        [2],
        "1747_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep2D_1747_12_02_00",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    251: [
        [0],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    252: [
        [1],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    253: [
        [2],
        "1731_DIV22",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep2D_1731_11_19_47",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    254: [
        [0],
        "1649_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep2D_1649_11_42_30",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    255: [
        [1],
        "1649_DIV21",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep2D_1649_11_42_30",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D_Double"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    256: [
        [0],
        "1747_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_51_08",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    257: [
        [1],
        "1747_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_51_08",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    258: [
        [2],
        "1747_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_51_08",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    259: [
        [0],
        "1731_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_37_15",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    260: [
        [1],
        "1731_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_37_15",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    261: [
        [2],
        "1731_DIV22_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_37_15",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    262: [
        [0],
        "1747_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_43_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    263: [
        [1],
        "1747_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_43_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    264: [
        [2],
        "1747_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1747/DIV22/2024_02_14_sweep1747_16_43_06",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1747_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    265: [
        [0],
        "1731_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_46_43",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    266: [
        [1],
        "1731_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_46_43",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    267: [
        [2],
        "1731_DIV22_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240123/1731/DIV22/2024_02_14_sweep1731_16_46_43",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240123_VoltageArrayMaps/2024_01_23_1731_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    268: [
        [0],
        "1649_DIV21_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep1649_15_40_40",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    269: [
        [1],
        "1649_DIV21_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep1649_15_40_40",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    270: [
        [0],
        "1649_DIV21_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep1649_15_56_13",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    271: [
        [1],
        "1649_DIV21_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1649/DIV21/2024_02_07_sweep1649_15_56_13",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1649_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    272: [
        [0],
        "1853_DIV21_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep1853_15_50_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
	save_location
    ],
    273: [
        [1],
        "1853_DIV21_1",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep1853_15_50_39",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    274: [
        [0],
        "1853_DIV21_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep1853_16_01_29",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ],
    275: [
        [1],
        "1853_DIV21_2",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/exp_seed_date_240116/1853/DIV21/2024_02_07_sweep1853_16_01_29",
        "/itet-stor/kjoel/neuronies/biohybrid_cmos/240116_VoltageArrayMaps/2024_01_16_1853_JKV.npy",
        [
            "frequency_1D"
        ],
        sneo,
        blanking,
        "",
        save_location
    ]
}
