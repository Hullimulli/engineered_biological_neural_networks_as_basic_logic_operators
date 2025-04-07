from scipy import optimize
import numpy as np

def gaussian(x, height: float = 1, mean: float = 0,std: float = 1):
    return height * np.exp(-(x - mean) ** 2 / (2 * std ** 2))

def sigmoid(x, *args):
    return args[0] / (np.exp(-args[2]*(x+args[1]))+1)+args[3]

def elu(x, *args):
    return args[0] * (np.exp(-args[1] * (x - args[2]))-1) * (1 - np.heaviside(x - args[2], 1))+args[3]

def threeGaussians(x, *args):
    return gaussian(x, args[0], args[1], args[2]) + gaussian(x, args[3], args[4], args[5]) + gaussian(x, args[6], args[7], args[8])

def gaussianFit(xAxis: np.ndarray, yAxis: np.ndarray):
    initValues = np.random.uniform(0.0001,1,9)
    errFunc = lambda p, x, y: np.mean((threeGaussians(x, *p) - y) ** 2)
    return optimize.minimize(errFunc,initValues, args=(xAxis,yAxis))

def sigmoidFit(xAxis: np.ndarray, yAxis: np.ndarray, initValues=np.asarray([0.5,-200,1/(50),0])):
    errFunc = lambda p, x, y: np.mean((sigmoid(x, *p) - y) ** 2)
    return optimize.minimize(errFunc,initValues, args=(xAxis,yAxis))

def eluFit(xAxis: np.ndarray, yAxis: np.ndarray,initValues = np.asarray([1/3,1/80,80,0.2])):
    errFunc = lambda p, x, y: np.mean((elu(x, *p) - y) ** 2)
    return optimize.minimize(errFunc,initValues, args=(xAxis,yAxis))

