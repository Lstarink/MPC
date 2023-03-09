import numpy as np

#dimensions
length = 1

#Cable locations global reference frame A
c01A = np.array([0, 0, 0])
c02A = np.array([0, 0, 1])
c03A = np.array([0, 1, 0])
c04A = np.array([0, 1, 1])
c05A = np.array([1, 0, 0])
c06A = np.array([1, 0, 1])
c07A = np.array([1, 1, 0])
c08A = np.array([1, 1, 1])

#Cable locations block reference frame B
c01B = np.array([1, 1, 1])*length
c02B = np.array([1, 1, -1])*length
c03B = np.array([1, -1, 1])*length
c04B = np.array([1, -1, -1])*length
c05B = np.array([-1, 1, 1])*length
c06B = np.array([-1, 1, -1])*length
c07B = np.array([-1, -1, 1])*length
c08B = np.array([-1, -1, -1])*length

