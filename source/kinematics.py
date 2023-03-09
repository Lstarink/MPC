import numpy as np
import math as mt


def RotationMatrix(angles, inverse=False):
    phi = angles[0]
    psi = angles[1]
    theta = angles[2]

    C1 = np.array([[1, 0, 0],
                   [0, mt.cos(phi), -mt.sin(phi)],
                   [0, mt.sin(phi), mt.cos(phi)]])
    C2 = np.array([[mt.cos(psi), 0, mt.sin(psi)],
                   [0, 1, 0],
                   [-mt.sin(psi), 0, mt.cos(psi)]])
    C3 = np.array([[mt.cos(theta), -mt.sin(theta), 0],
                   [mt.sin(theta), mt.cos(theta), 0],
                   [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(C3, C2), C1)

    if inverse:
        return np.transpose(rotation_matrix)
    return rotation_matrix


def RotateRibs(ribs, orientation):
    rotation_matrix = RotationMatrix(orientation)
    rotated_ribs = np.zeros(np.shape(ribs))
    for index, rib in enumerate(ribs):
        rotated_ribs[index][0][0:3] = np.matmul(rotation_matrix, rib[0])
        rotated_ribs[index][1][0:3] = np.matmul(rotation_matrix, rib[1])

    return rotated_ribs


def TranslateRibs(ribs, location):
    translated_ribs = np.zeros(np.shape(ribs))

    for index, rib in enumerate(ribs):
        translated_ribs[index][0][0:3] = rib[0]+location
        translated_ribs[index][1][0:3] = rib[1]+location

    return translated_ribs