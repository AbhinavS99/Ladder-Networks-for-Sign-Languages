import csv
import random 
import numpy as np
from time import time

class Data_Extracter:
    def extract_data(self, filename):
        f = open(filename)
        linesr = f.readlines()
        f.close()
        total = 0
        first = True
        lines = []
        labels = []
        for line in linesr:
            if first:
                first = False
                continue
            line = line.strip().split(',')
            line = np.array(map(lambda x: int(x), line))
            labels.append(line[0])
            lines.append(line[1:])
            total += 1
            if total >= 20000:
                break
        return np.array(lines), np.array(labels)

