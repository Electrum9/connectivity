# Author: Vikram Bhagavatula
# Date: 2022-04-06
# Description: Implementation of Connected-component labeling algorithm

import queue as q
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import sys
import os

def connected_comp(arr, neighbors=None):
    """
    Intakes an n-dimensional array and a neighbors (generator) function which intakes a
    location within the array, and yields all of its neighboring locations.
    """
    def nearby(point):
        """
        Default scheme for generating the neighbors of points, by giving all points in
        near vicinity (+/- 1 offset in each dimension).
        """
        dim = len(arr.shape)
        offsets = it.product(*it.repeat(range(-1,2),dim))
        # the predicate (all) given to filter() rejects 0-tuples which would have no offset
        lower = np.array([0,0,0])
        upper = np.array(arr.shape)
        for t in offsets:
            nb = point + t
            # #print(f"{nb=}")
            if np.all(lower < nb) and np.all(nb < upper):
                #print(f"{nb=}")
                yield tuple(nb)

    if neighbors is None:
        neighbors = nearby

    indices = it.product(*map(range, arr.shape))
    queue = q.Queue()
    max_class = np.max(arr)
    curr_label = max_class + 1
    # enables us to differentiate b/w whether a pixel is of a class or has been labeled

    for idx in indices:
        pix = arr[idx]
        if 0 < pix <= max_class: # unlabeled pixel
            arr[idx] = curr_label
            queue.put((pix, idx))

            while not queue.empty():
                (p, i) = queue.get()
                for n in neighbors(np.array(i)):
                    # #print("on neighbor")
                    focus = arr[n]
                    if focus == p: # pixel being focused on is same kind as p
                        # #print("match found")
                        arr[n] = curr_label
                        queue.put((focus, n))
                        # #print(f"{queue.qsize()=}")
                #print("\n")
            curr_label += 1

    return arr, curr_label - max_class

def main(file):
    arr = np.load(file)
    (name, _) = file.split('.')

    if not os.path.isdir(name): # check if there is a folder for the input images and output images
        os.makedirs(name)
        os.makedirs(name+"/input/")
        os.makedirs(name+"/output/")

    for i in range(arr.shape[-1]): # saving input images
        # cv2.imwrite(f"{name}/input/{name}-{i}-in.png", arr[:,:,i])
        plt.imsave(f"{name}/input/{name}-{i}-in.png", arr[:,:,i], cmap="plasma")

    (out, num) = connected_comp(arr)
    #print(f"{num=}")

    for i in range(arr.shape[-1]): # saving input images
        plt.imsave(f"{name}/output/{name}-{i}-out.png", out[:,:,i], cmap="plasma")
        # final = cv2.applyColorMap(out[:,:,i], cv2.COLORMAP_HOT)
        # cv2.imwrite(f"{name}/output/{name}-{i}-out.png", final)

    return

if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)
