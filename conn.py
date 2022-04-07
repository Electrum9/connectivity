# Author: Vikram Bhagavatula
# Date: 2022-04-06
# Description: Implementation of Connected-component labeling algorithm

import queue as q
import itertools as it
import numpy as np
import matplotlib
import cv2

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
        offsets = filter(all, it.product(*it.repeat(range(-1,2),dim)))
        # the predicate (all) given to filter() rejects 0-tuples which would have no offset
        lower = np.array([0,0,0])
        upper = np.array(arr.shape)
        for t in offsets:
            nb = point + t
            if np.all(lower <= nb) and np.all(nb < upper):
                yield tuple(point + t)

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
                    focus = arr[n]
                    if focus == p: # pixel being focused on is same kind as p
                        arr[n] = curr_label
                        queue.put((focus, n))
            curr_label += 1

    return arr, curr_label - (max_class + 1)

def main():
    pass

if __name__ == "__main__":
    main()
