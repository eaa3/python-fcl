
# coding: utf8
import numpy

import fcl
import numpy as np



if __name__ == "__main__":
    tree = fcl.fcl.ExOcTree(0.1)
    tree.insertPointCloud(numpy.array([[1.0, 0.0 ,0.0],
                                       [0.0, 0.0, 1.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, 0.0, -1.0]]),
                          numpy.array([0.0, 1.0, 0.0]))
    tree.updateInnerOccupancy()


    fcl_tree = fcl.OcTree(tree)



    print fcl_tree.aabb_center, "LOL"


    if tree.writeBinary("test.bt"):
        print("Create octree file.")
    else:
        print("Cannot create octree file.")
