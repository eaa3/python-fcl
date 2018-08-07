from libcpp cimport bool as cppbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport free
from libc.string cimport memcpy
import inspect

from cython.operator cimport dereference as deref, preincrement as inc, address
cimport numpy as np
import numpy as np
ctypedef np.float64_t DOUBLE_t

cimport fcl_defs as defs
cimport octomap_defs as octomap
cimport dynamicEDT3D_defs as edt
from collision_data import Contact, CostSource, CollisionRequest, ContinuousCollisionRequest, CollisionResult, ContinuousCollisionResult, DistanceRequest, DistanceResult


ctypedef octomap.OccupancyOcTreeBase[octomap.OcTreeNode].tree_iterator* tree_iterator_ptr
ctypedef octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_iterator* leaf_iterator_ptr
ctypedef octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_bbx_iterator* leaf_bbx_iterator_ptr

###############################################################################
# Transforms
###############################################################################
cdef class Transform:
    cdef defs.Transform3f *thisptr

    def __cinit__(self, *args):
        if len(args) == 0:
            self.thisptr = new defs.Transform3f()
        elif len(args) == 1:
            if isinstance(args[0], Transform):
                self.thisptr = new defs.Transform3f(deref((<Transform> args[0]).thisptr))
            else:
                data = np.array(args[0])
                if data.shape == (3,3):
                    self.thisptr = new defs.Transform3f(numpy_to_mat3f(data))
                elif data.shape == (4,):
                    self.thisptr = new defs.Transform3f(numpy_to_quaternion3f(data))
                elif data.shape == (3,):
                    self.thisptr = new defs.Transform3f(numpy_to_vec3f(data))
                else:
                    raise ValueError('Invalid input to Transform().')
        elif len(args) == 2:
            rot = np.array(args[0])
            trans = np.array(args[1]).squeeze()
            if not trans.shape == (3,):
                raise ValueError('Translation must be (3,).')

            if rot.shape == (3,3):
                self.thisptr = new defs.Transform3f(numpy_to_mat3f(rot), numpy_to_vec3f(trans))
            elif rot.shape == (4,):
                self.thisptr = new defs.Transform3f(numpy_to_quaternion3f(rot), numpy_to_vec3f(trans))
            else:
                raise ValueError('Invalid input to Transform().')
        else:
            raise ValueError('Too many arguments to Transform().')

    def __dealloc__(self):
        if self.thisptr:
            free(self.thisptr)

    def getRotation(self):
        return mat3f_to_numpy(self.thisptr.getRotation())

    def getTranslation(self):
        return vec3f_to_numpy(self.thisptr.getTranslation())

    def getQuatRotation(self):
        return quaternion3f_to_numpy(self.thisptr.getQuatRotation())

    def setRotation(self, R):
        self.thisptr.setRotation(numpy_to_mat3f(R))

    def setTranslation(self, T):
        self.thisptr.setTranslation(numpy_to_vec3f(T))

    def setQuatRotation(self, q):
        self.thisptr.setQuatRotation(numpy_to_quaternion3f(q))

###############################################################################
# Collision objects and geometries
###############################################################################

cdef class CollisionObject:
    cdef defs.CollisionObject *thisptr
    cdef defs.PyObject *geom
    cdef cppbool _no_instance

    def __cinit__(self, CollisionGeometry geom=None, Transform tf=None, _no_instance=False):
        if geom is None:
            geom = CollisionGeometry()
        defs.Py_INCREF(<defs.PyObject*> geom)
        self.geom = <defs.PyObject*> geom
        self._no_instance = _no_instance
        if geom.getNodeType() is not None:
            if tf is not None:
                self.thisptr = new defs.CollisionObject(defs.shared_ptr[defs.CollisionGeometry](geom.thisptr), deref(tf.thisptr))
            else:
                self.thisptr = new defs.CollisionObject(defs.shared_ptr[defs.CollisionGeometry](geom.thisptr))
            self.thisptr.setUserData(<void*> self.geom) # Save the python geometry object for later retrieval
        else:
            if not self._no_instance:
                raise ValueError

    def __dealloc__(self):
        if self.thisptr and not self._no_instance:
            free(self.thisptr)
        defs.Py_DECREF(self.geom)

    def getObjectType(self):
        return self.thisptr.getObjectType()

    def getNodeType(self):
        return self.thisptr.getNodeType()

    def getTranslation(self):
        return vec3f_to_numpy(self.thisptr.getTranslation())

    def setTranslation(self, vec):
        self.thisptr.setTranslation(numpy_to_vec3f(vec))
        self.thisptr.computeAABB()

    def getRotation(self):
        return mat3f_to_numpy(self.thisptr.getRotation())

    def setRotation(self, mat):
        self.thisptr.setRotation(numpy_to_mat3f(mat))
        self.thisptr.computeAABB()

    def getQuatRotation(self):
        return quaternion3f_to_numpy(self.thisptr.getQuatRotation())

    def setQuatRotation(self, q):
        self.thisptr.setQuatRotation(numpy_to_quaternion3f(q))
        self.thisptr.computeAABB()

    def getTransform(self):
        rot = self.getRotation()
        trans = self.getTranslation()
        return Transform(rot, trans)

    def setTransform(self, tf):
        self.thisptr.setTransform(deref((<Transform> tf).thisptr))
        self.thisptr.computeAABB()

    def isOccupied(self):
        return self.thisptr.isOccupied()

    def isFree(self):
        return self.thisptr.isFree()

    def isUncertain(self):
        return self.thisptr.isUncertain()

cdef class CollisionGeometry:
    cdef defs.CollisionGeometry *thisptr

    def __cinit__(self):
        pass

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def getNodeType(self):
        if self.thisptr:
            return self.thisptr.getNodeType()
        else:
            return None

    def computeLocalAABB(self):
        if self.thisptr:
            self.thisptr.computeLocalAABB()
        else:
            return None

    property aabb_center:
        def __get__(self):
            if self.thisptr:
                return vec3f_to_numpy(self.thisptr.aabb_center)
            else:
                return None
        def __set__(self, value):
            if self.thisptr:
                self.thisptr.aabb_center[0] = value[0]
                self.thisptr.aabb_center[1] = value[1]
                self.thisptr.aabb_center[2] = value[2]
            else:
                raise ReferenceError

cdef class TriangleP(CollisionGeometry):
    def __cinit__(self, a, b, c):
        self.thisptr = new defs.TriangleP(numpy_to_vec3f(a), numpy_to_vec3f(b), numpy_to_vec3f(c))

    property a:
        def __get__(self):
            return vec3f_to_numpy((<defs.TriangleP*> self.thisptr).a)
        def __set__(self, value):
            (<defs.TriangleP*> self.thisptr).a[0] = <double?> value[0]
            (<defs.TriangleP*> self.thisptr).a[1] = <double?> value[1]
            (<defs.TriangleP*> self.thisptr).a[2] = <double?> value[2]

    property b:
        def __get__(self):
            return vec3f_to_numpy((<defs.TriangleP*> self.thisptr).b)
        def __set__(self, value):
            (<defs.TriangleP*> self.thisptr).b[0] = <double?> value[0]
            (<defs.TriangleP*> self.thisptr).b[1] = <double?> value[1]
            (<defs.TriangleP*> self.thisptr).b[2] = <double?> value[2]

    property c:
        def __get__(self):
            return vec3f_to_numpy((<defs.TriangleP*> self.thisptr).c)
        def __set__(self, value):
            (<defs.TriangleP*> self.thisptr).c[0] = <double?> value[0]
            (<defs.TriangleP*> self.thisptr).c[1] = <double?> value[1]
            (<defs.TriangleP*> self.thisptr).c[2] = <double?> value[2]

cdef class Box(CollisionGeometry):
    def __cinit__(self, x, y, z):
        self.thisptr = new defs.Box(x, y, z)

    property side:
        def __get__(self):
            return vec3f_to_numpy((<defs.Box*> self.thisptr).side)
        def __set__(self, value):
            (<defs.Box*> self.thisptr).side[0] = <double?> value[0]
            (<defs.Box*> self.thisptr).side[1] = <double?> value[1]
            (<defs.Box*> self.thisptr).side[2] = <double?> value[2]

cdef class Sphere(CollisionGeometry):
    def __cinit__(self, radius):
        self.thisptr = new defs.Sphere(radius)

    property radius:
        def __get__(self):
            return (<defs.Sphere*> self.thisptr).radius
        def __set__(self, value):
            (<defs.Sphere*> self.thisptr).radius = <double?> value

cdef class Ellipsoid(CollisionGeometry):
    def __cinit__(self, a, b, c):
        self.thisptr = new defs.Ellipsoid(<double?> a, <double?> b, <double?> c)

    property radii:
        def __get__(self):
            return vec3f_to_numpy((<defs.Ellipsoid*> self.thisptr).radii)
        def __set__(self, values):
            (<defs.Ellipsoid*> self.thisptr).radii = numpy_to_vec3f(values)

cdef class Capsule(CollisionGeometry):
    def __cinit__(self, radius, lz):
        self.thisptr = new defs.Capsule(radius, lz)

    property radius:
        def __get__(self):
            return (<defs.Capsule*> self.thisptr).radius
        def __set__(self, value):
            (<defs.Capsule*> self.thisptr).radius = <double?> value

    property lz:
        def __get__(self):
            return (<defs.Capsule*> self.thisptr).lz
        def __set__(self, value):
            (<defs.Capsule*> self.thisptr).lz = <double?> value

cdef class Cone(CollisionGeometry):
    def __cinit__(self, radius, lz):
        self.thisptr = new defs.Cone(radius, lz)

    property radius:
        def __get__(self):
            return (<defs.Cone*> self.thisptr).radius
        def __set__(self, value):
            (<defs.Cone*> self.thisptr).radius = <double?> value

    property lz:
        def __get__(self):
            return (<defs.Cone*> self.thisptr).lz
        def __set__(self, value):
            (<defs.Cone*> self.thisptr).lz = <double?> value

cdef class Cylinder(CollisionGeometry):
    def __cinit__(self, radius, lz):
        self.thisptr = new defs.Cylinder(radius, lz)

    property radius:
        def __get__(self):
            return (<defs.Cylinder*> self.thisptr).radius
        def __set__(self, value):
            (<defs.Cylinder*> self.thisptr).radius = <double?> value

    property lz:
        def __get__(self):
            return (<defs.Cylinder*> self.thisptr).lz
        def __set__(self, value):
            (<defs.Cylinder*> self.thisptr).lz = <double?> value

cdef class Halfspace(CollisionGeometry):
    def __cinit__(self, n, d):
        self.thisptr = new defs.Halfspace(defs.Vec3f(<double?> n[0],
                                                     <double?> n[1],
                                                     <double?> n[2]),
                                          <double?> d)

    property n:
        def __get__(self):
            return vec3f_to_numpy((<defs.Halfspace*> self.thisptr).n)
        def __set__(self, value):
            (<defs.Halfspace*> self.thisptr).n[0] = <double?> value[0]
            (<defs.Halfspace*> self.thisptr).n[1] = <double?> value[1]
            (<defs.Halfspace*> self.thisptr).n[2] = <double?> value[2]

    property d:
        def __get__(self):
            return (<defs.Halfspace*> self.thisptr).d
        def __set__(self, value):
            (<defs.Halfspace*> self.thisptr).d = <double?> value

cdef class Plane(CollisionGeometry):
    def __cinit__(self, n, d):
        self.thisptr = new defs.Plane(defs.Vec3f(<double?> n[0],
                                                 <double?> n[1],
                                                 <double?> n[2]),
                                      <double?> d)

    property n:
        def __get__(self):
            return vec3f_to_numpy((<defs.Plane*> self.thisptr).n)
        def __set__(self, value):
            (<defs.Plane*> self.thisptr).n[0] = <double?> value[0]
            (<defs.Plane*> self.thisptr).n[1] = <double?> value[1]
            (<defs.Plane*> self.thisptr).n[2] = <double?> value[2]

    property d:
        def __get__(self):
            return (<defs.Plane*> self.thisptr).d
        def __set__(self, value):
            (<defs.Plane*> self.thisptr).d = <double?> value

cdef class BVHModel(CollisionGeometry):
    def __cinit__(self):
        self.thisptr = new defs.BVHModel()

    def num_tries_(self):
        return (<defs.BVHModel*> self.thisptr).num_tris

    def buildState(self):
        return (<defs.BVHModel*> self.thisptr).build_state

    def beginModel(self, num_tris_=0, num_vertices_=0):
        n = (<defs.BVHModel*> self.thisptr).beginModel(<int?> num_tris_, <int?> num_vertices_)
        return n

    def endModel(self):
        n = (<defs.BVHModel*> self.thisptr).endModel()
        return n

    def addVertex(self, x, y, z):
        n = (<defs.BVHModel*> self.thisptr).addVertex(defs.Vec3f(<double?> x, <double?> y, <double?> z))
        return self._check_ret_value(n)

    def addTriangle(self, v1, v2, v3):
        n = (<defs.BVHModel*> self.thisptr).addTriangle(numpy_to_vec3f(v1),
                                                        numpy_to_vec3f(v2),
                                                        numpy_to_vec3f(v3))
        return self._check_ret_value(n)

    def addSubModel(self, verts, triangles):
        cdef vector[defs.Vec3f] ps
        cdef vector[defs.Triangle] tris
        for vert in verts:
            ps.push_back(numpy_to_vec3f(vert))
        for tri in triangles:
            tris.push_back(defs.Triangle(<size_t?> tri[0], <size_t?> tri[1], <size_t?> tri[2]))
        n = (<defs.BVHModel*> self.thisptr).addSubModel(ps, tris)
        return self._check_ret_value(n)

    def _check_ret_value(self, n):
        if n == defs.BVH_OK:
            return True
        elif n == defs.BVH_ERR_MODEL_OUT_OF_MEMORY:
            raise MemoryError("Cannot allocate memory for vertices and triangles")
        elif n == defs.BVH_ERR_BUILD_OUT_OF_SEQUENCE:
            raise ValueError("BVH construction does not follow correct sequence")
        elif n == defs.BVH_ERR_BUILD_EMPTY_MODEL:
            raise ValueError("BVH geometry is not prepared")
        elif n == defs.BVH_ERR_BUILD_EMPTY_PREVIOUS_FRAME:
            raise ValueError("BVH geometry in previous frame is not prepared")
        elif n == defs.BVH_ERR_UNSUPPORTED_FUNCTION:
            raise ValueError("BVH funtion is not supported")
        elif n == defs.BVH_ERR_UNUPDATED_MODEL:
            raise ValueError("BVH model update failed")
        elif n == defs.BVH_ERR_INCORRECT_DATA:
            raise ValueError("BVH data is not valid")
        elif n == defs.BVH_ERR_UNKNOWN:
            raise ValueError("Unknown failure")
        else:
            return False


class NullPointerException(Exception):
    """
    Null pointer exception
    """
    def __init__(self):
        pass

cdef class OcTreeKey:
    """
    OcTreeKey is a container class for internal key addressing.
    The keys count the number of cells (voxels) from the origin as discrete address of a voxel.
    """
    cdef octomap.OcTreeKey *thisptr
    def __cinit__(self):
        self.thisptr = new octomap.OcTreeKey()
    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
    def __setitem__(self, key, value):
        self.thisptr[0][key] = value
    def __getitem__(self, key):
        return self.thisptr[0][key]
    def __richcmp__(self, other, int op):
        if op == 2:
            return (self.thisptr[0][0] == other[0] and \
                    self.thisptr[0][1] == other[1] and \
                    self.thisptr[0][2] == other[2])
        elif op == 3:
            return not (self.thisptr[0][0] == other[0] and \
                        self.thisptr[0][1] == other[1] and \
                        self.thisptr[0][2] == other[2])

cdef class OcTreeNode:
    """
    Nodes to be used in OcTree.
    They represent 3d occupancy grid cells. "value" stores their log-odds occupancy.
    """
    cdef octomap.OcTreeNode *thisptr
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    def addValue(self, float p):
        """
        adds p to the node's logOdds value (with no boundary / threshold checking!)
        """
        if self.thisptr:
            self.thisptr.addValue(p)
        else:
            raise NullPointerException
    def childExists(self, unsigned int i):
        """
        Safe test to check of the i-th child exists,
        first tests if there are any children.
        """
        if self.thisptr:
            return self.thisptr.childExists(i)
        else:
            raise NullPointerException
    def getValue(self):
        if self.thisptr:
            return self.thisptr.getValue()
        else:
            raise NullPointerException
    def setValue(self, float v):
        if self.thisptr:
            self.thisptr.setValue(v)
        else:
            raise NullPointerException
    def getOccupancy(self):
        if self.thisptr:
            return self.thisptr.getOccupancy()
        else:
            raise NullPointerException
    def getLogOdds(self):
        if self.thisptr:
            return self.thisptr.getLogOdds()
        else:
            raise NullPointerException
    def setLogOdds(self, float l):
        if self.thisptr:
            self.thisptr.setLogOdds(l)
        else:
            raise NullPointerException
    def hasChildren(self):
        if self.thisptr:
            return self.thisptr.hasChildren()
        else:
            raise NullPointerException

cdef class iterator_base:
    """
    Iterator over the complete tree (inner nodes and leafs).
    """
    cdef octomap.OcTree *treeptr
    cdef octomap.OccupancyOcTreeBase[octomap.OcTreeNode].iterator_base *thisptr
    def __cinit__(self):
        pass

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def __is_end(self):
        return deref(self.thisptr) == self.treeptr.end_tree()

    def __is_acceseable(self):
        if self.thisptr and self.treeptr:
            if not self.__is_end():
                return True
        return False

    def getCoordinate(self):
        """
        return the center coordinate of the current node
        """
        cdef octomap.Vector3 pt
        if self.__is_acceseable():
            pt = self.thisptr.getCoordinate()
            return np.array((pt.x(), pt.y(), pt.z()))
        else:
            raise NullPointerException

    def getDepth(self):
        if self.__is_acceseable():
            return self.thisptr.getDepth()
        else:
            raise NullPointerException

    def getKey(self):
        """
        the OcTreeKey of the current node
        """
        if self.__is_acceseable():
            key = OcTreeKey()
            key.thisptr[0][0] = self.thisptr.getKey()[0]
            key.thisptr[0][1] = self.thisptr.getKey()[1]
            key.thisptr[0][2] = self.thisptr.getKey()[2]
            return key
        else:
            raise NullPointerException

    def getIndexKey(self):
        """
        the OcTreeKey of the current node, for nodes with depth != maxDepth
        """
        if self.__is_acceseable():
            key = OcTreeKey()
            key.thisptr[0][0] = self.thisptr.getIndexKey()[0]
            key.thisptr[0][1] = self.thisptr.getIndexKey()[1]
            key.thisptr[0][2] = self.thisptr.getIndexKey()[2]
            return key
        else:
            raise NullPointerException

    def getSize(self):
        if self.__is_acceseable():
            return self.thisptr.getSize()
        else:
            raise NullPointerException

    def getX(self):
        if self.__is_acceseable():
            return self.thisptr.getX()
        else:
            raise NullPointerException
    def getY(self):
        if self.__is_acceseable():
            return self.thisptr.getY()
        else:
            raise NullPointerException
    def getZ(self):
        if self.__is_acceseable():
            return self.thisptr.getZ()
        else:
            raise NullPointerException

    def getOccupancy(self):
        if self.__is_acceseable():
            return (<octomap.OcTreeNode>deref(deref(self.thisptr))).getOccupancy()
        else:
            raise NullPointerException

    def getValue(self):
        if self.__is_acceseable():
            return (<octomap.OcTreeNode>deref(deref(self.thisptr))).getValue()
        else:
            raise NullPointerException


cdef class tree_iterator(iterator_base):
    """
    Iterator over the complete tree (inner nodes and leafs).
    """
    def __cinit__(self):
        pass

    def next(self):
        if self.thisptr and self.treeptr:
            if not self.__is_end():
                inc(deref(octomap.static_cast[tree_iterator_ptr](self.thisptr)))
                return self
            else:
                raise StopIteration
        else:
            raise NullPointerException

    def __iter__(self):
        if self.thisptr and self.treeptr:
            while not self.__is_end():
                yield self
                if self.thisptr:
                    inc(deref(octomap.static_cast[tree_iterator_ptr](self.thisptr)))
                else:
                    break
        else:
            raise NullPointerException

    def isLeaf(self):
        if self.__is_acceseable():
            return octomap.static_cast[tree_iterator_ptr](self.thisptr).isLeaf()
        else:
            raise NullPointerException

cdef class leaf_iterator(iterator_base):
    """
    Iterator over the complete tree (leafs).
    """
    def __cinit__(self):
        pass

    def next(self):
        if self.thisptr and self.treeptr:
            if not self.__is_end():
                inc(deref(octomap.static_cast[leaf_iterator_ptr](self.thisptr)))
                return self
            else:
                raise StopIteration
        else:
            raise NullPointerException

    def __iter__(self):
        if self.thisptr and self.treeptr:
            while not self.__is_end():
                yield self
                if self.thisptr:
                    inc(deref(octomap.static_cast[leaf_iterator_ptr](self.thisptr)))
                else:
                    break
        else:
            raise NullPointerException

cdef class leaf_bbx_iterator(iterator_base):
    """
    Iterator over the complete tree (leafs).
    """
    def __cinit__(self):
        pass

    def next(self):
        if self.thisptr and self.treeptr:
            if not self.__is_end():
                inc(deref(octomap.static_cast[leaf_bbx_iterator_ptr](self.thisptr)))
                return self
            else:
                raise StopIteration
        else:
            raise NullPointerException

    def __iter__(self):
        if self.thisptr and self.treeptr:
            while not self.__is_end():
                yield self
                if self.thisptr:
                    inc(deref(octomap.static_cast[leaf_bbx_iterator_ptr](self.thisptr)))
                else:
                    break
        else:
            raise NullPointerException

def _octree_read(filename):
    """
    Read the file header, create the appropriate class and deserialize.
    This creates a new octree which you need to delete yourself.
    """
    cdef octomap.istringstream iss
    cdef ExOcTree tree = ExOcTree(0.1)
    if filename.startswith(b"# Octomap OcTree file"):
        iss.str(string(<char*?>filename, len(filename)))
        del tree.thistree
        tree.thistree = <octomap.OcTree*>tree.thistree.read(<octomap.istream&?>iss)
        return tree
    else:
        del tree.thistree
        tree.thistree = <octomap.OcTree*>tree.thistree.read(string(<char*?>filename))
        return tree

cdef class ExOcTree:
    """
    octomap main map data structure, stores 3D occupancy grid map in an OcTree.
    """
    cdef octomap.OcTree *thistree
    cdef edt.DynamicEDTOctomap *edtptr
    def __cinit__(self, arg):
        import numbers
        if isinstance(arg, numbers.Number):
            self.thistree = new octomap.OcTree(<double?>arg)
        else:
            self.thistree = new octomap.OcTree(string(<char*?>arg))


    def __dealloc__(self):
        pass
        # if self.thistree:
        #     del self.thistree
        if self.edtptr:
            del self.edtptr

    def adjustKeyAtDepth(self, OcTreeKey key, depth):
        cdef octomap.OcTreeKey key_in = octomap.OcTreeKey()
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        cdef octomap.OcTreeKey key_out = self.thistree.adjustKeyAtDepth(key_in, <int?>depth)
        res = OcTreeKey
        res[0] = key_out[0]
        res[1] = key_out[1]
        res[2] = key_out[2]
        return res

    def bbxSet(self):
        return self.thistree.bbxSet()

    def calcNumNodes(self):
        return self.thistree.calcNumNodes()

    def clear(self):
        self.thistree.clear()

    def coordToKey(self, np.ndarray[DOUBLE_t, ndim=1] coord, depth=None):
        cdef octomap.OcTreeKey key
        if depth is None:
            key = self.thistree.coordToKey(octomap.point3d(coord[0],
                                                       coord[1],
                                                       coord[2]))
        else:
            key = self.thistree.coordToKey(octomap.point3d(coord[0],
                                                       coord[1],
                                                       coord[2]),
                                          <unsigned int?>depth)
        res = OcTreeKey()
        res[0] = key[0]
        res[1] = key[1]
        res[2] = key[2]
        return res

    def coordToKeyChecked(self, np.ndarray[DOUBLE_t, ndim=1] coord, depth=None):
        cdef octomap.OcTreeKey key
        cdef cppbool chk
        if depth is None:
            chk = self.thistree.coordToKeyChecked(octomap.point3d(coord[0],
                                                              coord[1],
                                                              coord[2]),
                                                 key)
        else:
            chk = self.thistree.coordToKeyChecked(octomap.point3d(coord[0],
                                                              coord[1],
                                                              coord[2]),
                                                 <unsigned int?>depth,
                                                 key)
        if chk:
            res = OcTreeKey()
            res[0] = key[0]
            res[1] = key[1]
            res[2] = key[2]
            return chk, res
        else:
            return chk, None

    def deleteNode(self, np.ndarray[DOUBLE_t, ndim=1] value, depth=1):
        return self.thistree.deleteNode(octomap.point3d(value[0],
                                                    value[1],
                                                    value[2]),
                                       <int?>depth)

    def castRay(self, np.ndarray[DOUBLE_t, ndim=1] origin,
                np.ndarray[DOUBLE_t, ndim=1] direction,
                np.ndarray[DOUBLE_t, ndim=1] end,
                ignoreUnknownCells=False,
                maxRange=-1.0):
        """
        A ray is cast from origin with a given direction,
        the first occupied cell is returned (as center coordinate).
        If the starting coordinate is already occupied in the tree,
        this coordinate will be returned as a hit.
        """
        cdef octomap.point3d e
        cdef cppbool hit
        hit = self.thistree.castRay(
            octomap.point3d(origin[0], origin[1], origin[2]),
            octomap.point3d(direction[0], direction[1], direction[2]),
            e,
            bool(ignoreUnknownCells),
            <double?>maxRange
        )
        if hit:
            end[0:3] = e.x(), e.y(), e.z()
        return hit

    read = _octree_read

    def write(self, filename=None):
        """
        Write file header and complete tree to file/stream (serialization)
        """
        cdef octomap.ostringstream oss
        if not filename is None:
            return self.thistree.write(string(<char*?>filename))
        else:
            ret = self.thistree.write(<octomap.ostream&?>oss)
            if ret:
                return oss.str().c_str()[:oss.str().length()]
            else:
                return False

    def readBinary(self, filename):
        cdef octomap.istringstream iss
        if filename.startswith(b"# Octomap OcTree binary file"):
            iss.str(string(<char*?>filename, len(filename)))
            return self.thistree.readBinary(<octomap.istream&?>iss)
        else:
            return self.thistree.readBinary(string(<char*?>filename))

    def writeBinary(self, filename=None):
        cdef octomap.ostringstream oss
        if not filename is None:
            return self.thistree.writeBinary(string(<char*?>filename))
        else:
            ret = self.thistree.writeBinary(<octomap.ostream&?>oss)
            if ret:
                return oss.str().c_str()[:oss.str().length()]
            else:
                return False

    def isNodeOccupied(self, node):
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thistree.isNodeOccupied(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        else:
            return self.thistree.isNodeOccupied(<octomap.OcTreeNode>deref(deref((<tree_iterator>node).thisptr)))

    def isNodeAtThreshold(self, node):
        if isinstance(node, OcTreeNode):
            if (<OcTreeNode>node).thisptr:
                return self.thistree.isNodeAtThreshold(deref((<OcTreeNode>node).thisptr))
            else:
                raise NullPointerException
        else:
            return self.thistree.isNodeAtThreshold(<octomap.OcTreeNode>deref(deref((<tree_iterator>node).thisptr)))

    def insertPointCloud(self,
                         np.ndarray[DOUBLE_t, ndim=2] pointcloud,
                         np.ndarray[DOUBLE_t, ndim=1] origin,
                         maxrange=-1.,
                         lazy_eval=False,
                         discretize=False):
        """
        Integrate a Pointcloud (in global reference frame), parallelized with OpenMP.

        Special care is taken that each voxel in the map is updated only once, and occupied
        nodes have a preference over free ones. This avoids holes in the floor from mutual
        deletion.
        :param pointcloud: Pointcloud (measurement endpoints), in global reference frame
        :param origin: measurement origin in global reference frame
        :param maxrange: maximum range for how long individual beams are inserted (default -1: complete beam)
        :param : whether update of inner nodes is omitted after the update (default: false).
        This speeds up the insertion, but you need to call updateInnerOccupancy() when done.
        """
        cdef octomap.Pointcloud pc = octomap.Pointcloud()
        for p in pointcloud:
            pc.push_back(<float>p[0],
                         <float>p[1],
                         <float>p[2])

        self.thistree.insertPointCloud(pc,
                                      octomap.Vector3(<float>origin[0],
                                                   <float>origin[1],
                                                   <float>origin[2]),
                                      <double?>maxrange,
                                      bool(lazy_eval),
                                      bool(discretize))

    def begin_tree(self, maxDepth=0):
        itr = tree_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].tree_iterator(self.thistree.begin_tree(maxDepth))
        itr.treeptr = self.thistree
        return itr

    def begin_leafs(self, maxDepth=0):
        itr = leaf_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_iterator(self.thistree.begin_leafs(maxDepth))
        itr.treeptr = self.thistree
        return itr

    def begin_leafs_bbx(self, np.ndarray[DOUBLE_t, ndim=1] bbx_min, np.ndarray[DOUBLE_t, ndim=1] bbx_max, maxDepth=0):
        itr = leaf_bbx_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_bbx_iterator(self.thistree.begin_leafs_bbx(octomap.point3d(bbx_min[0], bbx_min[1], bbx_min[2]),
                                                                                                                   octomap.point3d(bbx_max[0], bbx_max[1], bbx_max[2]),
                                                                                                                   maxDepth))
        itr.treeptr = self.thistree
        return itr

    def end_tree(self):
        itr = tree_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].tree_iterator(self.thistree.end_tree())
        itr.treeptr = self.thistree
        return itr

    def end_leafs(self):
        itr = leaf_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_iterator(self.thistree.end_leafs())
        itr.treeptr = self.thistree
        return itr

    def end_leafs_bbx(self):
        itr = leaf_bbx_iterator()
        itr.thisptr = new octomap.OccupancyOcTreeBase[octomap.OcTreeNode].leaf_bbx_iterator(self.thistree.end_leafs_bbx())
        itr.treeptr = self.thistree
        return itr

    def getBBXBounds(self):
        cdef octomap.point3d p = self.thistree.getBBXBounds()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXCenter(self):
        cdef octomap.point3d p = self.thistree.getBBXCenter()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXMax(self):
        cdef octomap.point3d p = self.thistree.getBBXMax()
        return np.array((p.x(), p.y(), p.z()))

    def getBBXMin(self):
        cdef octomap.point3d p = self.thistree.getBBXMin()
        return np.array((p.x(), p.y(), p.z()))

    def getRoot(self):
        node = OcTreeNode()
        node.thisptr = self.thistree.getRoot()
        return node

    def getNumLeafNodes(self):
        return self.thistree.getNumLeafNodes()

    def getResolution(self):
        return self.thistree.getResolution()

    def getTreeDepth(self):
        return self.thistree.getTreeDepth()

    def getTreeType(self):
        return self.thistree.getTreeType().c_str()

    def inBBX(self, np.ndarray[DOUBLE_t, ndim=1] p):
        return self.thistree.inBBX(octomap.point3d(p[0], p[1], p[2]))

    def keyToCoord(self, OcTreeKey key, depth=None):
        cdef octomap.OcTreeKey key_in = octomap.OcTreeKey()
        cdef octomap.point3d p = octomap.point3d()
        key_in[0] = key[0]
        key_in[1] = key[1]
        key_in[2] = key[2]
        if depth is None:
            p = self.thistree.keyToCoord(key_in)
        else:
            p = self.thistree.keyToCoord(key_in, <int?>depth)
        return np.array((p.x(), p.y(), p.z()))

    def memoryFullGrid(self):
        return self.thistree.memoryFullGrid()

    def memoryUsage(self):
        return self.thistree.memoryUsage()

    def memoryUsageNode(self):
        return self.thistree.memoryUsageNode()

    def resetChangeDetection(self):
        """
        Reset the set of changed keys. Call this after you obtained all changed nodes.
        """
        self.thistree.resetChangeDetection()

    def search(self, value, depth=0):
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            node.thisptr = self.thistree.search(octomap.OcTreeKey(<unsigned short int>value[0],
                                                              <unsigned short int>value[1],
                                                              <unsigned short int>value[2]),
                                               <unsigned int?>depth)
        else:
            node.thisptr = self.thistree.search(<double>value[0],
                                               <double>value[1],
                                               <double>value[2],
                                               <unsigned int?>depth)
        return node

    def setBBXMax(self, np.ndarray[DOUBLE_t, ndim=1] max):
        """
        sets the maximum for a query bounding box to use
        """
        self.thistree.setBBXMax(octomap.point3d(max[0], max[1], max[2]))

    def setBBXMin(self, np.ndarray[DOUBLE_t, ndim=1] min):
        """
        sets the minimum for a query bounding box to use
        """
        self.thistree.setBBXMin(octomap.point3d(min[0], min[1], min[2]))

    def setResolution(self, double r):
        """
        Change the resolution of the octree, scaling all voxels. This will not preserve the (metric) scale!
        """
        self.thistree.setResolution(r)

    def size(self):
        return self.thistree.size()

    def toMaxLikelihood(self):
        """
        Creates the maximum likelihood map by calling toMaxLikelihood on all tree nodes,
        setting their occupancy to the corresponding occupancy thresholds.
        """
        self.thistree.toMaxLikelihood()

    def updateNodes(self, values, update, lazy_eval=False):
        """
        Integrate occupancy measurements and Manipulate log_odds value of voxel directly. 
        """
        if values is None or len(values) == 0:
            return
        if isinstance(values[0], OcTreeKey):
            if isinstance(update, bool):
                for v in values:
                    self.thistree.updateNode(octomap.OcTreeKey(<unsigned short int>v[0],
                                                           <unsigned short int>v[1],
                                                           <unsigned short int>v[2]),
                                            <cppbool>update,
                                            <cppbool?>lazy_eval)
            else:
                for v in values:
                    self.thistree.updateNode(octomap.OcTreeKey(<unsigned short int>v[0],
                                                           <unsigned short int>v[1],
                                                           <unsigned short int>v[2]),
                                            <float?>update,
                                            <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                for v in values:
                    self.thistree.updateNode(<double?>v[0],
                                            <double?>v[1],
                                            <double?>v[2],
                                            <cppbool>update,
                                            <cppbool?>lazy_eval)
            else:
                for v in values:
                    self.thistree.updateNode(<double?>v[0],
                                            <double?>v[1],
                                            <double?>v[2],
                                            <float?>update,
                                            <cppbool?>lazy_eval)

    def updateNode(self, value, update, lazy_eval=False):
        """
        Integrate occupancy measurement and Manipulate log_odds value of voxel directly. 
        """
        node = OcTreeNode()
        if isinstance(value, OcTreeKey):
            if isinstance(update, bool):
                node.thisptr = self.thistree.updateNode(octomap.OcTreeKey(<unsigned short int>value[0],
                                                                      <unsigned short int>value[1],
                                                                      <unsigned short int>value[2]),
                                                       <cppbool>update,
                                                       <cppbool?>lazy_eval)
            else:
                node.thisptr = self.thistree.updateNode(octomap.OcTreeKey(<unsigned short int>value[0],
                                                                      <unsigned short int>value[1],
                                                                      <unsigned short int>value[2]),
                                                       <float?>update,
                                                       <cppbool?>lazy_eval)
        else:
            if isinstance(update, bool):
                node.thisptr = self.thistree.updateNode(<double?>value[0],
                                                       <double?>value[1],
                                                       <double?>value[2],
                                                       <cppbool>update,
                                                       <cppbool?>lazy_eval)
            else:
                node.thisptr = self.thistree.updateNode(<double?>value[0],
                                                       <double?>value[1],
                                                       <double?>value[2],
                                                       <float?>update,
                                                       <cppbool?>lazy_eval)
        return node

    def updateInnerOccupancy(self):
        """
        Updates the occupancy of all inner nodes to reflect their children's occupancy.
        """
        self.thistree.updateInnerOccupancy()

    def useBBXLimit(self, enable):
        """
        use or ignore BBX limit (default: ignore)
        """
        self.thistree.useBBXLimit(bool(enable))

    def volume(self):
        return self.thistree.volume()

    def getClampingThresMax(self):
        return self.thistree.getClampingThresMax()

    def getClampingThresMaxLog(self):
        return self.thistree.getClampingThresMaxLog()

    def getClampingThresMin(self):
        return self.thistree.getClampingThresMin()

    def getClampingThresMinLog(self):
        return self.thistree.getClampingThresMinLog()

    def getOccupancyThres(self):
        return self.thistree.getOccupancyThres()

    def getOccupancyThresLog(self):
        return self.thistree.getOccupancyThresLog()

    def getProbHit(self):
        return self.thistree.getProbHit()

    def getProbHitLog(self):
        return self.thistree.getProbHitLog()

    def getProbMiss(self):
        return self.thistree.getProbMiss()

    def getProbMissLog(self):
        return self.thistree.getProbMissLog()

    def setClampingThresMax(self, double thresProb):
        self.thistree.setClampingThresMax(thresProb)

    def setClampingThresMin(self, double thresProb):
        self.thistree.setClampingThresMin(thresProb)

    def setOccupancyThres(self, double prob):
        self.thistree.setOccupancyThres(prob)

    def setProbHit(self, double prob):
        self.thistree.setProbHit(prob)

    def setProbMiss(self, double prob):
        self.thistree.setProbMiss(prob)

    def getMetricSize(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thistree.getMetricSize(x, y, z)
        return (x, y, z)

    def getMetricMin(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thistree.getMetricMin(x, y, z)
        return (x, y, z)

    def getMetricMax(self):
        cdef double x = 0
        cdef double y = 0
        cdef double z = 0
        self.thistree.getMetricMax(x, y, z)
        return (x, y, z)

    def expandNode(self, node):
        self.thistree.expandNode((<OcTreeNode>node).thisptr)

    def createNodeChild(self, node, int idx):
        child = OcTreeNode()
        child.thisptr = self.thistree.createNodeChild((<OcTreeNode>node).thisptr, idx)
        return child

    def getNodeChild(self, node, int idx):
        child = OcTreeNode()
        child.thisptr = self.thistree.getNodeChild((<OcTreeNode>node).thisptr, idx)
        return child

    def isNodeCollapsible(self, node):
        return self.thistree.isNodeCollapsible((<OcTreeNode>node).thisptr)

    def deleteNodeChild(self, node, int idx):
        self.thistree.deleteNodeChild((<OcTreeNode>node).thisptr, idx)

    def pruneNode(self, node):
        return self.thistree.pruneNode((<OcTreeNode>node).thisptr)

    def dynamicEDT_generate(self, maxdist,
                            np.ndarray[DOUBLE_t, ndim=1] bbx_min,
                            np.ndarray[DOUBLE_t, ndim=1] bbx_max,
                            treatUnknownAsOccupied=False):
        self.edtptr = new edt.DynamicEDTOctomap(<float?>maxdist,
                                                self.thistree,
                                                octomap.point3d(bbx_min[0], bbx_min[1], bbx_min[2]),
                                                octomap.point3d(bbx_max[0], bbx_max[1], bbx_max[2]),
                                                <cppbool?>treatUnknownAsOccupied)

    def dynamicEDT_checkConsistency(self):
        if self.edtptr:
            return self.edtptr.checkConsistency()
        else:
            raise NullPointerException

    def dynamicEDT_update(self, updateRealDist):
        if self.edtptr:
            self.edtptr.update(<cppbool?>updateRealDist)
        else:
            raise NullPointerException

    def dynamicEDT_getMaxDist(self):
        if self.edtptr:
            return self.edtptr.getMaxDist()
        else:
            raise NullPointerException

    def dynamicEDT_getDistance(self, p):
        if self.edtptr:
            if isinstance(p, OcTreeKey):
                return self.edtptr.getDistance(edt.OcTreeKey(<unsigned short int>p[0],
                                                             <unsigned short int>p[1],
                                                             <unsigned short int>p[2]))
            else:
                return self.edtptr.getDistance(edt.point3d(<float?>p[0],
                                                           <float?>p[1],
                                                           <float?>p[2]))
        else:
            raise NullPointerException


cdef class OcTree(CollisionGeometry):
    cdef octomap.OcTree* tree

    def __cinit__(self, octree):

        data = octree.write()
        cdef octomap.istringstream iss
        cdef ExOcTree tree = ExOcTree(0.1)
        if data.startswith(b"# Octomap OcTree file"):
            iss.str(string(<char*?>data, len(data)))
            del tree.thistree
            self.tree = <octomap.OcTree*>tree.thistree.read(<octomap.istream&?>iss)

        else:
            del tree.thistree
            self.tree = <octomap.OcTree*>tree.thistree.read(string(<char*?>data))

        self.thisptr = new defs.OcTree(defs.shared_ptr[octomap.OcTree](self.tree))

###############################################################################
# Collision managers
###############################################################################

cdef class DynamicAABBTreeCollisionManager:
    cdef defs.DynamicAABBTreeCollisionManager *thisptr
    cdef list objs

    def __cinit__(self):
        self.thisptr = new defs.DynamicAABBTreeCollisionManager()
        self.objs = []

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def registerObjects(self, other_objs):
        cdef vector[defs.CollisionObject*] pobjs
        for obj in other_objs:
            self.objs.append(obj)
            pobjs.push_back((<CollisionObject?> obj).thisptr)
        self.thisptr.registerObjects(pobjs)

    def registerObject(self, obj):
        self.objs.append(obj)
        self.thisptr.registerObject((<CollisionObject?> obj).thisptr)

    def unregisterObject(self, obj):
        if obj in self.objs:
            self.objs.remove(obj)
            self.thisptr.unregisterObject((<CollisionObject?> obj).thisptr)

    def setup(self):
        self.thisptr.setup()

    def update(self, arg=None):
        cdef vector[defs.CollisionObject*] objs
        if hasattr(arg, "__len__"):
            for a in arg:
                objs.push_back((<CollisionObject?> a).thisptr)
            self.thisptr.update(objs)
        elif arg is None:
            self.thisptr.update()
        else:
            self.thisptr.update((<CollisionObject?> arg).thisptr)

    def getObjects(self):
        return list(self.objs)

    def collide(self, *args):
        if len(args) == 2 and inspect.isroutine(args[1]):
            fn = CollisionFunction(args[1], args[0])
            self.thisptr.collide(<void*> fn, CollisionCallBack)
        elif len(args) == 3 and isinstance(args[0], DynamicAABBTreeCollisionManager):
            fn = CollisionFunction(args[2], args[1])
            self.thisptr.collide((<DynamicAABBTreeCollisionManager?> args[0]).thisptr, <void*> fn, CollisionCallBack)
        elif len(args) == 3 and inspect.isroutine(args[2]):
            fn = CollisionFunction(args[2], args[1])
            self.thisptr.collide((<CollisionObject?> args[0]).thisptr, <void*> fn, CollisionCallBack)
        else:
            raise ValueError

    def distance(self, *args):
        if len(args) == 2 and inspect.isroutine(args[1]):
            fn = DistanceFunction(args[1], args[0])
            self.thisptr.distance(<void*> fn, DistanceCallBack)
        elif len(args) == 3 and isinstance(args[0], DynamicAABBTreeCollisionManager):
            fn = DistanceFunction(args[2], args[1])
            self.thisptr.distance((<DynamicAABBTreeCollisionManager?> args[0]).thisptr, <void*> fn, DistanceCallBack)
        elif len(args) == 3 and inspect.isroutine(args[2]):
            fn = DistanceFunction(args[2], args[1])
            self.thisptr.distance((<CollisionObject?> args[0]).thisptr, <void*> fn, DistanceCallBack)
        else:
            raise ValueError

    def clear(self):
        self.thisptr.clear()

    def empty(self):
        return self.thisptr.empty()

    def size(self):
        return self.thisptr.size()

    property max_tree_nonbalanced_level:
        def __get__(self):
            return self.thisptr.max_tree_nonbalanced_level
        def __set__(self, value):
            self.thisptr.max_tree_nonbalanced_level = <int?> value

    property tree_incremental_balance_pass:
        def __get__(self):
            return self.thisptr.tree_incremental_balance_pass
        def __set__(self, value):
            self.thisptr.tree_incremental_balance_pass = <int?> value

    property tree_topdown_balance_threshold:
        def __get__(self):
            return self.thisptr.tree_topdown_balance_threshold
        def __set__(self, value):
            self.thisptr.tree_topdown_balance_threshold = <int?> value

    property tree_topdown_level:
        def __get__(self):
            return self.thisptr.tree_topdown_level
        def __set__(self, value):
            self.thisptr.tree_topdown_level = <int?> value

    property tree_init_level:
        def __get__(self):
            return self.thisptr.tree_init_level
        def __set__(self, value):
            self.thisptr.tree_init_level = <int?> value

    property octree_as_geometry_collide:
        def __get__(self):
            return self.thisptr.octree_as_geometry_collide
        def __set__(self, value):
            self.thisptr.octree_as_geometry_collide = <cppbool?> value

    property octree_as_geometry_distance:
        def __get__(self):
            return self.thisptr.octree_as_geometry_distance
        def __set__(self, value):
            self.thisptr.octree_as_geometry_distance = <cppbool?> value

###############################################################################
# Collision and distance functions
###############################################################################

def collide(CollisionObject o1, CollisionObject o2,
            request=None, result=None):

    if request is None:
        request = CollisionRequest()
    if result is None:
        result = CollisionResult()

    cdef defs.CollisionResult cresult

    cdef size_t ret = defs.collide(o1.thisptr, o2.thisptr,
                                   defs.CollisionRequest(
                                       <size_t?> request.num_max_contacts,
                                       <cppbool?> request.enable_contact,
                                       <size_t?> request.num_max_cost_sources,
                                       <cppbool> request.enable_cost,
                                       <cppbool> request.use_approximate_cost,
                                       <defs.GJKSolverType?> request.gjk_solver_type
                                   ),
                                   cresult)

    result.is_collision = result.is_collision or cresult.isCollision()

    cdef vector[defs.Contact] contacts
    cresult.getContacts(contacts)
    for idx in range(contacts.size()):
        result.contacts.append(c_to_python_contact(contacts[idx], o1, o2))

    cdef vector[defs.CostSource] costs
    cresult.getCostSources(costs)
    for idx in range(costs.size()):
        result.cost_sources.append(c_to_python_costsource(costs[idx]))

    return ret

def continuousCollide(CollisionObject o1, Transform tf1_end,
                      CollisionObject o2, Transform tf2_end,
                      request = None, result = None):

    if request is None:
        request = ContinuousCollisionRequest()
    if result is None:
        result = ContinuousCollisionResult()

    cdef defs.ContinuousCollisionResult cresult

    cdef defs.FCL_REAL ret = defs.continuousCollide(o1.thisptr, deref(tf1_end.thisptr),
                                                    o2.thisptr, deref(tf2_end.thisptr),
                                                    defs.ContinuousCollisionRequest(
                                                        <size_t?>             request.num_max_iterations,
                                                        <defs.FCL_REAL?>      request.toc_err,
                                                        <defs.CCDMotionType?> request.ccd_motion_type,
                                                        <defs.GJKSolverType?> request.gjk_solver_type,
                                                        <defs.CCDSolverType?> request.ccd_solver_type,

                                                    ),
                                                    cresult)

    result.is_collide = result.is_collide or cresult.is_collide
    result.time_of_contact = min(cresult.time_of_contact, result.time_of_contact)
    return ret

def distance(CollisionObject o1, CollisionObject o2,
             request = None, result=None):

    if request is None:
        request = DistanceRequest()
    if result is None:
        result = DistanceResult()

    cdef defs.DistanceResult cresult

    cdef double dis = defs.distance(o1.thisptr, o2.thisptr,
                                    defs.DistanceRequest(
                                        <cppbool?> request.enable_nearest_points,
                                        <defs.GJKSolverType?> request.gjk_solver_type
                                    ),
                                    cresult)

    result.min_distance = min(cresult.min_distance, result.min_distance)
    result.nearest_points = [vec3f_to_numpy(cresult.nearest_points[0]),
                             vec3f_to_numpy(cresult.nearest_points[1])]
    result.o1 = c_to_python_collision_geometry(cresult.o1, o1, o2)
    result.o2 = c_to_python_collision_geometry(cresult.o2, o1, o2)
    result.b1 = cresult.b1
    result.b2 = cresult.b2
    return dis

###############################################################################
# Collision and Distance Callback Functions
###############################################################################

def defaultCollisionCallback(CollisionObject o1, CollisionObject o2, cdata):
    request = cdata.request
    result = cdata.result

    if cdata.done:
        return True

    collide(o1, o2, request, result)

    if (not request.enable_cost and result.is_collision and len(result.contacts) > request.num_max_contacts):
        cdata.done = True

    return cdata.done

def defaultDistanceCallback(CollisionObject o1, CollisionObject o2, cdata):
    request = cdata.request
    result = cdata.result

    if cdata.done:
        return True, result.min_distance

    distance(o1, o2, request, result)

    dist = result.min_distance

    if dist <= 0:
        return True, dist

    return cdata.done, dist

cdef class CollisionFunction:
    cdef:
        object py_func
        object py_args

    def __init__(self, py_func, py_args):
        self.py_func = py_func
        self.py_args = py_args

    cdef cppbool eval_func(self, defs.CollisionObject*o1, defs.CollisionObject*o2):
        cdef object py_r = defs.PyObject_CallObject(self.py_func,
                                                    (copy_ptr_collision_object(o1),
                                                     copy_ptr_collision_object(o2),
                                                     self.py_args))
        return <cppbool?> py_r

cdef class DistanceFunction:
    cdef:
        object py_func
        object py_args

    def __init__(self, py_func, py_args):
        self.py_func = py_func
        self.py_args = py_args

    cdef cppbool eval_func(self, defs.CollisionObject*o1, defs.CollisionObject*o2, defs.FCL_REAL& dist):
        cdef object py_r = defs.PyObject_CallObject(self.py_func,
                                                    (copy_ptr_collision_object(o1),
                                                     copy_ptr_collision_object(o2),
                                                     self.py_args))
        (&dist)[0] = <defs.FCL_REAL?> py_r[1]
        return <cppbool?> py_r[0]

cdef inline cppbool CollisionCallBack(defs.CollisionObject*o1, defs.CollisionObject*o2, void*cdata):
    return (<CollisionFunction> cdata).eval_func(o1, o2)

cdef inline cppbool DistanceCallBack(defs.CollisionObject*o1, defs.CollisionObject*o2, void*cdata, defs.FCL_REAL& dist):
    return (<DistanceFunction> cdata).eval_func(o1, o2, dist)


###############################################################################
# Helper Functions
###############################################################################

cdef quaternion3f_to_numpy(defs.Quaternion3f q):
    return np.array([q.getW(), q.getX(), q.getY(), q.getZ()])

cdef defs.Quaternion3f numpy_to_quaternion3f(a):
    return defs.Quaternion3f(<double?> a[0], <double?> a[1], <double?> a[2], <double?> a[3])

cdef vec3f_to_numpy(defs.Vec3f vec):
    return np.array([vec[0], vec[1], vec[2]])

cdef defs.Vec3f numpy_to_vec3f(a):
    return defs.Vec3f(<double?> a[0], <double?> a[1], <double?> a[2])

cdef mat3f_to_numpy(defs.Matrix3f m):
    return np.array([[m(0,0), m(0,1), m(0,2)],
                        [m(1,0), m(1,1), m(1,2)],
                        [m(2,0), m(2,1), m(2,2)]])

cdef defs.Matrix3f numpy_to_mat3f(a):
    return defs.Matrix3f(<double?> a[0][0], <double?> a[0][1], <double?> a[0][2],
                         <double?> a[1][0], <double?> a[1][1], <double?> a[1][2],
                         <double?> a[2][0], <double?> a[2][1], <double?> a[2][2])

cdef c_to_python_collision_geometry(defs.const_CollisionGeometry*geom, CollisionObject o1, CollisionObject o2):
    cdef CollisionGeometry o1_py_geom = <CollisionGeometry> ((<defs.CollisionObject*> o1.thisptr).getUserData())
    cdef CollisionGeometry o2_py_geom = <CollisionGeometry> ((<defs.CollisionObject*> o2.thisptr).getUserData())
    if geom == <defs.const_CollisionGeometry*> o1_py_geom.thisptr:
        return o1_py_geom
    else:
        return o2_py_geom

cdef c_to_python_contact(defs.Contact contact, CollisionObject o1, CollisionObject o2):
    c = Contact()
    c.o1 = c_to_python_collision_geometry(contact.o1, o1, o2)
    c.o2 = c_to_python_collision_geometry(contact.o2, o1, o2)
    c.b1 = contact.b1
    c.b2 = contact.b2
    c.normal = vec3f_to_numpy(contact.normal)
    c.pos = vec3f_to_numpy(contact.pos)
    c.penetration_depth = contact.penetration_depth
    return c

cdef c_to_python_costsource(defs.CostSource cost_source):
    c = CostSource()
    c.aabb_min = vec3f_to_numpy(cost_source.aabb_min)
    c.aabb_max = vec3f_to_numpy(cost_source.aabb_max)
    c.cost_density = cost_source.cost_density
    c.total_cost = cost_source.total_cost
    return c

cdef copy_ptr_collision_object(defs.CollisionObject*cobj):
    geom = <CollisionGeometry> cobj.getUserData()
    co = CollisionObject(geom, _no_instance=True)
    (<CollisionObject> co).thisptr = cobj
    return co

