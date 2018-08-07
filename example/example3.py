import numpy as np
import fcl


def print_collision_result(o1_name, o2_name, result):
    print 'Collision between {} and {}:'.format(o1_name, o2_name)
    print '-'*30
    print 'Collision?: {}'.format(result.is_collision)
    print 'Number of contacts: {}'.format(len(result.contacts))
    print ''

def print_continuous_collision_result(o1_name, o2_name, result):
    print 'Continuous collision between {} and {}:'.format(o1_name, o2_name)
    print '-'*30
    print 'Collision?: {}'.format(result.is_collide)
    print 'Time of collision: {}'.format(result.time_of_contact)
    print ''

def print_distance_result(o1_name, o2_name, result):
    print 'Distance between {} and {}:'.format(o1_name, o2_name)
    print '-'*30
    print 'Distance: {}'.format(result.min_distance)
    print 'Closest Points:'
    print result.nearest_points[0]
    print result.nearest_points[1]
    print ''



tree = fcl.fcl.ExOcTree(0.1)
tree.insertPointCloud(np.array([[1.0, 0.0 ,0.0],
                                       [0.0, 0.0, 1.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, 0.0, -1.0]]),
                          np.array([0.0, 1.0, 0.0]))
tree.updateInnerOccupancy()


fcl_tree = fcl.OcTree(tree)


verts = np.array([[1.0, 1.0, 1.0],
                  [2.0, 1.0, 1.0],
                  [1.0, 2.0, 1.0],
                  [1.0, 1.0, 2.0]])
tris  = np.array([[0,2,1],
                  [0,3,2],
                  [0,1,3],
                  [1,2,3]])


# Create mesh geometry
mesh = fcl.BVHModel()
mesh.beginModel(len(verts), len(tris))
mesh.addSubModel(verts, tris)
mesh.endModel()


req = fcl.CollisionRequest(enable_contact=True)
res = fcl.CollisionResult()


n_contacts = fcl.collide(fcl.CollisionObject(mesh, fcl.Transform(np.array([-1.0, 0.0, 0.0]))),
                         fcl.CollisionObject(fcl_tree, fcl.Transform(np.array([0.0, 0.0, 0.0]))),
                         req, res)
print_collision_result('Mesh', 'Tree', res)


req = fcl.DistanceRequest(enable_nearest_points=True)
res = fcl.DistanceResult()

dist = fcl.distance(fcl.CollisionObject(mesh, fcl.Transform()),
                    fcl.CollisionObject(fcl_tree, fcl.Transform(np.array([1.01,0.0,0.0]))),
                    req, res)
print_distance_result('Mesh', 'Tree', res)


# req = fcl.ContinuousCollisionRequest()
# res = fcl.ContinuousCollisionResult()

# dist = fcl.continuousCollide(fcl.CollisionObject(mesh, fcl.Transform()),
#                              fcl.Transform(np.array([5.0, 0.0, 0.0])),
#                              fcl.CollisionObject(fcl_tree, fcl.Transform(np.array([5.0,0.0,0.0]))),
#                              fcl.Transform(np.array([0.0, 0.0, 0.0])),
#                              req, res)
# print_continuous_collision_result('Mesh', 'Tree', res)
