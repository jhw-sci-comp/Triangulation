from builtins import print
import math
import numpy as np
import tkinter as tk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')


def calculate2Norm(vector_arg):
    sum_temp = 0.0
    for x in vector_arg:
        sum_temp += pow(x, 2)
    return math.sqrt(sum_temp)


class Triangle:
    def __init__(self, v1_arg, v2_arg, v3_arg):
        self.__vertices = [v1_arg, v2_arg, v3_arg]
        self.__edges =[]
        self.createEdges()
        self.__angles = []
        self.calculateAngles()
        self.__quality = 1/(2.0 * math.sin(min(self.__angles)))

    def getVertices(self):
        return self.__vertices

    def getQuality(self):
        return self.__quality

    def createEdges(self):
        if self.isCW():
            self.__vertices.reverse()

        for i in range(0, len(self.__vertices)):
            if i == len(self.__vertices) - 1:
                self.__edges.append(Edge(self.__vertices[i], self.__vertices[0]))
            else:
                self.__edges.append(Edge(self.__vertices[i], self.__vertices[i + 1]))

    def getEdges(self):
        return self.__edges

    def calculateAngles(self):
        for i in range(0, len(self.__vertices)):
            if i == 0:
                vec_1 = np.array([self.__vertices[i + 1].getX() - self.__vertices[i].getX(), self.__vertices[i + 1].getY() - self.__vertices[i].getY()])
                vec_2 = np.array([self.__vertices[len(self.__vertices) - 1].getX() - self.__vertices[i].getX(), self.__vertices[len(self.__vertices) - 1].getY() - self.__vertices[i].getY()])
                self.__angles.append(math.asin(abs(np.linalg.det([vec_1, vec_2])) / (calculate2Norm(vec_1) * calculate2Norm(vec_2))))
            elif i == len(self.__vertices) - 1:
                vec_1 = np.array([self.__vertices[0].getX() - self.__vertices[i].getX(), self.__vertices[0].getY() - self.__vertices[i].getY()])
                vec_2 = np.array([self.__vertices[i - 1].getX() - self.__vertices[i].getX(), self.__vertices[i - 1].getY() - self.__vertices[i].getY()])
                self.__angles.append(math.asin(abs(np.linalg.det([vec_1, vec_2])) / (calculate2Norm(vec_1) * calculate2Norm(vec_2))))
            else:
                vec_1 = np.array([self.__vertices[i + 1].getX() - self.__vertices[i].getX(), self.__vertices[i + 1].getY() - self.__vertices[i].getY()])
                vec_2 = np.array([self.__vertices[i - 1].getX() - self.__vertices[i].getX(), self.__vertices[i - 1].getY() - self.__vertices[i].getY()])

                self.__angles.append(math.asin(round(abs(np.linalg.det([vec_1, vec_2])) / (calculate2Norm(vec_1) * calculate2Norm(vec_2)), 12)))

    def getAngles(self):
        return self.__angles

    def isCW(self):
        v_det_1 = np.empty(3)
        v_det_2 = np.empty(3)
        v_det_3 = np.empty(3)
        sum_det = 0.0
        is_cw = False

        for i in range(0, len(self.__vertices)):
            if i == 0:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__vertices[len(self.__vertices) - 1].getX()
                v_det_1[2] = self.__vertices[len(self.__vertices) - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__vertices[i].getX()
                v_det_2[2] = self.__vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__vertices[i + 1].getX()
                v_det_3[2] = self.__vertices[i + 1].getY()

                sum_det += np.linalg.det(np.array([v_det_1, v_det_2, v_det_3]))
            elif i == len(self.__vertices) - 1:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__vertices[i - 1].getX()
                v_det_1[2] = self.__vertices[i - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__vertices[i].getX()
                v_det_2[2] = self.__vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__vertices[0].getX()
                v_det_3[2] = self.__vertices[0].getY()

                sum_det += np.linalg.det([v_det_1, v_det_2, v_det_3])
            else:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__vertices[i - 1].getX()
                v_det_1[2] = self.__vertices[i - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__vertices[i].getX()
                v_det_2[2] = self.__vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__vertices[i + 1].getX()
                v_det_3[2] = self.__vertices[i + 1].getY()

                sum_det += np.linalg.det([v_det_1, v_det_2, v_det_3])

        if sum_det < 0.0:
            is_cw = True

        return is_cw

    def __eq__(self, tri_arg):
        if set(self.__vertices) == set(tri_arg.getVertices()):
            return True
        else:
            return False


class Vertex:
    def __init__(self, x_arg, y_arg):
        self.__x = x_arg
        self.__y = y_arg

    def getX(self):
        return self.__x

    def setX(self, x_arg):
        self.__x = x_arg

    def getY(self):
        return self.__y

    def setY(self, y_arg):
        self.__y = y_arg

    def __add__(self, v_arg):
        return Vertex(self.__x + v_arg.getX(), self.__y + v_arg.getY())

    def __sub__(self, v_arg):
        return Vertex(self.__x - v_arg.getX(), self.__y - v_arg.getY())

    def __eq__(self, v_arg):
        if self.__x == v_arg.getX() and self.__y == v_arg.getY():
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.__x, self.__y))


class Edge:
    def __init__(self, v1_arg, v2_arg):
        self.__v_start = v1_arg
        self.__v_end = v2_arg
        vector_temp = np.array([-(v2_arg.getY() - v1_arg.getY()), (v2_arg.getX() - v1_arg.getX())])
        self.__normal_vector = np.array([vector_temp[0]/calculate2Norm(vector_temp), vector_temp[1]/calculate2Norm(vector_temp)])

    def getVStart(self):
        return self.__v_start

    def setVStart(self, v_arg):
        self.__v_start = v_arg

    def getVEnd(self):
        return self.__v_end

    def setVEnd(self, v_arg):
        self.__v_end = v_arg

    def getNormalVector(self):
        return self.__normal_vector

    def distanceVertex(self, v_arg):
        p_temp = np.array([self.__v_start.getX(), self.__v_start.getY()])  # start vector of line
        distance00 = p_temp.dot(self.__normal_vector)                      # distance between p_temp and origin
        v_temp = np.array([v_arg.getX(), v_arg.getY()])
        return v_temp.dot(self.__normal_vector) - distance00

    def intersectsEdge(self, edge_arg):
        if (self.distanceVertex(edge_arg.getVStart()) > 0.0 and self.distanceVertex(edge_arg.getVEnd()) < 0.0 or self.distanceVertex(edge_arg.getVStart()) < 0.0 and self.distanceVertex(edge_arg.getVEnd()) > 0.0) and (edge_arg.distanceVertex(self.getVStart()) > 0.0 and edge_arg.distanceVertex(self.getVEnd()) < 0.0 or edge_arg.distanceVertex(self.getVStart()) < 0.0 and edge_arg.distanceVertex(self.getVEnd()) > 0.0):
            return True
        else:
            return False

    def intersectVertex(self, vertex_arg):
        x_min = min(self.getVStart().getX(), self.getVEnd().getX())
        y_min = min(self.getVStart().getY(), self.getVEnd().getY())
        x_max = max(self.getVStart().getX(), self.getVEnd().getX())
        y_max = max(self.getVStart().getY(), self.getVEnd().getY())

        if abs(self.distanceVertex(vertex_arg)) < np.finfo(float).eps and vertex_arg.getX() >= x_min and vertex_arg.getX() <= x_max and vertex_arg.getY() >= y_min and vertex_arg.getY() <= y_max:
            return True
        else:
            return False

    def __lt__(self, e_arg):
        length_self = calculate2Norm(np.array([self.__v_end.getX() - self.__v_start.getX(), self.__v_end.getY() - self.__v_start.getY()]))
        length_e_arg = calculate2Norm(np.array([e_arg.getVEnd().getX() - e_arg.getVStart().getX(), e_arg.getVEnd().getY() - e_arg.getVStart().getY()]))

        if length_self < length_e_arg:
            return True
        else:
            return False

    def __le__(self, e_arg):
        length_self = calculate2Norm(np.array([self.__v_end.getX() - self.__v_start.getX(), self.__v_end.getY() - self.__v_start.getY()]))
        length_e_arg = calculate2Norm(np.array([e_arg.getVEnd().getX() - e_arg.getVStart().getX(), e_arg.getVEnd().getY() - e_arg.getVStart().getY()]))

        if length_self <= length_e_arg:
            return True
        else:
            return False

    def __eq__(self, e_arg):
        if self.getVStart() == e_arg.getVStart() and self.getVEnd() == e_arg.getVEnd() or self.getVStart() == e_arg.getVEnd() and self.getVEnd() == e_arg.getVStart():
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.getVStart(), self.getVEnd()))


class Domain:
    def __init__(self, boundary_vertices_arg):
        self.__boundary_vertices = boundary_vertices_arg.copy()
        self.__boundary_edges = []
        self.createBoundaryEdges()

    def getBoundaryVertices(self):
        return self.__boundary_vertices

    def getBoundaryEdges(self):
        return self.__boundary_edges

    def createBoundaryEdges(self):
        if not self.isCW():
            self.__boundary_vertices.reverse()

        for i in range(0, len(self.__boundary_vertices)):
            if i == len(self.__boundary_vertices) - 1:
                self.__boundary_edges.append(Edge(self.__boundary_vertices[i], self.__boundary_vertices[0]))
            else:
                self.__boundary_edges.append(Edge(self.__boundary_vertices[i], self.__boundary_vertices[i+1]))

    def isCW(self):
        v_det_1 = np.empty(3)
        v_det_2 = np.empty(3)
        v_det_3 = np.empty(3)
        sum_det = 0.0
        is_cw = False

        for i in range(0, len(self.__boundary_vertices)):
            if i == 0:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__boundary_vertices[len(self.__boundary_vertices) - 1].getX()
                v_det_1[2] = self.__boundary_vertices[len(self.__boundary_vertices) - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__boundary_vertices[i].getX()
                v_det_2[2] = self.__boundary_vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__boundary_vertices[i + 1].getX()
                v_det_3[2] = self.__boundary_vertices[i + 1].getY()

                sum_det += np.linalg.det(np.array([v_det_1, v_det_2, v_det_3]))
            elif i == len(self.__boundary_vertices) - 1:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__boundary_vertices[i - 1].getX()
                v_det_1[2] = self.__boundary_vertices[i - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__boundary_vertices[i].getX()
                v_det_2[2] = self.__boundary_vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__boundary_vertices[0].getX()
                v_det_3[2] = self.__boundary_vertices[0].getY()

                sum_det += np.linalg.det([v_det_1, v_det_2, v_det_3])
            else:
                v_det_1[0] = 1.0
                v_det_1[1] = self.__boundary_vertices[i - 1].getX()
                v_det_1[2] = self.__boundary_vertices[i - 1].getY()

                v_det_2[0] = 1.0
                v_det_2[1] = self.__boundary_vertices[i].getX()
                v_det_2[2] = self.__boundary_vertices[i].getY()

                v_det_3[0] = 1.0
                v_det_3[1] = self.__boundary_vertices[i + 1].getX()
                v_det_3[2] = self.__boundary_vertices[i + 1].getY()

                sum_det += np.linalg.det([v_det_1, v_det_2, v_det_3])

        if sum_det < 0.0:
            is_cw = True

        return is_cw


class Triangulation:
    def __init__(self, domain_arg):
        self.__domain_vertices = domain_arg.getBoundaryVertices().copy()
        self.__tri_vertices = domain_arg.getBoundaryVertices().copy()
        self.__domain_edges = domain_arg.getBoundaryEdges().copy()
        self.__tri_edges = domain_arg.getBoundaryEdges().copy()
        self.__triangles = []
        self.__new_edges = []

    def getTriangles(self):
        return self.__triangles

    def getTriEdges(self):
        return self.__tri_edges

    def getDomainVertices(self):
        return self.__domain_vertices

    def getNewEdges(self):
        return self.__new_edges

    def createFan(self):
        for i in range(0, len(self.__domain_vertices)):
            if i == 0:
                v_prev = self.__domain_vertices[len(self.__domain_vertices) - 1]
                v_i = self.__domain_vertices[i]
                v_next = self.__domain_vertices[i + 1]

                boundary_index_list = [j for j in range(0, len(self.__domain_vertices)) if j > i + 1]
            elif i == len(self.__domain_vertices) - 1:
                v_prev = self.__domain_vertices[i - 1]
                v_i = self.__domain_vertices[i]
                v_next = self.__domain_vertices[0]

                boundary_index_list = []
            else:
                v_prev = self.__domain_vertices[i - 1]
                v_i = self.__domain_vertices[i]
                v_next = self.__domain_vertices[i + 1]

                boundary_index_list = [j for j in range(0, len(self.__domain_vertices)) if j > i + 1]

            vector_prev_i = np.array([v_prev.getX() - v_i.getX(), v_prev.getY() - v_i.getY()])
            vector_i_next = np.array([v_next.getX() - v_i.getX(), v_next.getY() - v_i.getY()])

            angle_1 = self.calculateCWAngle(vector_i_next, vector_prev_i)

            for k in boundary_index_list:
                v_k = self.__domain_vertices[k]
                vector_ik = np.array([v_k.getX() - v_i.getX(), v_k.getY() - v_i.getY()])
                angle_2 = self.calculateCWAngle(vector_i_next, vector_ik)

                if angle_1 > angle_2:
                    intersection_found = False
                    e_temp = Edge(v_i, v_k)
                    for e in self.__tri_edges:
                        intersection_found = intersection_found or e_temp.intersectsEdge(e)

                    vertices_index_list = [j for j in range(0, len(self.__domain_vertices)) if j != i and j != k]
                    for j in vertices_index_list:
                        intersection_found = intersection_found or e_temp.intersectVertex(self.__domain_vertices[j])

                    if not intersection_found:
                        self.__new_edges.append(e_temp)
                        self.__tri_edges.append(e_temp)

    def calculateCWAngle(self, vector1_arg, vector2_arg):
        det_val = np.linalg.det([vector1_arg, vector2_arg])
        inner_prod = vector1_arg.dot(vector2_arg)

        angle_cw = 0.0

        angle_inner = math.acos(round(inner_prod/(calculate2Norm(vector1_arg) * calculate2Norm(vector2_arg)), 12))

        if abs(inner_prod) < np.finfo(float).eps:
            if det_val > 0.0:
                angle_cw = 1.5 * math.pi
            else:
                angle_cw = math.pi / 2.0
        elif abs(det_val) < np.finfo(float).eps:
            angle_cw = math.pi
        elif det_val > 0.0:
            angle_cw = 2 * math.pi - angle_inner
        elif det_val < 0.0:
            angle_cw = angle_inner

        return angle_cw

    def createTrianulation(self):
        self.createFan()

        for e_new in self.__new_edges:
            v_start = e_new.getVStart()
            v_end = e_new.getVEnd()

            # find triangles with 2 domain edges and one new edge
            for e_domain in self.__domain_edges:
                index_e_domain = self.__domain_edges.index(e_domain)
                if index_e_domain == len(self.__domain_edges) - 1:
                    e_domain_next = self.__domain_edges[0]
                else:
                    e_domain_next = self.__domain_edges[index_e_domain + 1]

                if e_domain.getVStart() == v_start and e_domain_next.getVEnd() == v_end or e_domain.getVStart() == v_end and e_domain_next.getVEnd() == v_start:
                    triangle_temp = Triangle(e_domain.getVStart(), e_domain.getVEnd(), e_domain_next.getVEnd())
                    if triangle_temp.isCW():
                        triangle_temp.getVertices().reverse()
                    self.__triangles.append(triangle_temp)

            # find triangles with 3 new edges
            e_new_v_start_connected = [e for e in self.__new_edges if (e.getVStart() == e_new.getVStart() or e.getVEnd() == e_new.getVStart()) and e != e_new]
            e_new_v_end_connected = [e for e in self.__new_edges if (e.getVStart() == e_new.getVEnd() or e.getVEnd() == e_new.getVEnd()) and e != e_new]
            v_start_set = set([e.getVStart() if e.getVStart() != e_new.getVStart() else e.getVEnd() for e in e_new_v_start_connected])
            v_end_set = set([e.getVStart() if e.getVStart() != e_new.getVEnd() else e.getVEnd() for e in e_new_v_end_connected])
            vertex_set_intersection = v_start_set.intersection(v_end_set)

            if len(vertex_set_intersection) != 0:
                triangle_temp = Triangle(e_new.getVStart(), e_new.getVEnd(), vertex_set_intersection.pop())

                if triangle_temp.isCW():
                    triangle_temp.getVertices().reverse()

                if triangle_temp not in self.__triangles:
                    self.__triangles.append(triangle_temp)

        # find triangles with 1 domain edge and 2 new edges
        for e_domain in self.__domain_edges:
            v_start = e_domain.getVStart()
            v_end = e_domain.getVEnd()

            e_domain_v_start_connected = [e for e in self.__new_edges if (e.getVStart() == e_domain.getVStart() or e.getVEnd() == e_domain.getVStart()) and e != e_domain]
            e_domain_v_end_connected = [e for e in self.__new_edges if (e.getVStart() == e_domain.getVEnd() or e.getVEnd() == e_domain.getVEnd()) and e != e_domain]
            v_start_set = set([e.getVStart() if e.getVStart() != e_domain.getVStart() else e.getVEnd() for e in e_domain_v_start_connected])
            v_end_set = set([e.getVStart() if e.getVStart() != e_domain.getVEnd() else e.getVEnd() for e in e_domain_v_end_connected])
            vertex_set_intersection = v_start_set.intersection(v_end_set)

            if len(vertex_set_intersection) != 0:
                triangle_temp = Triangle(e_domain.getVStart(), e_domain.getVEnd(), vertex_set_intersection.pop())

                if triangle_temp.isCW():
                    triangle_temp.getVertices().reverse()

                if triangle_temp not in self.__triangles:
                    self.__triangles.append(triangle_temp)

        self.restructureTriangulation()
        #self.refineTriangulation(10)

    def restructureTriangulation(self):
        self.__triangles.sort(key=lambda tri_arg: tri_arg.getQuality(), reverse=True)
        index_tri = 0

        while index_tri < len(self.__triangles):
            e_max = max(self.__triangles[index_tri].getEdges())

            index_tri_shared_edge = self.shareEdge(self.__triangles[index_tri], e_max)
            if index_tri_shared_edge >= 0:
                v_tri = [v for v in self.__triangles[index_tri].getVertices() if v != e_max.getVStart() and v != e_max.getVEnd()].pop()
                v_tri_shared_edge = [v for v in self.__triangles[index_tri_shared_edge].getVertices() if v != e_max.getVStart() and v != e_max.getVEnd()].pop()

                # flip the shared edge if the node of a triangle that shares an edge is in the circumcircle of the triangle
                if self.isInCircumcircle(self.__triangles[index_tri], v_tri_shared_edge):
                    tri_index_temp = self.__triangles[index_tri]
                    tri_index_shared_edge_temp = self.__triangles[index_tri_shared_edge]
                    self.__triangles.remove(tri_index_temp)
                    self.__triangles.remove(tri_index_shared_edge_temp)
                    self.removeEdge(e_max)
                    tri_new_1 = Triangle(v_tri, e_max.getVStart(),v_tri_shared_edge)
                    tri_new_2 = Triangle(v_tri_shared_edge, e_max.getVEnd(), v_tri)
                    self.__triangles.append(tri_new_1)
                    self.__triangles.append(tri_new_2)
                    e_new = Edge(v_tri, v_tri_shared_edge)
                    self.__tri_edges.append(e_new)
                    self.__new_edges.append(e_new)
                    self.__triangles.sort(key=lambda tri_arg: tri_arg.getQuality(), reverse=True)

                    index_tri += -1

            index_tri += 1

    def refineTriangulation(self, n_tri):
        #triangles_temp = self.__triangles.copy()
        index_tri = 0

        while index_tri < len(self.__triangles):
            tri_temp = self.__triangles[index_tri]
            e_max = max(tri_temp.getEdges())
            e_remaining = [e for e in tri_temp.getEdges() if e != e_max]
            index_shared_e_max = self.shareEdge(tri_temp, e_max)

            self.removeEdge(e_max)
            v_mid = Vertex(0.5*(e_max.getVStart().getX() + e_max.getVEnd().getX()), 0.5*(e_max.getVStart().getY() + e_max.getVEnd().getY()))
            e_new_1 = Edge(e_max.getVStart(), v_mid)
            e_new_2 = Edge(v_mid, e_max.getVEnd())
            v_temp = [v for v in tri_temp.getEdges() if v != e_max.getVStart() and v != e_max.getVEnd()].pop()
            e_new_3 = Edge(v_mid, v_temp)
            self.__tri_edges.append(e_new_1)
            self.__new_edges.append(e_new_1)
            self.__tri_edges.append(e_new_2)
            self.__new_edges.append(e_new_2)
            self.__tri_edges.append(e_new_3)
            self.__new_edges.append(e_new_3)
            self.__triangles.remove(tri_temp)
            tri_new_1 = Triangle(e_max.getVStart(), v_mid, v_temp)
            tri_new_2 = Triangle(e_max.getVEnd(), v_mid, v_temp)

            for e in e_remaining:
                index = self.shareEdge(tri_temp, e)

                if index >= 0:
                    tri_shared_e = self.__triangles[index]
                    v_temp_2 = [v for v in tri_shared_e.getEdges() if v != e.getVStart() and v != e.getVEnd()].pop()
                    if self.__triangles[self.shareEdge(tri_shared_e, e)] == tri_new_1:
                        tri_qualities = tri_shared_e.getQuality() + tri_new_1.getQuality()
                        tri_flip_1 = Triangle(v_mid, v_temp_2, tri_temp)
                        #tri_flip_2 = Triangle(v_mid, )


            if index_shared_e_max >= 0:
                pass

            index_tri += 1





    def shareEdge(self, tri_arg,  e_arg):
        index_tri = -1

        for t in self.__triangles:
            if t != tri_arg and e_arg in t.getEdges():
                index_tri = self.__triangles.index(t)

        return index_tri

    def isInCircumcircle(self, tri_arg, v_arg):
        A = tri_arg.getVertices()[0]
        B = tri_arg.getVertices()[1]
        C = tri_arg.getVertices()[2]

        denominator = 2 * (A.getX() * (B.getY() - C.getY()) + B.getX() * (C.getY() - A.getY()) + C.getX() * (A.getY() - B.getY()))
        x_center = ((A.getX() ** 2 + A.getY() ** 2) * (B.getY() - C.getY()) + (B.getX() ** 2 + B.getY() ** 2) * (C.getY() - A.getY()) + (C.getX() ** 2 + C.getY() ** 2) * (A.getY() - B.getY())) / denominator
        y_center = ((A.getX() ** 2 + A.getY() ** 2) * (C.getX() - B.getX()) + (B.getX() ** 2 + B.getY() ** 2) * (A.getX() - C.getX()) + (C.getX() ** 2 + C.getY() ** 2) * (B.getX() - A.getX())) / denominator
        radius = calculate2Norm(np.array([x_center - A.getX(), y_center - A.getY()]))

        distCenterV = calculate2Norm(np.array([v_arg.getX() - x_center, v_arg.getY() - y_center]))

        if distCenterV <= radius:
            return True
        else:
            return False

    def removeEdge(self, e_arg):
        e_arg_reverse = Edge(e_arg.getVEnd(), e_arg.getVStart())

        if e_arg in self.__new_edges:
            self.__new_edges.remove(e_arg)
        elif e_arg_reverse in self.__new_edges:
            self.__new_edges.remove(e_arg_reverse)

        if e_arg in self.__tri_edges:
            self.__tri_edges.remove(e_arg)
        elif e_arg_reverse in self.__tri_edges:
            self.__tri_edges.remove(e_arg_reverse)




#-----Printing and Plotting------
def printVertex(v_arg):
    print("(" + str(v_arg.getX()) + ", " + str(v_arg.getY()) + ")")

def printEdge(e_arg):
    print("(" + str(e_arg.getVStart().getX()) + ", " + str(e_arg.getVStart().getY()) + ")" + " -> " +"(" + str(e_arg.getVEnd().getX()) + ", " + str(e_arg.getVEnd().getY()) + ")")

def printTriangle(t_arg):
    print("(" + str(t_arg.getVertices()[0].getX()) + ", " + str(t_arg.getVertices()[0].getY()) + ")" + ", " +"(" + str(t_arg.getVertices()[1].getX()) + ", " + str(t_arg.getVertices()[1].getY()) + ")" + ", " + "(" + str(t_arg.getVertices()[2].getX()) + ", " + str(t_arg.getVertices()[2].getY()) + "), quality = " + str(t_arg.getQuality()) + ", min angle: " + str(min(t_arg.getAngles())))

def plotDomain(plot_ax_arg, domain_arg):
    x = []
    y = []

    for edge in domain_arg.getBoundaryEdges():
        x.append(edge.getVStart().getX())
        y.append(edge.getVStart().getY())

        x.append(edge.getVEnd().getX())
        y.append(edge.getVEnd().getY())

    x.append(domain_arg.getBoundaryVertices()[0].getX())
    y.append(domain_arg.getBoundaryVertices()[0].getY())

    plot_ax_arg.plot(x, y)


def plotTriangulation(plot_ax_arg, tri_arg):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # for edge in tri_arg.getTriEdges():
    #     x1.append(edge.getVStart().getX())
    #     y1.append(edge.getVStart().getY())
    #
    #     x1.append(edge.getVEnd().getX())
    #     y1.append(edge.getVEnd().getY())
    #
    #     plot_ax_arg.plot(x1, y1, color='r', linewidth=1)
    #     x1.clear()
    #     y1.clear()

    for edge in tri_arg.getNewEdges():
        x1.append(edge.getVStart().getX())
        y1.append(edge.getVStart().getY())

        x1.append(edge.getVEnd().getX())
        y1.append(edge.getVEnd().getY())

        plot_ax_arg.plot(x1, y1, color='r', linewidth=1)
        x1.clear()
        y1.clear()

    x1.append(tri_arg.getDomainVertices()[0].getX())
    y1.append(tri_arg.getDomainVertices()[0].getY())

    for vertex in tri_arg.getDomainVertices():
        x2.append(vertex.getX())
        y2.append(vertex.getY())

    x2.append(tri_arg.getDomainVertices()[0].getX())
    y2.append(tri_arg.getDomainVertices()[0].getY())

    #plot_ax_arg.plot(x1, y1, color='r', linewidth=1)
    plot_ax_arg.plot(x2, y2, color='b', linewidth=1)

#---------------------------------------


if __name__ == "__main__":
    V_domain = [Vertex(-6.0, 7.0), Vertex(2.0, 9.0), Vertex(7.0, 5.0), Vertex(6.0, -5.0), Vertex(1.0, -6.0), Vertex(-3.0, -7.0), Vertex(-5.0, -7.0), Vertex(-5.0, -3.0), Vertex(-8.0, -2.0), Vertex(-5.0, 2.0), Vertex(-8.0, 6.0)]
    #V_domain = [Vertex(-1.0, 1.0), Vertex(-1.0, 0.0), Vertex(0.0, 0.0), Vertex(0.0, -1.0), Vertex(1.0, -1.0), Vertex(1.0, 1.0)]
    #V_domain = [Vertex(-4.0, 6.0), Vertex(2.0, 7.0), Vertex(5.0, 4.0), Vertex(2.0, -3.0), Vertex(4.0, -5.0), Vertex(7.0, -3.0), Vertex(7.0, -7.0), Vertex(-3.0, -7.0), Vertex(-7.0, -1.0)]
    #V_domain = [Vertex(1, 0), Vertex(0, 2), Vertex(0, 1), Vertex(-1, 1), Vertex(-1, 0)]
    #V_domain = [Vertex(-1.0, 0.0), Vertex(2.0, 0.0), Vertex(2.0, 1.0), Vertex(1.0, 1.0), Vertex(1.0, 0.5), Vertex(0.0, 1.0), Vertex(-1.0, 1.0)]

    domain = Domain(V_domain)

    tri = Triangulation(domain)

    tri.createTrianulation()

    print("Triangles:")
    for tr in tri.getTriangles():
        printTriangle(tr)

    main_window = tk.Tk("Triangulation")
    main_window.geometry("800x700")

    figure = Figure(figsize=(8, 7), dpi=100)
    figure_canvas = FigureCanvasTkAgg(figure=figure, master=main_window)

    plot_ax = figure.add_subplot()
    plot_ax.axis("on")
    plot_ax.grid(True)

    #plotDomain(plot_ax, domain)
    plotTriangulation(plot_ax, tri)

    figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    main_window.mainloop()



