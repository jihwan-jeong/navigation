from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Literal, Callable
import numpy as np
import pygame


class Element(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __repr__(self) -> str:
        ...


class Point(Element):
    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y
        self._pos = np.array([x, y])

    @property
    def pos(self) -> np.array:
        return self._pos

    def __add__(self, vector: "Vector") -> "Point":
        if not isinstance(vector, Vector):
            raise TypeError("Can only add a vector to a point.")
        return Point(*(self._pos + vector._pos))

    def __sub__(self, other: "Point") -> "Vector":
        if not isinstance(other, Point):
            raise TypeError(
                "Can only substract a point from a point to generate a vector."
            )
        return Vector(*(self._pos - other._pos))

    def __eq__(self, other: "Point") -> bool:
        return np.array_equal(self._pos, other._pos)
    
    def __hash__(self):
        return hash((self._x, self._y))

    def __repr__(self) -> str:
        return "Point at ({x:.2f}, {y:.2f})".format(x=self._x, y=self._y)


class Vector(Point):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
        self._length = np.linalg.norm(self._pos)

    @property
    def length(self) -> float:
        return self._length

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos + other._pos))
    
    def __radd__(self, other) -> "Vector":
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(*(self._pos - other._pos))

    def __neg__(self):
        return Vector(*(-self._pos))

    def __mul__(self, other: Union[float, "Vector"]) -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._pos * other._pos))
        elif isinstance(other, (int, float)):
            return Vector(*(self._pos * other))
        else:
            raise TypeError(f"Unsupported type {type(other)} for __mul__")

    def __rmul__(self, other: Union[float, "Vector"]) -> "Vector":
        return self * other
    
    def __truediv__(self, a: float) -> "Vector":
        assert isinstance(a, (float, int))
        return Vector(*(self._pos / a))

    def dot(self, other: "Vector") -> float:
        return np.dot(self._pos, other._pos)

    def cross(self, other: "Vector") -> float:
        return np.cross(self._pos, other._pos)

    def __repr__(self) -> str:
        return "Vector ({x:.2f}, {y:.2f})".format(x=self._x, y=self._y)


class Line(Element):
    def __init__(self, a: Point, b: Point) -> None:
        self._a = a
        self._b = b
        self._envelop = (
            Point(*np.minimum(a._pos, b._pos)),
            Point(*np.maximum(a._pos, b._pos)),
        )

    def is_repel(self, other: "Line") -> bool:
        if all(
            (
                self._envelop[0]._x <= other._envelop[1]._x,
                self._envelop[1]._x >= other._envelop[0]._x,
                self._envelop[0]._y <= other._envelop[1]._y,
                self._envelop[1]._y >= other._envelop[0]._y,
            )
        ):
            return False
        return True

    def is_straddle(self, other: "Line") -> bool:
        """
        The criterion is "<=", which implies that the ends of one line can
        be "on" the extension cord of the other. Hence, two lines will be
        considered as "straddle" even if they overlap one another.
        """
        ac = other._a - self._a
        ad = other._b - self._a
        ab = self._b - self._a
        if ac.cross(ab) * ad.cross(ab) <= 0:
            return True
        return False

    def get_overlap(self, other: "Line") -> "Line":
        """
        self: ab   other: cd
        Preserves the direction for self.
        """
        def __check_zero_length(line: "Line"):
            if line._a == line._b:
                return None
            return line
        
        if self.is_repel(other):
            return None

        ac = other._a - self._a
        ad = other._b - self._a
        ab = self._b - self._a
        if ac.cross(ab) == 0 and ad.cross(ab) == 0:
            c_on = self.point_on(other._a)
            d_on = self.point_on(other._b)
            if c_on and d_on:
                return other
            if (not c_on) and (not d_on):
                return self
            
            on = other._a if c_on else other._b
            out = other._a if d_on else other._b
            ao = out - self._a
            if ab.dot(ao) > 0:
                return __check_zero_length(Line(on, self._b))
            return __check_zero_length(Line(self._a, on))
        return None

    def is_cross(self, other: "Line") -> bool:
        if self.is_repel(other):
            return False

        if all((self.is_straddle(other), other.is_straddle(self))):
            return True
        return False

    def cross_point(self, other: "Line") -> Union[bool, Point]:
        """
        Finds the cross point if it exists, returns if the two lines
        overlap is the cross point does not exist.

        self: ab   other: cd

        Args:
            other (Line): _description_

        Returns:
            Union[bool, Point]: (Point) if the cross point exists
                                (True) if they overlap
                                (False) if they do not cross over at all
        """
        ca = self._a - other._a
        cb = self._b - other._a
        da = self._a - other._b
        db = self._b - other._b
        S_abc = ca.cross(cb)  # 2 times the area size
        S_abd = da.cross(db)
        S_cda = ca.cross(da)
        S_cdb = cb.cross(db)

        if S_abc == S_abd == 0:
            ab = self._b - self._a
            cd = other._b - other._a
            # On the extension cord
            if self.is_repel(other):
                # Away from each other
                return False
            elif any(
                (
                    self._a == other._a and ab.dot(cd) < 0,
                    self._a == other._b and ab.dot(cd) > 0,
                )
            ):
                return self._a
            elif any(
                (
                    self._b == other._a and ab.dot(cd) > 0,
                    self._b == other._b and ab.dot(cd) < 0,
                )
            ):
                return self._b
            else:
                return True
        elif (S_abc * S_abd > 0) or (S_cda * S_cdb > 0):
            return False
        else:
            t = S_cda / (S_abd - S_abc)
            delta = t * (self._b - self._a)
            return self._a + delta

    def point_on(self, point: Point) -> bool:
        ac = point - self._a
        bc = point - self._b
        if all(
            (
                ac.cross(bc) == 0,
                ac.dot(bc) <= 0,
            )
        ):
            return True
        return False

    def __repr__(self) -> str:
        return "Line ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})".format(
            x1=self._a._x, y1=self._a._y, x2=self._b._x, y2=self._b._y
        )


class DirectEdge(Line):
    def __init__(self, a: Point, v: Vector) -> None:
        super().__init__(a, a + v)
        self._v = v

    def get_overlap(self, other: Line) -> "DirectEdge":
        """
        This method SHOULD ONLY BE CALLED when you know that they are overlapping!
        """
        raise NotImplementedError



class Polygon(Element):
    IN = -1
    ON = 0
    OUT = 1

    def __init__(self, vertices: List[Point]) -> None:
        assert len(vertices) >= 3
        self._vertices = vertices
        self._edges = [
            Line(a, b) for a, b in zip(vertices, vertices[1:] + [vertices[0]])
        ]
        for e, e_n in zip(self._edges, self._edges[1:] + [self._edges[0]]):
            if (e._b - e._a).cross(e_n._b - e_n._a) == 0:
                raise ValueError(f"Edge {e} and {e_n} are parallel!")
        self._envelop = (
            Point(*np.minimum.reduce([p._pos for p in vertices])),
            Point(*np.maximum.reduce([p._pos for p in vertices])),
        )

    def point_relative_pos(self, point: Point) -> Literal[-1, 0, 1]:
        """
        The logic is ported from (in Chinese):
        https://blog.csdn.net/zsjzliziyang/article/details/108813349
        """
        in_flag = False
        for edge in self._edges:
            a, b = edge._a, edge._b
            if edge.point_on(point):
                return Polygon.ON

            if any(
                (
                    a._y < point._y and b._y >= point._y,
                    a._y >= point._y and b._y < point._y,
                )
            ):
                # The if above guarantees that a.y != b.y
                x_cross = a._x + (point._y - a._y) * ((b._x - a._x) / (b._y - a._y))
                if x_cross == point._x:
                    return Polygon.ON
                if x_cross > point._x:
                    in_flag = not in_flag

        return Polygon.IN if in_flag else Polygon.OUT

    def __repr__(self) -> str:
        points = [str(p) for p in self._vertices]
        return "Polygon defined by (ordered) vertices: " + ", ".join(points)
    

RelativePos = Literal[-1, 0, 1]


class Box(Polygon):
    """
    0  ------------>  x

    |   a -- width -- b
    |
    |   |             |
    |   |           height
    v   |             |

    y   c ----------- d
    """

    def __init__(self, vertex: Point, width: float, height: float) -> None:
        ab = Vector(width, 0)
        ac = Vector(0, height)
        super().__init__([vertex, vertex + ab, vertex + ab + ac, vertex + ac])

    def point_relative_pos(self, point: Point) -> RelativePos:
        if all((
            (point._pos >= self._envelop[0]._pos).all(),
            (point._pos <= self._envelop[1]._pos).all(),
        )):
            for edge in self._edges:
                if edge.point_on(point):
                    return Polygon.ON
            return Polygon.IN
        return Polygon.OUT
    

class Region(Element):
    def __init__(self, zone: Polygon) -> None:
        self._zone = zone

    def check_segment_cross(self, p0: Point, proposition: Vector) -> Tuple[List[Point], List[RelativePos]]:
        def __handle_corners(edge: Line, edge_next: Line):
            if edge._b == p0:
                return True
            ba = edge._a - edge._b
            bc = edge_next._b - edge_next._a
            if proposition.cross(ba) * proposition.cross(bc) > 0:
                return False
            return True
        
        anchor_pts: List[Point] = [p0, p0 + proposition]
        pp = Line(p0, p0 + proposition)
        for edge, edge_next in zip(self._zone._edges, self._zone._edges[1:] + [self._zone._edges[0]]):
            cross = pp.cross_point(edge)
            if cross is True:
                overlap = pp.get_overlap(edge)
                anchor_pts.extend([overlap._a, overlap._b])
            
            elif isinstance(cross, Point):
                # Vertex crosspoint will only be handled on the corner.
                if cross == edge._b:
                    if any((
                        cross == p0,
                        __handle_corners(edge, edge_next),
                    )):
                        anchor_pts.append(edge._b)
                elif cross != edge._a:
                    anchor_pts.append(cross)

        anchor_pts = list(set(anchor_pts))
        lengths = [(cross - p0).length for cross in anchor_pts]
        _, anchor_pts = zip(*sorted(zip(lengths, anchor_pts), key=lambda x: x[0]))
        seg2poly_relation = []
        for p1, p2 in zip(anchor_pts[:-1], anchor_pts[1:]):
            mid = p1 + (p2 - p1) / 2
            seg2poly_relation.append(self._zone.point_relative_pos(mid))

        return anchor_pts, seg2poly_relation
    
    @property
    @abstractmethod
    def should_apply(self):
        ...
        
    def first_contact(self, p0: Point, propositions: List[Vector]) -> Tuple[Point, float]:
        if isinstance(propositions, Vector):
            propositions = [propositions]
        
        if self._zone.point_relative_pos(p0) == Polygon.IN:
            return p0, -1

        p = p0
        cuml_dist = 0
        for prop in propositions:
            anchor_pts, rel_poss = self.check_segment_cross(p, prop)
            for anchor1, anchor2, rel_pos in zip(anchor_pts[:-1], anchor_pts[1:], rel_poss):
                if rel_pos in self.should_apply:
                    return anchor1, cuml_dist
                cuml_dist += (anchor2 - anchor1).length
            p += prop

        return None, np.inf
    
    def render(self, viewer, to_pixel: Callable, color: Tuple[int, int, int] = None):
        pygame.draw.polygon(
            viewer,
            self.COLOR if color is None else color,
            [to_pixel(v.pos) for v in self._zone._vertices],
            width=0,        # 0 for fill, > 0 for line thickness
        )

class DynamicRegion(Region):
    COLOR = [160, 160, 160]

    @abstractmethod
    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        ...
    
    @property
    def should_apply(self):
        return [Polygon.IN]


class SimpleDynamicRegion(DynamicRegion):
    def apply_dynamic(self, p0: Point, proposition: Vector):
        if not isinstance(proposition, Vector):
            raise TypeError("SimpleDynamicRegion does not support complex movements.")
        return self._apply_dynamic(p0, proposition)
    
    @abstractmethod
    def _apply_dynamic(self, p0: Point, proposition: Vector):
        ...


class PunchRegion(SimpleDynamicRegion):
    """
    An extra, constant motion will be applied as long as the agent ENTERS the region.
    """
    def __init__(self, zone: Polygon, force: Vector) -> None:
        super().__init__(zone)
        self._force = force

    def _apply_dynamic(self, p0: Point, proposition: Vector):
        _, rel_poss = self.check_segment_cross(p0, proposition)
        if Polygon.IN in rel_poss:
            return proposition + self._force
        return proposition

    def __repr__(self) -> str:
        return f"Punch zone with a fixed force '{self._force._pos}'. " + str(self._zone)
    

class NoEntryRegion(DynamicRegion):
    """ The agent CAN walk alongside the walls. """
    COLOR = [0, 0, 0]

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        for prop in propositions:
            anchor_pts, rel_poss = self.check_segment_cross(p, prop)
            for anchor, rel_pos in zip(anchor_pts[:-1], rel_poss):
                if rel_pos in self.should_apply:
                    movements.append(anchor - p)
                    return movements
            movements.append(prop)
            p += prop
        return movements
    
    def __repr__(self) -> str:
        return "No entry zone. " + str(self._zone)
    

class SlipperyRegion(DynamicRegion):
    def __init__(self, zone: Polygon, force: Vector) -> None:
        super().__init__(zone)
        self._force = force

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        total_length = sum([prop.length for prop in propositions])
        for prop in propositions:
            normal_length = 0
            anchor_pts, rel_poss = self.check_segment_cross(p, prop)
            movement = None
            for anchor1, anchor2, rel_pos in zip(anchor_pts[:-1], anchor_pts[1:], rel_poss):
                if rel_pos not in self.should_apply:
                    normal_length += (anchor2 - anchor1).length
                else:
                    delta_t = (prop.length - normal_length) / total_length
                    movement = [] if anchor1 == p else [anchor1 - p]
                    movement.append(p + prop - anchor1 + self._force * delta_t)
                    break
            movement = [prop] if movement is None else movement
            movements.extend(movement)
            for m in movement:
                p += m

        return movements
    
    def __repr__(self) -> str:
        return f"Slippery zone with a force rate '{self._force._pos}'. " + str(self._zone)
    

class BlackHoleRegion(DynamicRegion):
    def __init__(self, zone: Polygon, center: Point, force: float) -> None:
        super().__init__(zone)
        self._center = center
        self._force = force

    def apply_dynamic(self, p0: Point, propositions: List[Vector]):
        p = p0
        movements = []
        total_length = sum([prop.length for prop in propositions])
        for prop in propositions:
            normal_length = 0
            anchor_pts, rel_poss = self.check_segment_cross(p, prop)
            movement = None
            for anchor1, anchor2, rel_pos in zip(anchor_pts[:-1], anchor_pts[1:], rel_poss):
                if rel_pos not in self.should_apply:
                    normal_length += (anchor2 - anchor1).length
                else:
                    vector = self._center - anchor1
                    direction = vector / vector.length
                    delta_t = (prop.length - normal_length) / total_length
                    movement = [] if anchor1 == p else [anchor1 - p]
                    applied_force = (
                        vector
                        if vector.length < self._force * delta_t    # The final movement should not exceed the center.
                        else direction * self._force * delta_t
                    )
                    movement.append(p + prop - anchor1 + applied_force)
                    break
            movement = [prop] if movement is None else movement
            movements.extend(movement)
            for m in movement:
                p += m

        return movements

    def __repr__(self) -> str:
        return f"Black hold zone centered on {self._center} with a force rate {self._force}. " + str(self._zone)


class RewardRegion(Region):
    """ Reward WILL BE applied when the agent touches the boundary. """
    COLOR = [255, 102, 102]

    def __init__(self, zone: Polygon, reward: float) -> None:
        super().__init__(zone)
        self._reward = reward

    def apply_reward(self, p0: Point, propositions: List[Vector]):
        p = p0
        total_length = 0
        counted_length = 0
        for prop in propositions:
            total_length += prop.length
            anchor_pts, rel_poss = self.check_segment_cross(p, prop)
            for anchor1, anchor2, rel_pos in zip(anchor_pts[:-1], anchor_pts[1:], rel_poss):
                if rel_pos in self.should_apply:
                    counted_length += (anchor2 - anchor1).length
            p += prop

        if total_length == 0:
            rel_pos = self._zone.point_relative_pos(p0)
            if rel_pos in self.should_apply:
                return self._reward
            return 0
            
        return self._reward * counted_length / total_length
    
    @property
    def should_apply(self):
        return [Polygon.IN, Polygon.ON]

    def __repr__(self) -> str:
        return f"Reward zone with reward rate '{self._reward}'. " + str(self._zone)


class SimpleRewardRegion(RewardRegion):
    """ Penalizes the agent as long as it touches the region. """

    def apply_reward(self, p0: Point, propositions: List[Vector]):
        rel_pos = self._zone.point_relative_pos(p0 + sum(propositions))
        if rel_pos in self.should_apply:
            return self._reward
        return 0
    
    def __repr__(self) -> str:
        return f"Simple reward zone with constant reward '{self._reward}/step'. " + str(self._zone)

