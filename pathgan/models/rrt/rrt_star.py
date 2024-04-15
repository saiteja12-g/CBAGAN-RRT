from typing import Tuple, List, Optional, Iterator

import time
import numpy as np
from copy import deepcopy
from PIL import Image
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt

from .rrt_base import RRTBase


class RRTStar(RRTBase):
    """RRT Star.

    Parameters
    ----------
    grid_map: np.ndarray
        Grid map.
    xy_init: Tuple[int, int]
        Start point coordinates.
    xy_goal: Tuple[int, int]
        Goal point coordinates.
    roi: List[tuple], optional (default=None)
        RoI for grid map.
    path_resolution: int, optional (default=1)
        Resolution of path.
    step_len: float, optional (default=0.5)
        Length of step to be made.
    max_iter: int, optional (default=10000)
        Number of maximum iterations.
    mu: float, optional (default=0.1)
        Mu value.
    seed: int, optional (default=42)
        Seed value.
    dist_init_goal: float, optional (default=None)
        Probability of selecting goal point.
    """
    def __init__(
        self,
        grid_map: np.ndarray,
        xy_init: Tuple[int, int],
        xy_goal: Tuple[int, int],
        roi: Optional[List[Tuple[int, int]]] = None,
        path_resolution: int = 1,
        step_len: float = 0.5,
        max_iter: int = 10000,
        gamma: float = 1,
        mu: float = 0.1,
        seed: int = 42,
        dist_init_goal: Optional[float] = None,
    ):
        """Initialize."""
        params = {name: value for name, value in locals().items() \
                  if name in RRTBase.__init__.__code__.co_varnames and name != 'self'}
        super().__init__(**params)

    def compute_search_radius(self) -> float:
        """Compute radius."""
        n = self.nodes_taken
        r = self.gamma * np.sqrt(np.log(n) / n)
        return r

    def get_near_costs(
        self,
        xy_near: Tuple[int, int],
        xy_new: Tuple[int, int],
        near_nodes: Iterator,
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Get costs of near points.

        Parameters
        ----------
        xy_near: Tuple[int, int]
            Near point coordinates.
        xy_new: Tuple[int, int]
            New point coordinates.
        near_nodes: Iterator
            Near nodes.
        """
        search_radius = self.compute_search_radius()
        costs_near = []
        for xy_near_ in near_nodes:
            if self.euclidean_distance(xy_near_, xy_new)[1] > search_radius:
                break
            costs_near.append((self.compute_cost(self.xy_init, xy_near_)[0] + \
                               self.euclidean_distance(xy_near_, xy_new)[1], xy_near_))

        if len(costs_near) == 0:
            costs_near = [(self.compute_cost(self.xy_init, xy_near)[0] + \
                           self.euclidean_distance(xy_near, xy_new)[1], xy_near)]
        costs_near.sort(key=itemgetter(0))

        return costs_near

    def choose_parent(
        self,
        xy_near: Tuple[int, int],
        xy_new: Tuple[int, int],
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Choose parent point.

        Parameters
        ----------
        xy_near: Tuple[int, int]
            Near point coordinates.
        xy_new: Tuple[int, int]
            New point coordinates.
        """
        near_nodes = iter(self.nearest(xy_new, self.nodes_taken))
        costs_near = self.get_near_costs(xy_near, xy_new, near_nodes)

        for cost, xy_near_ in costs_near:
            path_cost = cost + self.euclidean_distance(xy_new, self.xy_goal)[1]
            if path_cost < self.best_path.cost and self.is_connectable(xy_near_, xy_new):
                break
        return costs_near

    def rewire(
        self,
        xy_new: Tuple[int, int],
        costs_near: List[Tuple[float, Tuple[int, int]]],
    ):
        """Rewire.

        Parameters
        ----------
        xy_new: Tuple[int, int]
            New point coordinates.
        costs_near: List[Tuple[float, Tuple[int, int]]]
            List of near points with cost and coordinates.
        """
        for cost, xy_near in costs_near:
            cur_cost = self.compute_cost(self.xy_init, xy_near)[0]
            new_cost = self.compute_cost(self.xy_init, xy_new)[0] + \
                       self.euclidean_distance(xy_new, xy_near)[1]
            if new_cost < cur_cost and self.is_obstacle_free(xy_new, xy_near):
                self.add_edge(xy_new, xy_near)

    def run(self, max_iter: Optional[int] = None):
        """Run RRT algorithm on grid map.

        Parameters
        ----------
        max_iter: int, optional (default=None)
            Number of maximum iterations.
        """
        if max_iter is not None:
            self.max_iter = max_iter

        start_time = time.time()
        self.add_node(self.xy_init)
        self.add_edge(None, self.xy_init)

        for t in range(self.max_iter):
            if (t + 1) % 10 == 0:
                self.samples_taken_history.append(self.samples_taken)
                self.nodes_taken_history.append(self.nodes_taken)

            xy_rand = self.sample_free()
            xy_near = next(self.nearest(xy_rand))
            xy_new = self.steer(xy_near, xy_rand, self.step_len)

            if not self.V.count(xy_new) == 0 or not self.is_cell_free(xy_new):
                continue

            self.samples_taken += 1
            costs_near = self.choose_parent(xy_near, xy_new)
            if xy_new in self.E:
                self.rewire(xy_new, costs_near)

            path, cost = self.get_path_and_cost()
            if path and cost and cost < self.best_path.cost:
                end_time = time.time()
                self.time_elapsed = (end_time - start_time) / 10**3

                if not self.first_found_flag:
                    self.first_found_flag = True
                    self.first_path.update(path, cost,
                                           self.time_elapsed, t + 1,
                                           self.samples_taken, self.nodes_taken)
                    self.best_path = deepcopy(self.first_path)
                else:
                    self.best_path.update(path, cost,
                                          self.time_elapsed, t + 1,
                                          self.samples_taken, self.nodes_taken)
                self.costs_history.append(cost)
                self.path_lengths_history.append(len(path))

        return self.first_found_flag