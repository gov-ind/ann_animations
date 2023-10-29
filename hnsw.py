from __future__ import annotations

from manimlib import *
from manimlib import Dot
import numpy as np
import random

from bisect import insort

k = 3
RED = "#AD0202"
RED = "#AF0000"

class Node:
    def __init__(self, val, neighbours, index):
        self.val = val
        self.neighbours = neighbours
        self.index = index

def create_shaped_data(n=30):
    y = np.linspace(-3, 3, n)
    x = np.sin(y) + 3
    scale = 3

    data = np.stack((x, y)).T
    x_, y_ = data[-1]
    data += np.random.randn(*data.shape) / scale

    rest = np.linspace([x_, y_], [-4, 3], n)
    rest += np.random.randn(*rest.shape) / scale
    data = np.vstack((data, rest))

    rest = np.linspace([-4, 3], [-4, -1.5], n)
    rest += np.random.randn(*rest.shape) / scale
    data = np.vstack((data, rest))

    return data

class InteractiveDevelopment(Scene):
    def wait(self, time):
        if self.run_time <= .1:
            return
        return super().wait(time)
    
    def create_graph_exhaustive(self, nodes, k=k, animate=False, arrow=False):
        for ix, node in enumerate(nodes):
            if animate and ix < 2:
                anims = []
                dot = Dot(self.dot_mp[node.index].get_center(), radius=.2, color=BLACK)
                self.play(GrowFromCenter(dot), run_time=self.run_time)
                if ix == 0:
                    self.wait(1)
            neighbours = []
            for neighbour in nodes:
                if node == neighbour:
                    continue
                neighbours.append((np.linalg.norm(node.val - neighbour.val), neighbour))

            neighbours = [a[1] for a in sorted(neighbours, key=lambda a: a[0])][:k]
            for neighbour in neighbours:
                if node not in neighbour.neighbours:
                    neighbour.neighbours.append(node)
            
                if neighbour not in node.neighbours:
                    node.neighbours.append(neighbour)

                if animate and ix < 2:
                    anims.append(self.dot_mp[neighbour.index].animate.set_color(RED))
            
            if animate and ix < 2:
                self.play(AnimationGroup(*anims), run_time=self.run_time)
                self.wait(1)
                self.draw_edges([node], arrow=arrow)
                self.wait(2)

                anims = [Uncreate(dot)]
                anims += [self.dot_mp[a.index].animate.set_color(BLACK) for a in neighbours]
                self.play(AnimationGroup(*anims), run_time=self.run_time)

    def create_graph_random(self, nodes, k):
        for node in nodes:
            neighbours = [a for a in nodes if a != node and a not in node.neighbours]

            to_add = (k - len(node.neighbours))
            if to_add > 0:
                neighbours = np.random.choice(neighbours, to_add, replace=False)
                node.neighbours += neighbours.tolist()
            
            # TODO: Uncomment for reverse-neighbourhood
            #for neighbour in node.neighbours:
            #    if node not in neighbour.neighbours:
            #        neighbour.neighbours.append(node)

        return nodes

    def in_dots(self, dot, dots):
        for x in dots:
            if np.isclose(x, dot).all():
                return True
        return False

    def dot2ix(self, dot):
        for ix in self.dot_mp:
            if np.isclose(self.dot_mp[ix].get_center()[:2], dot).all():
                return ix

    def draw_edges(self, nodes, arrow=False):
        anims = []
        for node in nodes:
            ix = self.dot2ix(node.val)
            dot = self.dot_mp[ix].get_center()[:2]

            for neighbour in node.neighbours:
                end = Dot([neighbour.val[0], neighbour.val[1], 0], color=BLACK)
                ix_end = self.dot2ix(neighbour.val)
                end = self.dot_mp[ix_end]

                if ix_end not in self.edge_mp[ix]:
                    if arrow:
                        edge = Arrow(dot, end, color=BLACK)
                    else:
                        edge = Line(dot, end, color=BLACK)
                    self.edge_mp[ix][ix_end] = edge

                    anims.append(ShowCreation(edge))

        self.play(AnimationGroup(*anims), run_time=self.run_time)
                    
    def erase_edges(self):
        edges = []
        for ix in self.edge_mp:
            for ix2 in self.edge_mp[ix]:
                edge = self.edge_mp[ix][ix2]
                if edge not in edges:
                    edges.append(edge)

        self.play(AnimationGroup(*[Uncreate(a) for a in edges]), run_time=self.run_time)
        self.clear_edge_mp()

    def clear_edge_mp(self):
        for ix in self.edge_mp:
            self.edge_mp[ix] = {}
    
    def clear_node_neighbours(self, nodes):
        for node in nodes:
            node.neighbours = []
    
    def traverse(self, start, target, candid=None, animate_dots=True, animate_nns=True, ef=k, slow=False):
        visit = set()
        anims = []

        if candid is None:
            self.play(self.dot_mp[start.index].animate.set_color(RED), run_time=self.run_time)

            self.wait(1)

            dist = np.linalg.norm(start.val - target.val)
            candid = [(dist, start)]
            for neighbour in start.neighbours:
                insort(candid, (np.linalg.norm(neighbour.val - target.val), neighbour))
            
            for _, node in candid:
                anims.append(self.dot_mp[node.index].animate.set_color(RED))

        nns = list(candid)
        self.play(AnimationGroup(*anims), run_time=self.run_time)
        
        if slow:
            self.wait(2)
        
        iter = 0
        while candid:
            prev_nns = list(nns)
            dist, curr = candid[0]
            candid = candid[1:]

            if curr not in visit:
                visit.add(curr)
                neighbours = curr.neighbours

                if any([a not in visit for a in neighbours]):
                    if animate_dots and animate_nns:
                        curr_dot = Dot(self.dot_mp[curr.index].get_center(), radius=.2, color=BLACK)
                        self.play(GrowFromCenter(curr_dot), run_time=self.run_time)
                else:
                    continue
                
                if slow and iter == 0:
                    self.wait(1)

                neighbour_edges = []
                anims = []
                for node in neighbours:
                    dist = np.linalg.norm(node.val - target.val)

                    if node not in [a[1] for a in nns]:
                        if animate_dots and animate_nns:
                            neighbour_edge = Line(self.dot_mp[curr.index].get_center(), self.dot_mp[node.index].get_center(), color=RED)
                            anims.append(ShowCreation(neighbour_edge))
                            neighbour_edges.append(neighbour_edge)

                        if not candid or dist < candid[-1][0]:
                            insort(candid, (dist, node))
                            insort(nns, (dist, node))
                            while len(candid) > ef:
                                candid.pop()
                            while len(nns) > ef:
                                nns.pop()
                
                if animate_dots and animate_nns:
                    self.play(AnimationGroup(*anims), run_time=self.run_time)
                    
                    if slow and iter == 0:
                        self.wait(1)
                    self.wait(1)
                    
                    anims = [Uncreate(curr_dot)]
                    
                    for edge in neighbour_edges:
                        anims.append(Uncreate(edge))
                    self.play(AnimationGroup(*anims), run_time=self.run_time)
                        

            if animate_nns:
                prev_nn_nodes = [a[1] for a in prev_nns]
                nn_nodes = [a[1] for a in nns]
                
                anims = []
                for node in nn_nodes:
                    if node not in prev_nn_nodes:
                        anims.append(self.dot_mp[node.index].animate.set_color(RED))
                        
                if slow and iter == 0:
                    self.play(AnimationGroup(*anims), run_time=self.run_time)
                    anims = []
                    self.wait(2)
                        
                for node in prev_nn_nodes:
                    if node not in nn_nodes:
                        anims.append(self.dot_mp[node.index].animate.set_color(BLACK))
                
                self.play(AnimationGroup(*anims), run_time=self.run_time)
                if slow and iter == 0:
                    self.wait(2)
            
            iter += 1
        
        return nns
    
    def clear_world(self, clear_target=False):
        anims = []
        for ix in self.dot_mp:
            anims.append(Uncreate(self.dot_mp[ix]))
            if self.dot_mp[ix].color != BLACK:
                anims.append(self.dot_mp[ix].animate.set_color(BLACK))
        if clear_target and self.target is not None:
            anims.append(Uncreate(self.target_dot))
        self.play(AnimationGroup(*anims), run_time=self.run_time)
    
    def reset_world(self):
        anims = []
        for ix in self.dot_mp:
            anims.append(self.dot_mp[ix].animate.set_color(BLACK))
        self.play(AnimationGroup(*anims), run_time=self.run_time)

    def create_world(self, nums, n_worlds=1):
        self.nodes = []
        self.dot_mp = {}
        self.edge_mp = {}
        dots = []

        self.worlds = [[] for _ in range(n_worlds)]
        self.world_mps = [{} for _ in range(n_worlds)]

        anims = []
        for ix, num in enumerate(nums):
            node = Node(num, [], ix)
            self.nodes.append(node)

            for world_ix, _ in enumerate(self.worlds):
                if n_worlds > 1 and world_ix == 0:
                    if random.randint(0, 10) < 2:
                        node_ = Node(num, [], ix)
                        self.worlds[world_ix].append(node_)
                        self.world_mps[world_ix][ix] = node_
                else:
                    self.worlds[world_ix].append(node)
                    self.world_mps[world_ix][ix] = node

            dot = Dot([node.val[0], node.val[1], 0], color=BLACK)
            dots.append(dot)
            self.dot_mp[ix] = dot
            self.edge_mp[ix] = {}

            anims.append(GrowFromCenter(dot))

        self.play(AnimationGroup(*anims), run_time=self.run_time)

    def one_world_traverse(self, n=20, run_time=1, traverse_only=False):
        x_mn, x_mx, y_mn, y_mx = -5, 5, -3, 3
        np.random.seed(1336)

        nums = np.vstack((np.random.uniform(x_mn, x_mx, n), np.random.uniform(y_mn, y_mx, n))).T
        
        self.run_time = run_time
        self.create_world(nums, n_worlds=1)

        if self.target is None:
            ix = len(self.nodes)
            self.target_dot = Dot([-2, -1, 0], color=RED, radius=.2)
            self.target = Node([-2, -1], [], ix + 1)
            self.play(GrowFromCenter(self.target_dot), run_time=self.run_time)

        self.wait(1)
        self.create_graph_exhaustive(self.worlds[0], animate=False)
        self.draw_edges(self.nodes, arrow=not traverse_only)

        if not traverse_only:
            self.wait(2)
            self.erase_edges()
            self.clear_node_neighbours(self.nodes)
            self.create_graph_exhaustive(self.worlds[0], animate=True, arrow=False)
            self.draw_edges(self.nodes)

        self.wait(2)
        self.traverse(self.world_mps[0][0], self.target, candid=None, animate_dots=True, slow=not traverse_only)
        self.wait(1)
        self.erase_edges()
        self.wait(2)
        self.clear_world()
    
    def create_n_prune(self, nums, k, slow=True, arrow=True, animate=True, n_rounds=1, should_clear=True):
        self.create_world(nums, n_worlds=1)
        nodes = self.create_graph_random(self.worlds[0], k=k)
        self.draw_edges(self.worlds[0], arrow=arrow)
        self.wait(2)
        
        for _ in range(n_rounds):
            self.prune(nodes, k=k, slow=slow, arrow=arrow, animate=animate)
            self.wait(2)
            
        if should_clear:
            self.erase_edges()
            self.clear_world()
        self.wait(2)
            
    def prune(self, nodes, k, animate=True, slow=True, arrow=True, anim_interval=20):
        glob_anims = []
        
        for i, node in enumerate(nodes):
            edges_to_delete = []
            edges_to_add = []
            neighbours_to_delete = []
            neighbours_to_add = []

            ix = self.dot2ix(node.val)

            if animate:
                self.play(self.dot_mp[ix].animate.set_color(RED), run_time=self.run_time)
            
                ix_ends = [self.dot2ix(a.val) for a in node.neighbours]
                anims = []
                for ix_end in ix_ends:
                    anims.append(self.dot_mp[ix_end].animate.set_color(RED))
                self.play(AnimationGroup(*anims), run_time=self.run_time)

            if slow:
                self.wait(.2)
                
            new_neighbours = [(np.linalg.norm(a.val - node.val), a) for a in node.neighbours]
            visited = [node.val] + [a.val for a in node.neighbours]
            for neighbour in node.neighbours:
                for friend in neighbour.neighbours:
                    if self.in_dots(friend.val, visited):
                        continue
                    visited.append(friend.val)
                    new_neighbours.append((np.linalg.norm(friend.val - node.val), friend))
            new_neighbours = [a[1] for a in sorted(new_neighbours)][:k]

            for neighbour in node.neighbours:
                if neighbour not in new_neighbours:
                    if len(neighbour.neighbours) == 1:
                        new_neighbours = [neighbour] + new_neighbours
                        continue
                    ix_end = self.dot2ix(neighbour.val)

                    if ix_end in self.edge_mp[ix] or ix in self.edge_mp[ix_end]:
                        if ix_end in self.edge_mp[ix]:
                            edge = self.edge_mp[ix][ix_end]
                            del self.edge_mp[ix][ix_end]
                        else:
                            edge = self.edge_mp[ix_end][ix]
                            del self.edge_mp[ix_end][ix]
                    edges_to_delete.append(edge)
                    neighbours_to_delete.append(neighbour)
            
            new_neighbour_dots = []
            anims = []

            for neighbour in new_neighbours:
                if neighbour not in node.neighbours:
                    ix_end = self.dot2ix(neighbour.val)
                    dot = Dot(self.dot_mp[ix_end].get_center(), radius=.2, color=BLACK)
                    new_neighbour_dots.append(dot)
                    anims.append(GrowFromCenter(dot))

                    if arrow:
                        edge = Arrow(self.dot_mp[ix], self.dot_mp[ix_end], color=BLACK)
                    else:
                        edge = Line(self.dot_mp[ix], self.dot_mp[ix_end], color=BLACK)
                    
                    if ix_end in self.edge_mp[ix]:
                        edges_to_delete.append(self.edge_mp[ix][ix_end])
                    if ix in self.edge_mp[ix_end]:
                        edges_to_delete.append(self.edge_mp[ix_end][ix])
                    self.edge_mp[ix][ix_end] = edge
                    self.edge_mp[ix_end][ix] = edge
                    edges_to_add.append(edge)
                    neighbours_to_add.append(neighbour)
            
            if animate:
                if slow:
                    self.wait(1)
                    self.play(AnimationGroup(*anims), run_time=self.run_time)
                    self.wait(2)
                
                self.play(AnimationGroup(*[ShowCreation(edge) for edge in edges_to_add]), run_time=self.run_time)
                
                if slow:
                    self.wait(2)
                
                self.play(AnimationGroup(*[Uncreate(edge) for edge in edges_to_delete]), run_time=self.run_time)
            
                anims = []
                anims.append(self.dot_mp[ix].animate.set_color(BLACK))
                for ix_end in ix_ends:
                    anims.append(self.dot_mp[ix_end].animate.set_color(BLACK))
                for dot in new_neighbour_dots:
                    anims.append(Uncreate(dot))
                    
                self.play(AnimationGroup(*anims), run_time=self.run_time)
                    
            else:
                glob_anims += [ShowCreation(edge) for edge in edges_to_add]
                glob_anims += [Uncreate(edge) for edge in edges_to_delete]


            for neighbour in neighbours_to_delete:
                node.neighbours.remove(neighbour)
                # TODO: Uncomment for reverse-neighborhood
                # neighbour.neighbours.remove(node)
            for neighbour in neighbours_to_add:
                node.neighbours.append(neighbour)
                # TODO: Uncomment for reverse-neighborhood
                # neighbour.neighbours.append(node)
            
            
            if i % anim_interval == 0 and i != 0:
                self.play(AnimationGroup(*glob_anims), run_time=self.run_time)
                self.wait(1)
                glob_anims = []
                
        if len(glob_anims) > 0:
            self.play(AnimationGroup(*glob_anims), run_time=self.run_time)


    def construct(self):
        nns = None
        self.target = None

        self.one_world_traverse(n=20, run_time=.8)
        self.one_world_traverse(n=120, run_time=.2, traverse_only=True)

        np.random.seed(3)
        global k
        k = 3
        self.wait(1)
        nums = create_shaped_data(n=20)
        self.create_world(nums, n_worlds=2)
        self.target.index = len(nums)

        self.create_graph_exhaustive(self.worlds[0], animate=False)
        self.create_graph_exhaustive(self.worlds[1], animate=False)
        
        self.wait(2)
        self.run_time = 1
        self.draw_edges(self.nodes)
        self.wait(1)
        
        self.run_time = .5
        self.traverse(self.world_mps[1][0], self.target, candid=nns, animate_dots=True)
        self.wait(2)

        self.erase_edges()
        self.wait(1)
        self.reset_world()

        self.wait(2)
        self.draw_edges(self.worlds[0])
        self.wait(2)

        start_ix = self.worlds[0][0].index
        self.run_time = .5
        nns = self.traverse(self.world_mps[0][start_ix], self.target, animate_dots=True)
        self.wait(1)

        self.erase_edges()
        self.wait(2)
        nns = [(a, self.world_mps[1][node.index]) for a, node in nns]
        start_ix = nns[0][1].index
        self.draw_edges(self.worlds[1])
        self.wait(2)
        
        self.traverse(self.world_mps[1][start_ix], self.target, candid=nns, animate_dots=False)
        self.erase_edges()
        self.wait(2)
        
        self.clear_world(clear_target=True)
        self.wait(2)
        

        np.random.seed(1)
        n = 6
        x_mn, x_mx, y_mn, y_mx = -5, 5, -3, 3
        nums = np.vstack((np.random.uniform(x_mn, x_mx, n), np.random.uniform(y_mn, y_mx, n))).T
        self.create_n_prune(nums, k=2)
        self.wait(2)
        
        self.run_time = 1
        np.random.seed(1)
        n = 50
        nums = np.vstack((np.random.uniform(x_mn, x_mx, n), np.random.uniform(y_mn, y_mx, n))).T
        self.create_n_prune(nums, k=3, slow=False, arrow=False, animate=False, n_rounds=5, should_clear=False)


        