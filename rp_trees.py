from __future__ import annotations

from manimlib import *
import numpy as np
import random
from collections import defaultdict
from math import inf

random.seed(4)
np.random.seed(0)

x_min, x_max, y_min, y_max = -12, 12, -7, 7
BLUE = "#3B4D61"
RED = "#AF0000"

n_points = 30

x1_ = np.random.normal(-30, 5, n_points)
y1_ = np.random.normal(0, 7, n_points)
x1 = np.stack((x1_, y1_), axis=1) / 5

x2_ = np.random.normal(30, 5, n_points)
y2_ = np.random.normal(0, 7, n_points)
x2 = np.stack((x2_, y2_), axis=1) / 5

x3_ = np.random.normal(-20, 3, n_points)
y3_ = np.random.normal(0, 15, n_points)
x3 = np.stack((x3_, y3_), axis=1) / 5

x4_ = np.random.normal(30, 7, n_points)
y4_ = np.random.normal(0, .1, n_points)
x4 = np.stack((x4_, y4_), axis=1) / 5

angle = np.radians(45)
rot_mat = np.array([[np.cos(angle), np.sin(-angle)], [np.sin(angle), np.cos(angle)]])


x1 = x1 @ rot_mat.T
x2 = x2 @ rot_mat.T
x3 = x3 @ rot_mat.T
x4 = x4 @ rot_mat.T

class Node:
     def __init__(self, val, left, right, nums=None, dot=None):
          self.val = val
          self.left = left
          self.right = right
          self.nums = nums
          self.dot = dot

nums = [-4., -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4]
new_nums = [-5, -4.8, -4.6, -4.4, -4.2, -3, -2.8, -2.6, -2.4, 2.6, 2.8, 3, 3.2, 4.4, 4.6, 4.8, 5]
btree_node = Node(
    val=0.,
    left=Node(
        -2,
        left=Node(
            -3,
            left=None,
            right=None,
            nums=nums[:5]
        ),
        right=Node(
            -1,
            left=None,
            right=None,
            nums=nums[5:9]
        )
    ),
    right=Node(
        2,
        left=Node(
            1,
            left=None,
            right=None,
            nums=nums[9:13]
        ),
        right=Node(
            3,
            left=None,
            right=None,
            nums=nums[13:]
        )
    )
)
btree_node_bad = Node(
    val=0.,
    left=Node(
        -2,
        left=Node(
            -3,
            left=None,
            right=None,
            nums=new_nums[5:9]
        ),
        right=Node(
            -1,
            left=None,
            right=None,
            nums=[]
        )
    ),
    right=Node(
        2,
        left=Node(
            1,
            left=None,
            right=None,
            nums=[]
        ),
        right=Node(
            3,
            left=None,
            right=None,
            nums=new_nums[9:13]
        )
    )
)
tree_node = Node(
    val=0.,
    left=Node(
        -3.6,
        left=Node(
            -4.6,
            left=None,
            right=None,
            nums=new_nums[:5]
        ),
        right=Node(
            -2.7,
            left=None,
            right=None,
            nums=new_nums[5:9]
        )
    ),
    right=Node(
        3.8,
        left=Node(
            2.9,
            left=None,
            right=None,
            nums=new_nums[9:13]
        ),
        right=Node(
            4.7,
            left=None,
            right=None,
            nums=new_nums[13:]
        )
    )
)

start_ht = 3. # (Start height)
lines, lines_bad, tree_lines = [], [], []
paths, paths_bad, tree_paths = [], [], []

def dfs(node, lines, depth=1):
    lines.append(Line(
        [node.val, start_ht - (depth - 1) * .5, 0.],
        [node.val, start_ht - depth * .5, 0.],
        color=BLACK
    ))
    if node.left and node.left.val:
            lines.append(Line(
                [node.val, start_ht - depth * .5, 0.],
                [node.left.val, start_ht - depth * .5, 0.],
                color=BLACK
            ))
            dfs(node.left, lines=lines, depth=depth + 1)
    if node.right and node.right.val:
            lines.append(Line(
                [node.val, start_ht - depth * .5, 0.],
                [node.right.val, start_ht - depth * .5, 0.],
                color=BLACK
            ))
            dfs(node.right, lines=lines, depth=depth + 1)

dfs(btree_node, lines=lines)
dfs(btree_node_bad, lines=lines_bad)
dfs(tree_node, lines=tree_lines)

def dfs_path(node, num, searching=True, ix=0, lines=lines, paths=paths):
    if searching:
        paths.append(lines[ix])

    if node.left is None and node.right is None:
        return ix

    if node.left:
        ix += 1

        _searching = False
        if searching and num <= node.val and node.left.val:
            paths.append(lines[ix])
            _searching = True

        ix = dfs_path(node.left, num, searching=_searching, ix=ix + 1, lines=lines, paths=paths)

    if node.right:
        ix += 1

        _searching = False
        if searching and num > node.val and node.right.val:
            paths.append(lines[ix])
            _searching = True

        ix = dfs_path(node.right, num, searching=_searching, ix=ix + 1, lines=lines, paths=paths)

    return ix

num = -3.8
dfs_path(btree_node, num)
dfs_path(btree_node_bad, num, lines=lines_bad, paths=paths_bad)
dfs_path(tree_node, num, lines=tree_lines, paths=tree_paths)

# BUG: This should be dot product to check which side of the hyperplane
# is better. If taking L2, the RP-Tree won't work. This isn't visualized,
# so no one will know, but still, needs to be fixed.
def l2(p1, p2):
    if isinstance(p1, float) or isinstance(p1, int):
        return (p1 - p2) ** 2
    return ((p1 - p2) ** 2).sum()

def two_means(nums, anim_data=[], n_iters=200):
    i, j = 0, 2
    centroid_i = centroid_i_prev = nums[i]
    centroid_j = centroid_j_prev = nums[j]
    n_i, n_j = 1, 1

    for iter in range(n_iters):
        point = random.choice(nums)

        distance_i = l2(point, centroid_i) # * n_i (this multiplication is there in Annoy)
        distance_j = l2(point, centroid_j) # * n_j
        
        #if iter >= 4:
        #    if iter % 5 == 0:
        #        centroid_i_prev, centroid_j_prev = centroid_i, centroid_j
        #else:
        #    centroid_i_prev, centroid_j_prev = centroid_i, centroid_j

        if distance_i < distance_j:
            centroid_i = (centroid_i * n_i + point) / (n_i + 1)
            n_i += 1
            
            
        if distance_i > distance_j:
            centroid_j = (centroid_j * n_j + point) / (n_j + 1)
            n_j += 1
        
        if True: #iter < 4 or iter % 5 == 0:
            anim_data.append({
                "point": point,
                "centroid_i_prev": centroid_i_prev,
                "centroid_j_prev": centroid_j_prev,
                "centroid_i": centroid_i,
                "centroid_j": centroid_j 
            })
            centroid_i_prev, centroid_j_prev = centroid_i, centroid_j
    
    return centroid_i, centroid_j

skip_run = False
skip_wait = False
run_time = .5

class InteractiveDevelopment(Scene):
    def animate_two_means(self, anim_data, root=np.array([0, 0, 0]), height=2.5, accelerate=1):
        mobjects = []
        centroid_i, centroid_j = anim_data[0]["centroid_i_prev"], anim_data[0]["centroid_j_prev"]
        centroid_i = Dot([centroid_i, height, 0], color=BLACK)
        centroid_j = Dot([centroid_j, height, 0], color=BLACK)
        self.play(GrowFromCenter(centroid_i), run_time=.05)
        self.play(GrowFromCenter(centroid_j), run_time=.05)

        if accelerate == 1:
            self.wait(2)
            
        for i, data in enumerate(anim_data):
            run_time = 1 / ((i + 1) * accelerate)

            point = data["point"]
            centroid_i_prev = data["centroid_i_prev"]
            centroid_j_prev = data["centroid_j_prev"]
            centroid_i_next = data["centroid_i"]
            centroid_j_next = data["centroid_j"]

            dl1 = Line([centroid_i_prev, height, 0], [point, 0, 0], color=BLACK)
            dl2 = Line([centroid_j_prev, height, 0], [point, 0, 0], color=BLACK)

            self.play(ShowCreation(dl1), run_time=run_time)
            self.play(ShowCreation(dl2), run_time=run_time)

            anims = []
            anims.append(ApplyMethod(centroid_i.move_to, [centroid_i_next, height, 0]))
            anims.append(ApplyMethod(dl1.put_start_and_end_on, np.array([centroid_i_next, height, 0]), np.array([point, 0, 0])))
            self.play(AnimationGroup(*anims), run_time=run_time)

            anims = []
            anims.append(ApplyMethod(centroid_j.move_to, [centroid_j_next, height, 0]))
            anims.append(ApplyMethod(dl2.put_start_and_end_on, np.array([centroid_j_next, height, 0]), np.array([point, 0, 0])))
            self.play(AnimationGroup(*anims), run_time=run_time)


            self.play(Uncreate(dl1), run_time=run_time)
            self.play(Uncreate(dl2), run_time=run_time)
        
        centroid_i_center, centroid_j_center = centroid_i.get_center(), centroid_j.get_center()
        connector = Line(centroid_i.get_center(), centroid_j.get_center(), color=BLACK)
        mid_pt = Dot([root[0], height, 0], color=BLACK)

        mid_coords = mid_pt.get_center()
        root_line = Line(np.array([mid_coords[0], mid_coords[1] + .5, 0]), mid_pt, color=BLACK)

        mobjects.append(connector)
        mobjects.append(root_line)
        mobjects.append(mid_pt)

        if accelerate == 1:
            self.wait(2)

        self.play(AnimationGroup(
            ShowCreation(connector),
            GrowFromCenter(mid_pt),
            Uncreate(centroid_i),
            Uncreate(centroid_j),
            ShowCreation(root_line)
        ))


        if accelerate == 1:
            self.wait(2)

        return centroid_i_center, centroid_j_center, mobjects

    def add_points(self, x):
        self.points = []
        anims = []

        for x_, y_ in x:
            point = Dot([x_, y_, 0], radius=.05, color=BLACK)
            self.points.append(point)
            anims.append(GrowFromCenter(point))

        self.play(AnimationGroup(*anims), run_time=run_time)

    def animate_split(self, X,depth=1, animate=True, root_name=None, should_wait=False, run_time=run_time, randomize_split=False):
        centroid_i, centroid_j = two_means(X)

        centroids = []
        for centroid in [centroid_i, centroid_j]:
            dot = Dot([centroid[0], centroid[1], 0], radius=.15, color=BLUE)
            centroids.append(dot)
            if animate:
                self.play(GrowFromCenter(dot), run_time=run_time)
        
        connecting_line = Line(
            [centroid_i[0], centroid_i[1], 0],
            [centroid_j[0], centroid_j[1], 0],
            color=BLUE
        )
        if animate:
            self.play(ShowCreation(connecting_line), run_time=run_time)
            if should_wait:
                self.wait(2)

        if randomize_split:
            mid = (centroid_i + centroid_j) * random.randint(0, 10) / 2
        else:
            mid = (centroid_i + centroid_j) / 2
        slope = (centroid_j[1] - centroid_i[1]) / (centroid_j[0] - centroid_i[0])
        slope = -1 / slope

        if abs(slope) > 1:
            start_y = mid[1] + (y_max // 2) / (depth ** 2)
            start_x = (slope * mid[0] - mid[1] + start_y) / slope
            end_y = mid[1] + (y_min // 2) / (depth ** 2)
            end_x = (slope * mid[0] - mid[1] + end_y) / slope
        else:
            start_x = mid[0] + (x_max // 2) / (depth ** 2)
            start_y = slope * start_x + mid[1] - slope * mid[0]
            end_x = mid[0] + (x_min // 2) / (depth ** 2)
            end_y = slope * end_x + mid[1] - slope * mid[0]

        hyperplane = Line(
            [start_x, start_y, 0],
            [end_x, end_y, 0],
            color=BLUE
        )

        if animate:
            anims = []
            anims.append(ShowCreation(hyperplane))
            mid_dot = Dot([mid[0], mid[1], 0], radius=.15, color=BLUE)
            if root_name is not None:
                dot_text = Text(root_name)
                dot_text.scale(.4)
                dot_text.move_to(mid_dot.get_center())
                mid_dot = VGroup(mid_dot, dot_text)
            anims.append(ShowCreation(mid_dot))
            self.play(AnimationGroup(*anims), run_time=run_time)
            if should_wait:
                self.wait(2)

        anims = []
        for o in centroids + [connecting_line]:
            anims.append(Uncreate(o))
        
        
        if animate:
            self.play(AnimationGroup(*anims), run_time=run_time)

        return centroid_i, centroid_j, mid, hyperplane, mid_dot

    def create_tree(self, X, mx_depth=2, depth=0, hyperplanes=[], roots=[], animate=True, root_id=0, run_time=run_time, randomize_split=False):
        if depth > mx_depth:
            return None, root_id
        

        centroid_i, centroid_j, node, hyperplane, mid_dot = self.animate_split(
            X,
            depth=depth + 1,
            animate=animate,
            should_wait=depth == 0,
            run_time=run_time,
            randomize_split=randomize_split
        )
        hyperplanes.append(hyperplane)
        roots.append(mid_dot)
        root_id += 1

        X1, X2 = [], []
        for x in X:
            if l2(x, centroid_i) <= l2(x, centroid_j):
                X1.append(x)
            else:
                X2.append(x)

        left, root_id = self.create_tree(np.array(X1), depth=depth + 1, hyperplanes=hyperplanes, roots=roots, mx_depth=mx_depth, root_id=root_id)
        right, root_id = self.create_tree(np.array(X2), depth=depth + 1, hyperplanes=hyperplanes, roots=roots, mx_depth=mx_depth, root_id=root_id)

        return Node(node, left=left, right=right, dot=mid_dot), root_id
    
    def dfs(self, node, anims=[], roots=[], root_connectors=[]):
        roots.append(node.dot)
        if node.left:
            line = Line(
                node.dot.get_center(),
                node.left.dot.get_center(),
                color=BLUE
            )
            root_connectors.append(line)
            anims.append(ShowCreation(line))
            self.dfs(node.left, anims=anims, roots=roots, root_connectors=root_connectors)
        if node.right:
            line = Line(
                node.dot.get_center(),
                node.right.dot.get_center(),
                color=BLUE
            )
            root_connectors.append(line)
            anims.append(ShowCreation(line))
            self.dfs(node.right, anims=anims, roots=roots, root_connectors=root_connectors)

    def greedy_search(self, start_dot, target, curr):
        while curr:
            if not curr.left:
                break
            if l2(target[:2], curr.left.val) < l2(target[:2], curr.right.val):
                self.play(start_dot.animate.move_to(np.hstack((curr.left.val, [0]))))
                curr = curr.left
            else:
                self.play(start_dot.animate.move_to(np.hstack((curr.right.val, [0]))))
                curr = curr.right
            self.wait(2)
        return curr

    def bfs(self, root, target, threshold=.1):
        costs = defaultdict(list)
        costs[0].append((root, [root]))
        best_dist, best_node = inf, None
        history = []

        while costs:
            min_cost = min(costs)
            min_cost_queue = costs[min_cost]
            for node, visited in min_cost_queue:
                anim_data = []
                for next_node in [node.left, node.right]:
                    if not next_node:
                        continue
                    if next_node not in visited:
                        dist = l2(next_node.val, target)
                        if dist < threshold:
                            break
                        if dist < best_dist:
                            best_dist, best_node = dist, next_node
                        costs[dist].append((next_node, visited + [next_node]))
                        anim_data.append((node.val, next_node.val))
                if anim_data:
                    history.append(anim_data)
            del costs[min_cost]

        return best_node, history
    
    def forward_through_tree(self, root, target_dot):
        curr = root
        while True:
            if curr.nums is not None:
                break
            if num <= curr.val:
                curr = curr.left
            else:
                curr = curr.right
        
        dashed = []
        for n in curr.nums:
            dashed_line = DashedLine(target_dot.get_center(), [n, 0, 0], color=BLACK)
            dashed.append(dashed_line)
            self.play(ShowCreation(dashed_line), run_time=.1)

        self.wait(2)
        for dashed_line in dashed:
            self.play(Uncreate(dashed_line), run_time=.1)
        self.wait(2)

    def play(self, *args, **kw):
        if skip_run:
            kw["run_time"] = 1e-5
        return super().play(*args, **kw)

    def wait(self, *args, **kw):
        if skip_wait:
            return super().wait(0)
        return super().wait(*args, **kw)
    
    def construct(self):
        global skip_run
        #skip_run = True
        global skip_wait
        #skip_wait = True
        nl2 = NumberLine([-6, 6], include_numbers=True, stroke_color=BLACK)
        self.add(nl2)

        points = []
        for n in nums:
            point = Dot([n, 0., 0.], color=BLACK)
            points.append(point)
            self.play(GrowFromCenter(point), run_time=.05)

        target_dot = Dot([num, 0, 0], color=RED)
        self.play(GrowFromCenter(target_dot))

        self.wait(1)
        self.play(target_dot.animate.move_to([num, 1, 0]))
        self.wait(1)

        connector_lines = []
        for n in nums:
            line = DashedLine(target_dot.get_center(), np.array([n, 0, 0]), color=BLACK)
            self.play(ShowCreation(line), run_time=.05)
            connector_lines.append(line)
        
        self.wait(2)
        for ix, line in enumerate(connector_lines):
            if ix != 0:
                self.play(Uncreate(line), run_time=.1)
        

        self.wait(2)
        self.play(Uncreate(connector_lines[0]), run_time=.1)

        dot = Dot(lines[0].start, color=BLUE)
        self.play(GrowFromCenter(dot), run_time=.03)

        for line in lines:
            self.play(ShowCreation(line), run_time=.1)

        self.wait(2)
        for i, path in enumerate(paths):
            self.play(MoveAlongPath(dot, path), rate_func=linear, run_time=.5)
            if i % 2 == 0:
                self.wait(1)

        self.forward_through_tree(btree_node, target_dot)
        
        for path in paths[::-1]:
            self.play(MoveAlongPath(dot, Line(path.end, path.start)), rate_func=linear, run_time=.1)

        # New dataset
        anims = []
        for point, new_point in zip(points, new_nums):
            _, y, z = point.get_center()
            anims.append(ApplyMethod(point.move_to, [new_point, y, z]))
        anims.append(ApplyMethod(target_dot.move_to, [num, 1, 0]))
        
        self.play(AnimationGroup(*anims))

        self.wait(2)
        #skip_run = skip_wait = False
        for path in paths_bad:
            self.play(MoveAlongPath(dot, path), run_time=1)
        self.wait(2)
        
        self.forward_through_tree(btree_node_bad, target_dot)

        for path in paths_bad[::-1]:
            self.play(MoveAlongPath(dot, Line(path.end, path.start)), rate_func=linear, run_time=.1)


        anims = []
        for line, tree_line in zip(lines, tree_lines):
            anims.append(ApplyMethod(line.put_start_and_end_on, tree_line.start, tree_line.end))

        self.play(AnimationGroup(*anims))

        self.wait(2)
        for i, path in enumerate(tree_paths):
            self.play(MoveAlongPath(dot, path), rate_func=linear, run_time=.5)
            if i % 2 == 0:
                self.wait(1)

        curr = tree_node
        while True:
            if curr.nums is not None:
                break
            if num <= curr.val:
                curr = curr.left
            else:
                curr = curr.right
        
        dashed = []
        for n in curr.nums:
            dashed_line = DashedLine(target_dot.get_center(), [n, 0, 0], color=BLACK)
            dashed.append(dashed_line)
            self.play(ShowCreation(dashed_line), run_time=.1)

        self.wait(2)
        self.play(AnimationGroup(
            *[Uncreate(a) for a in paths],
            *[Uncreate(a) for a in dashed],
            *[Uncreate(a) for a in lines],
            Uncreate(dot),
            Uncreate(target_dot)
        ))

        mobjects = []
        anim_data = []
        two_means(new_nums, anim_data)
        centroid_i, centroid_j, mobjects_ = self.animate_two_means(anim_data)
        mobjects += mobjects_

        anim_data = []
        centroid_i_0, centroid_j_0 = two_means(new_nums[:9], anim_data, n_iters=100)
        _, _, mobjects_ = self.animate_two_means(anim_data, root=centroid_i, height=2, accelerate=10)
        mobjects += mobjects_

        centroid_i_0 = np.array([centroid_i_0, 2, 0])
        centroid_j_0 = np.array([centroid_j_0, 2, 0])
        left_line = Line(centroid_i_0, centroid_i_0 - np.array([0, .5, 0]), color=BLACK)
        right_line = Line(centroid_j_0, centroid_j_0 - np.array([0, .5, 0]), color=BLACK)
        self.play(AnimationGroup(
            ShowCreation(left_line),
            ShowCreation(right_line),
        ))
        mobjects += [left_line, right_line]

        anim_data = []
        centroid_i_1, centroid_j_1 = two_means(new_nums[9:], anim_data, n_iters=100)
        _, _, mobjects_ = self.animate_two_means(anim_data, root=centroid_j, height=2, accelerate=10)
        mobjects += mobjects_

        centroid_i_1 = np.array([centroid_i_1, 2, 0])
        centroid_j_1 = np.array([centroid_j_1, 2, 0])
        left_line = Line(centroid_i_1, centroid_i_1 - np.array([0, .5, 0]), color=BLACK)
        right_line = Line(centroid_j_1, centroid_j_1 - np.array([0, .5, 0]), color=BLACK)
        self.play(AnimationGroup(
            ShowCreation(left_line),
            ShowCreation(right_line),
        ))
        mobjects += [left_line, right_line]

        self.wait(2)
        self.play(AnimationGroup(
            *[Uncreate(a) for a in mobjects + [nl2] + points]
        ))

        self.axes = Axes(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            width=12,
            height=7,
            axis_config={
                "stroke_color": BLACK
            }
        )
        self.play(ShowCreation(self.axes))

        X = np.vstack((x1, x2))
        X = np.vstack([self.axes.c2p(*a) for a in X])[:, :-1]

        self.add_points(X)
        self.wait(2)
        self.play(Uncreate(self.axes))

        hyperplanes = []
        roots = []
        node, _ = self.create_tree(X, hyperplanes=hyperplanes, roots=roots)
        self.wait(2)

        self.play(AnimationGroup(*[Uncreate(hyperplane) for hyperplane in hyperplanes]))

        anims = []
        roots = []
        root_connectors = []
        self.dfs(node, anims=anims, roots=roots, root_connectors=root_connectors)
        self.play(*anims)

        target = [0, 2.5, 0]
        target_dot = Dot(target, color=RED)
        start_dot = Dot(np.hstack((node.val, [0])), color="#ECECEC")
        self.play(AnimationGroup(GrowFromCenter(target_dot), GrowFromCenter(start_dot)), run_time=run_time)
        self.wait(2)

        self.greedy_search(start_dot, target, node)
        
        self.play(AnimationGroup(*[Uncreate(o) for o in roots + root_connectors + [target_dot, start_dot]]))

        self.axes = Axes(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            width=12,
            height=7,
            axis_config={
                "stroke_color": BLACK
            }
        )
        X = np.vstack((x3, x4))
        X = np.vstack([self.axes.c2p(*a) for a in X])[:, :-1]
        self.play(AnimationGroup(*[point.animate.move_to([x[0], x[1], 0]) for point, x in zip(self.points, X)]))

        hyperplanes = []
        roots = []

        target = [-2.5, 3.5, 0]
        target_dot = Dot(target, color=RED)
        self.play(GrowFromCenter(target_dot), run_time=run_time)

        node, _ = self.create_tree(X, hyperplanes=hyperplanes, run_time=.2)
        self.wait(2)
        self.play(AnimationGroup(*[Uncreate(o) for o in hyperplanes]))


        anims = []
        roots = []
        root_connectors = []
        self.dfs(node, anims=anims, roots=roots, root_connectors=root_connectors)
        self.play(*anims)
        start_dot = Dot(np.hstack((node.val, [0])), color="#ECECEC")
        self.play(GrowFromCenter(start_dot), run_time=run_time)

        self.wait(2)
        self.greedy_search(start_dot, target, node)

        self.play(Uncreate(start_dot))
        _, history = self.bfs(node, target[:2])

        dots = [Dot(np.hstack((history[0][0][0], [0])), color="#ECECEC")]
        self.play(ShowCreation(dots[0]))

        for (left_start, left_end), (right_start, right_end) in history:
            self.wait(2)
            start_dot_left = Dot(np.hstack((left_start, [0])), color="#ECECEC")
            start_dot_right = Dot(np.hstack((right_start, [0])), color="#ECECEC")
            self.play(AnimationGroup(
                start_dot_left.animate.move_to(np.hstack((left_end, [0]))),
                start_dot_right.animate.move_to(np.hstack((right_end, [0])))
            ))
            dots += [start_dot_left, start_dot_right]
        
        self.wait(2)

        self.play(AnimationGroup(*[Uncreate(a) for a in roots + root_connectors + dots]))
        hyperplanes = []
        roots = []
        _, _ = self.create_tree(X, hyperplanes=hyperplanes, roots=roots, mx_depth=0)
        self.play(Uncreate(roots[0]))
        self.play(hyperplanes[0].animate.move_to([1.5, 0, 0]))
        self.wait(2)
        self.play(Uncreate(hyperplanes[0]))

        random.seed(0)
        hyperplanes = []
        _, _ = self.create_tree(X, randomize_split=True, hyperplanes=hyperplanes)