"""Bounded joint-space RRT-Connect, with checked edges and deterministic seeds."""
import numpy as np


def connect(start, goal, low, high, free, seed=0, iterations=350):
    start, goal = np.asarray(start), np.asarray(goal)
    low, high = np.asarray(low), np.asarray(high)
    scale = np.maximum(high-low, 1e-4)
    rng = np.random.default_rng(seed)
    checks = {}
    def valid(q):
        key = tuple(np.round(q, 5))
        if key not in checks:
            checks[key] = free(q)
        return checks[key]
    def edge(a, b):
        n = max(2, int(np.ceil(np.max(np.abs(b-a)/np.minimum(.015, scale*.025)))))
        return all(valid(a+(b-a)*t) for t in np.linspace(0, 1, n+1)[1:])
    if not valid(goal):
        return None
    if edge(start, goal):
        return [start, goal]
    trees = [([start], [-1]), ([goal], [-1])]
    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin([np.linalg.norm((q-target)/scale) for q in nodes]))
        a = nodes[nearest]
        delta = (target-a)/scale
        norm = np.linalg.norm(delta)
        q = target if norm < .12 else a+delta/norm*.12*scale
        if not edge(a, q):
            return None
        nodes.append(q.copy()); parents.append(nearest)
        return len(nodes)-1
    def chain(tree, index):
        nodes, parents = tree
        out = []
        while index >= 0:
            out.append(nodes[index]); index = parents[index]
        return out[::-1]
    for k in range(iterations):
        side = k % 2
        sample = goal if rng.random() < .15 else rng.uniform(low, high)
        ia = extend(trees[side], sample)
        if ia is None:
            continue
        target = trees[side][0][ia]
        for _ in range(40):
            ib = extend(trees[1-side], target)
            if ib is None:
                break
            if np.linalg.norm((trees[1-side][0][ib]-target)/scale) < 1e-5:
                a, b = chain(trees[side], ia), chain(trees[1-side], ib)
                path = a+b[-2::-1] if side == 0 else b+a[-2::-1]
                # Deterministic shortcutting removes sampling detours.
                i = 0
                while i < len(path)-2:
                    j = len(path)-1
                    while j > i+1 and not edge(path[i], path[j]):
                        j -= 1
                    path[i+1:j] = []
                    i += 1
                return path
    return None
