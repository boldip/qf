#!/usr/bin/python3
"""
An implementation of the unordered tree edit distance based on
the A* algorithm. The general A* framing is taken from Yoshino,
Higuchi, and Hirata (2013). But this implementation contains three
novel lower bound functions which work for general cost functions,
one in linear, one in quadratic, and one in cubic time, which provide
tighter and tighter lower bounds on the actual edit distance.

"""
# Copyright (C) 2021
# Benjamin Paaßen
# Humboldt-University of Berlin

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from queue import PriorityQueue
from scipy.optimize import linear_sum_assignment

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@hu-berlin.de'

def outermost_right_leaves(adj):
    """ Computes the outermost right leaves of a tree based on its adjacency
    list. The outermost right leaf of a tree is defined as recursively
    accessing the right-most child of a node until we hit a leaf.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Parameters
    ----------
    adj: list
        An adjacency list representation of the tree, i.e. an array such that
        for every i, adj[i] is the list of child indices for node i.

    Returns
    -------
    orl: int array
        An array containing the outermost right leaf index for every node
        in the tree.

    """
    # the number of nodes in the tree
    m = len(adj)
    # the array into which we will write the outermost right leaves for each
    # node
    orl = np.full(m, -1, dtype=int)
    for i in range(m):
        # keep searching until we hit a node for which the outermost right
        # leaf is already defined or until we hit a leaf
        r = i
        while True:
            # if r has no children, r is the outermost right leaf for i
            if not adj[r]:
                orl[i] = r
                break
            # if the outermost right leaf for r is defined, that is also the
            # outermost right leaf for i
            if orl[r] >= 0:
                orl[i] = orl[r]
                break
            # otherwise, continue searching
            r = adj[r][-1]
    return orl

def parents(adj):
    """ Returns the parent representation of the tree with the given
    adjacency list.

    Parameters
    ----------
    adj: list
        The adjacency list of the tree.

    Returns
    -------
    par: int array
        a numpy integer array with len(adj) elements, where the ith
        element contains the index of the parent of the ith node.
        Nodes without children contain the entry -1.

    """
    par = np.full(len(adj), -1, dtype=int)
    for i in range(len(adj)):
        for j in adj[i]:
            par[j] = i
    return par

class YoshinoHeuristic:
    """ A histogram-based heuristic in linear time which bounds the
    standard unordered tree edit distance between two forests by the
    difference in size, the distance between their degree histograms, and
    the distance between their label histograms.

    This heuristic is from Yoshino, Higuchi, and Hirata (2013).

    Attributes
    ----------
    x_labels: ndarray
        An array storing an index identifying each node label in x.
    x_degs: ndarray
        An array storing the degree for each node in x.
    y_labels: ndarray
        An array storing an index identifying each node label in y.
    y_degs: ndarray
        An array storing the degree for each node in y.
    self.max_degree: int
        The maximum degree which occurs in x or y.
    self.num_labels: int
        The number of distinct labels which occur in x or y.

    """
    def __init__(self, x_nodes, x_adj, y_nodes, y_adj):
        """ Sets up the Yoshino heuristic by pre-computing all attributes.

        Parameters
        ----------
        x_nodes: list
            The node list for the entire tree x.
        x_adj: list
            The adjacency list for the entire tree x.
        y_nodes: list
            The node list for the entire tree y.
        y_adj: list
            The adjacency list for the entire tree y.

        """
        
        self.x_nodes = x_nodes
        self.x_adj = x_adj
        self.y_nodes = y_nodes
        self.y_adj = y_adj

        # pre-compute the degrees and max degrees for all nodes
        self.x_degs = np.array([len(adj) for adj in self.x_adj])
        self.y_degs = np.array([len(adj) for adj in self.y_adj])
        self.max_degree = max(np.max(self.x_degs), np.max(self.y_degs))

        # pre-compute index representation of labels
        label_map = {}
        for sym in (self.x_nodes + self.y_nodes):
            label_map.setdefault(sym, len(label_map))
        # pre-compute index representation for all nodes
        self.x_labels = [label_map[sym] for sym in self.x_nodes]
        self.y_labels = [label_map[sym] for sym in self.y_nodes]
        self.num_labels = len(label_map)

    def apply(self, i, k, remaining):
        """ Applies the heuristic to compare two subforests of the input tree,
        namely the subforest x[i:k+1] and the subforest y[remaining].

        Parameters
        ----------
        i: int
            The start index of the subforest in x.
        k: int
            The end index of the subforest in x.
        remaining: set
            The set of indices for the suborest in y.

        Returns
        -------
        d: float
            A lower bound for the unordered tree edit distance between
            the subforest x[i:k+1] and the subforest y[remaining].

        """
        m = k - i
        n = len(remaining)
        # if m or n is zero, compute the value quickly
        if m == 0 or n == 0:
            return abs(m - n)

        # compute the degree histogram for x and y
        hist_deg_x = np.zeros(self.max_degree+1, dtype=int)
        for i2 in range(i, k):
            hist_deg_x[self.x_degs[i2]] += 1
        hist_deg_y = np.zeros(self.max_degree+1, dtype=int)
        for j2 in remaining:
            hist_deg_y[self.y_degs[j2]] += 1

        # compute the label histogram for x and y
        hist_labels_x = np.zeros(self.num_labels, dtype=int)
        for i2 in range(i, k):
            hist_labels_x[self.x_labels[i2]] += 1
        hist_labels_y = np.zeros(self.num_labels, dtype=int)
        for j2 in remaining:
            hist_labels_y[self.y_labels[j2]] += 1

        # compute five lower-bounding functions.
        bounds = np.zeros(5)
        # first, the difference of sizes
        bounds[0] = abs(m - n)
        # second, the l1 distance between degree histograms divided by 3
        bounds[1] = np.sum(np.abs(hist_deg_x - hist_deg_y))/ 3.
        # third, the maximum distance between degree histograms
        bounds[2] = np.max(np.abs(hist_deg_x - hist_deg_y))
        # fourth, the l1 distance between label histograms divided by 2
        bounds[3] = np.sum(np.abs(hist_labels_x - hist_labels_y))/ 2.
        # fifth, the maximum distance between label histograms
        bounds[4] = np.max(np.abs(hist_labels_x - hist_labels_y))

        return np.max(bounds)


class LinearHeuristic:
    """ A linear-time heuristic which lower-bounds the unordered
    tree edit distance between two forests by the minimal cost of deletions
    and insertions we have to apply, ignoring all replacement costs.

    Attributes
    ----------
    dels: ndarray
        An m-element vector of deletion costs.
    inss: ndarray
        An n-element vector of insertion costs.
    reps: ndarray
        An m x n matrix of replacement costs (will be ignored)

    """
    def __init__(self, dels, inss, reps):
        self.dels = dels
        self.inss = inss

    def apply(self, i, k, remaining):
        """ Applies the heuristic to compare two subforests of the input tree,
        namely the subforest x[i:k+1] and the subforest y[remaining].

        Parameters
        ----------
        i: int
            The start index of the subforest in x.
        k: int
            The end index of the subforest in x.
        remaining: set
            The set of indices for the suborest in y.

        Returns
        -------
        d: float
            A lower bound for the unordered tree edit distance between
            the subforest x[i:k+1] and the subforest y[remaining].

        """
        m = k - i
        remaining = list(remaining)
        n = len(remaining)
        # if m or n is zero, compute the value quickly
        if m == 0:
            if n == 0:
                return 0.
            return self.inss[remaining].sum()
        if n == 0:
            return self.dels[i:k].sum()
        # check which input is smaller
        if m == n:
            return 0.
        elif m < n:
            # use the numpy partition function to find the n - m smallest
            # insertion costs without having to do an nlogn sorting
            l = n - m
            sorted_inss = np.partition(self.inss[remaining], l)
            return np.sum(sorted_inss[:l])
        else:
            # use the numpy partition function to find the n - m smallest
            # deletion costs without having to do an nlogn sorting
            l = m - n
            sorted_dels = np.partition(self.dels[i:k], l)
            return np.sum(sorted_dels[:l])


class QuadraticHeuristic:
    """ A quadratic-time heuristic which lower-bounds the unordered
    tree edit distance between two forests by the minimal costs of deletions
    and insertions we have to apply, plus the minimal replacement costs for
    all remaining nodes. The quadratic time occurs because we have to search
    for the minimum replacement cost for all nodes.

    Attributes
    ----------
    dels: ndarray
        An m-element vector of deletion costs.
    inss: ndarray
        An n-element vector of insertion costs.
    reps: ndarray
        An m x n matrix of replacement costs.

    """
    def __init__(self, dels, inss, reps):
        self.dels = dels
        self.inss = inss
        self.reps = reps

    def apply(self, i, k, remaining):
        """ Applies the heuristic to compare two subforests of the input tree,
        namely the subforest x[i:k+1] and the subforest y[remaining].

        Parameters
        ----------
        i: int
            The start index of the subforest in x.
        k: int
            The end index of the subforest in x.
        remaining: set
            The set of indices for the suborest in y.

        Returns
        -------
        d: float
            A lower bound for the unordered tree edit distance between
            the subforest x[i:k+1] and the subforest y[remaining].

        """
        m = k - i
        remaining = list(remaining)
        n = len(remaining)
        # if m or n is zero, compute the value quickly
        if m == 0:
            if n == 0:
                return 0.
            return self.inss[remaining].sum()
        if n == 0:
            return self.dels[i:k].sum()
        # check which input is smaller
        if m == n:
            return 0.
        elif m < n:
            l = n - m
            # find the nodes for which we win the most/lose the
            # least by inserting them instead of replacing them
            minrep = np.min(self.reps[i:k, remaining], 0)
            # use the numpy argpartition function to find the l smallest
            # insertion costs without having to do an nlogn sorting
            ins_remaining = self.inss[remaining]
            inserted = np.argpartition(ins_remaining - minrep, l)[:l]
            minrep[inserted] = 0.
            return np.sum(ins_remaining[inserted]) + np.sum(minrep)
        else:
            l = m-n
            # find the nodes for which we win the most/lose the
            # least by deleting them instead of replacing them
            minrep = np.min(self.reps[i:k, remaining], 1)
            # use the numpy argpartition function to find the l smallest
            # deletion costs without having to do an nlogn sorting
            dels_remaining = self.dels[i:k]
            deleted = np.argpartition(dels_remaining - minrep, l)[:l]
            minrep[deleted] = 0.
            return np.sum(dels_remaining[deleted]) + np.sum(minrep)



class CubicHeuristic:
    """ A cubic-time heuristic which lower-bounds the unordered
    tree edit distance between two forests by the Hungarian
    algorithm (which takes cubic time).

    Attributes
    ----------
    dels: ndarray
        An m-element vector of deletion costs.
    inss: ndarray
        An n-element vector of insertion costs.
    reps: ndarray
        An m x n matrix of replacement costs.

    """
    def __init__(self, dels, inss, reps):
        self.dels = dels
        self.inss = inss
        self.reps = reps

    def apply(self, i, k, remaining):
        """ Applies the heuristic to compare two subforests of the input tree,
        namely the subforest x[i:k+1] and the subforest y[remaining].

        Parameters
        ----------
        i: int
            The start index of the subforest in x.
        k: int
            The end index of the subforest in x.
        remaining: set
            The set of indices for the suborest in y.

        Returns
        -------
        d: float
            A lower bound for the unordered tree edit distance between
            the subforest x[i:k+1] and the subforest y[remaining].

        """
        m = k-i
        remaining = list(remaining)
        n = len(remaining)
        # if m or n is zero, compute the value quickly
        if m == 0:
            if n == 0:
                return 0.
            return self.inss[remaining].sum()
        if n == 0:
            return self.dels[i:k].sum()
        # set up a cost matrix for the set edit distance
        C = np.full((m+n, m+n), np.inf)
        C[:m, :n] = self.reps[i:k, remaining]
        for i2 in range(m):
            C[i2, n+i2] = self.dels[i+i2]
        for j2 in range(n):
            C[m+j2, j2] = self.inss[remaining[j2]]
        C[m:, n:] = 0.
        # apply the Hungarian algorithm to find the optimal
        # assignment
        I, J = linear_sum_assignment(C)
        # return the edit distance
        return C[I, J].sum()


def uted_astar(x_nodes, x_adj, y_nodes, y_adj, delta = None, heuristic = 1, verbose = False):
    """ Computes the unordered tree edit distance between two
    trees via an A* algorithm. The lower bounding heuristic
    is based on the unordered set edit distance, which can be
    approximated at three levels of accuracy, each of which
    is slower.

    Parameters
    ----------
    x_nodes: list
        Nodes of the first tree.
    x_adj: list
        Adjacency list of the first tree. Note that the tree must be
        in depth first search order.
    y_nodes: list
        Nodes of the second tree.
    y_adj: list
        Adjacency list of the second tree. Note that the tree must be
        in depth first search order.
    delta: function (default = None)
        a function that takes two nodes as inputs and returns their pairwise
        distance, where delta(x, None) should be the cost of deleting x and
        delta(None, y) should be the cost of inserting y. If undefined, we use
        unit costs.
    heuristic: int (default = 1)
        Level of the heuristic, either 1, 2, 3, or 'yoshino'. Heuristics 1,
        2, and 3 can handle generic delta functions, whereas the yoshino
        heuristic is limited to unit distances. Note that the runtime is
        different between heuristics: Yoshino and 1 are linear-time,
        2 is quadratic, and 3 is cubic time. However, 2 provides a tighter
        lower bound than 1, and 3 a tighter bound than 2.

    Returns
    -------
    d: float
        The unordered tree edit distance
    alignment: list
        The alignment corresponding to the distance. x[i] is aligned with
        y[alignment[i]]. If alignment[i] = -1, this means that node i is
        deleted.
    search_size: int
        The number of nodes in the edit distance search tree.

    """

    # pre-compute deletion, insertion, and replacement costs
    m = len(x_nodes)
    n = len(y_nodes)

    if delta is None:
        dels = np.ones(m)
        inss = np.ones(n)
        reps = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if x_nodes[i] != y_nodes[j]:
                    reps[i, j] = 1.
    else:
        dels = np.zeros(m)
        for i in range(m):
            dels[i] = delta(x_nodes[i], None)
        inss = np.zeros(n)
        for j in range(n):
            inss[j] = delta(None, y_nodes[j])
        reps = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                reps[i, j] = delta(x_nodes[i], y_nodes[j])

    # set heuristic
    if heuristic == 'yoshino':
        if delta is not None:
            raise ValueError('Yoshino heuristic only works with unit costs.')
        h = YoshinoHeuristic(x_nodes, x_adj, y_nodes, y_adj)
    elif heuristic == 1:
        h = LinearHeuristic(dels, inss, reps)
    elif heuristic == 2:
        h = QuadraticHeuristic(dels, inss, reps)
    elif heuristic == 3:
        h = CubicHeuristic(dels, inss, reps)
    else:
        raise ValueError('Heuristic must be 1, 2, 3, or \'yoshino\'.')

    # pre-compute outermost right leaves for x
    x_orl = outermost_right_leaves(x_adj)
    # pre-compute outermist right leaves for y
    y_orl = outermost_right_leaves(y_adj)
    # pre-compute parents for x
    x_pi = parents(x_adj)
    # pre-compute parents for y
    y_pi = parents(y_adj)

    # set up a list of nodes in the edit distance search tree
    # note: in the edit distance search tree, depth codes the
    # index i in the left tree and the label codes the index j
    # in the right tree. The path from the root to a node
    # constitutes a partial alignment of both trees.
    edist_nodes   = [0]
    # set up the parent reference in the edit distance search tree
    edist_parents = [-1]
    # set up a storage for heuristic sibling alingment costs
    edist_h_sibs  = [0]
    # set up a priority queue which stores the currently best
    # node
    Q = PriorityQueue()
    g = reps[0, 0]
    h_child = h.apply(1, m, list(range(1, n)))
    Q.put((g + h_child, g, 0))

    best_distance = np.inf
    best_solution = None

    # keep searching
    while not Q.empty():
        # pop the edit path with the currently best lower
        # bound
        f_lo_parent, g_parent, v_parent = Q.get()

        if verbose:
            print('remaining best lower bound: %g; current best solution: %g' % (f_lo_parent, best_distance))

        # if the lower bound is already higher than the currently best
        # solution, stop the search here
        if f_lo_parent >= best_distance:
            break

        # extract which nodes are already aligned in the current branch
        # of the search
        partial_alignment = []
        u = v_parent
        while u >= 0:
            partial_alignment.append(edist_nodes[u])
            u = edist_parents[u]
        partial_alignment.reverse()
        i = len(partial_alignment)

        if verbose:
            print('trying to align node %d from partial alignment %s' % (i, str(partial_alignment)))

        # extract which nodes in the right-hand-side tree are still
        # available for alignment.
        # For this purpose, we start with the nodes that are
        # explicitly mentioned in the partial alignment and fill in
        # all nodes on ancestor paths towards the root
        taken = set(partial_alignment)
        # we have to remove nodes on the path from the root
        # to aligned nodes, though, because these are already
        # counted in g_parent
        for j in partial_alignment:
            if j < 1:
                continue
            l = y_pi[j]
            while l > 0 and l not in taken:
                taken.add(l)
                l = y_pi[l]
        # next, we check if there are any nodes left to align in x
        if i == len(x_nodes):
            # if this is not the case, we can construct a complete
            # solution by inserting all remaining nodes from y
            remaining = set(range(n)) - taken
            dist = g_parent
            for j in remaining:
                dist += inss[j]
            # if the solution is better than the current best, store it
            if dist < best_distance:
                best_distance = dist
                best_solution = partial_alignment
            continue

        # next, we search upwards in x until we find 
        # an aligned ancestor (at worst: the root of x).
        # Let x[k] be this ancestor and let y[l] be its partner in y.
        # Then, we can only align descendants of x[k] to descendants of
        # y[l] (which are not yet taken).
        k = x_pi[i]
        while partial_alignment[k] < 0:
            k = x_pi[k]
        l = partial_alignment[k]
        # compute all descendants of y[l] and remove all nodes which are
        # already aligned
        remaining = set(range(l, y_orl[l]+1)) - taken

        # recover the heuristic cost for aligning all right siblings
        # of x[k], which we have already computed before
        # for that purpose, we first have to retrieve the node in
        # the edit distance search tree
        u = v_parent
        for r in range(k+1, i):
            u = edist_parents[u]
        # and then get its h_sib_value
        h_sib_parent = edist_h_sibs[u]

        # now, construct all children in the edit distance search tree.

        # The first child represents the deletion of i.
        v = len(edist_nodes)
        edist_nodes.append(-1)
        edist_parents.append(v_parent)
        # Accordingly, our new g value is the previous g value plus
        # the deletion costs for i
        g = g_parent + dels[i]
        # the heuristic value for aligning children is zero because by
        # deleting x[i] we raise all its children to the sibling level
        h_child = 0.
        # the heuristic value for aligning siblings concerns all
        # descendants of k which are after i in depth first search
        # and all remaining nodes in y which we computed before.
        # To this, we add the sibling alignment cost for k.
        h_sib = h.apply(i+1, x_orl[k]+1, remaining) + h_sib_parent
        edist_h_sibs.append(h_sib)
        # put node onto the priority queue
        Q.put((g + h_child + h_sib, g, v))

        if verbose:
            print('deletion cost %g, h_sib = %g' % (g, h_sib))

        # The remaining children in the edit distance search tree concern
        # the alignment of i with a node that is in remaining
        for j in remaining:
            # add the alignment of i with j to the edit distance search tree
            v = len(edist_nodes)
            edist_nodes.append(j)
            edist_parents.append(v_parent)
            # our new g value if the parent g value plus the replacement cost
            # of i with j plus the insertion of all nodes on a path from l to j
            taken_j = taken.copy()
            taken_j.add(j)
            g = g_parent + reps[i, j]
            l2 = y_pi[j]
            while l2 != l:
                g += inss[l2]
                taken_j.add(l2)
                l2 = y_pi[l2]
            # the heuristic value for aligning children concerns the
            # descendants of i and the descendants of j
            remaining_child = set(range(j+1, y_orl[j]+1)) - taken_j
            h_child = h.apply(i+1, x_orl[i]+1, remaining_child)
            # the heuristic value for aligning siblings concerns the
            # remaining descendants of k and l after removing the
            # descendants of i and j
            remaining_j = remaining - set(range(j+1, y_orl[j]+1)).union(taken_j)
            # and we add the heuristic sibling value for the parent
            h_sib = h.apply(x_orl[i]+1, x_orl[k]+1, remaining_j) + h_sib_parent

            edist_h_sibs.append(h_sib)
            # put node onto the priority queue
            Q.put((g + h_child + h_sib, g, v))

            if verbose:
                print('replacement cost with %d: g = %g, h_child = %g, h_sib = %g' % (j, g, h_child, h_sib))


    return best_distance, best_solution, len(edist_nodes)

def uted_constrained(x_nodes, x_adj, y_nodes, y_adj, delta = None):
    """ Implements the constrained unordered tree edit distance algorithm
    of Zhang (1996).

    In this scheme, we do not permit mappings between disjoint subtrees.
    Equivalently, one can say that a deletion does not just delete a single
    node but forces all child subtrees except for a single one to be
    deleted as well (and the same for insertions). While this does restrict
    the expressiveness of the edit distance, it does make it polynomial.


    Parameters
    ----------
    x_nodes: list
        Nodes of the first tree.
    x_adj: list
        Adjacency list of the first tree. Note that the tree must be
        in depth first search order.
    y_nodes: list
        Nodes of the second tree.
    y_adj: list
        Adjacency list of the second tree. Note that the tree must be
        in depth first search order.
    delta: function (default = None)
        a function that takes two nodes as inputs and returns their pairwise
        distance, where delta(x, None) should be the cost of deleting x and
        delta(None, y) should be the cost of inserting y. If undefined, we use
        unit costs.

    Returns
    -------
    d: float
        The unordered tree edit distance

    """
    # pre-compute deletion, insertion, and replacement costs first.
    m = len(x_nodes)
    n = len(y_nodes)

    if delta is None:
        dels = np.ones(m)
        inss = np.ones(n)
        reps = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if x_nodes[i] != y_nodes[j]:
                    reps[i, j] = 1.
    else:
        dels = np.zeros(m)
        for i in range(m):
            dels[i] = delta(x_nodes[i], None)
        inss = np.zeros(n)
        for j in range(n):
            inss[j] = delta(None, y_nodes[j])
        reps = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                reps[i, j] = delta(x_nodes[i], y_nodes[j])

    # next, pre-compute deletion and insertion costs for subtrees and
    # subforests rooted at each node.
    D_tree   = np.zeros((m+1, n+1))
    D_forest = np.zeros((m+1, n+1))

    for i in range(m-1, -1, -1):
        # Compute the subforest deletion cost for i, i.e. the cost
        # for deleting all of i's child subtrees
        for c in x_adj[i]:
            D_forest[i, n] += D_tree[c, n]
        # Deleting the tree rooted at i means deleting node i and all its
        # children
        D_tree[i, n] = dels[i] + D_forest[i, n]

    for j in range(n-1, -1, -1):
        # Compute the subforest insertion cost for j, i.e. the cost
        # for inserting all of j's child subtrees
        for c in y_adj[j]:
            D_forest[m, j] += D_tree[m, c]
        # Inserting the tree rooted at j means inserting node j and all its
        # children
        D_tree[m, j] = inss[j] + D_forest[m, j]

    # now, start the actual recursion
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            m_i = len(x_adj[i])
            n_j = len(y_adj[j])
            # First, we compute the forest edit distance, i.e. the cost for
            # editing all children of i into all children of j.

            # We consider first the special case that either i or
            # j have no children. Then, the computation is really
            # simple because we can only delete/insert
            if m_i == 0:
                D_forest[i, j] = D_forest[m, j]
            elif n_j == 0:
                D_forest[i, j] = D_forest[i, n]
            else:
                # if both nodes have children, perform the actual computation.
                # For that, we have three options.
                # First, we could delete all children of i except for a single
                # subtree
                del_options = np.zeros(m_i)
                for c in range(m_i):
                    i_c = x_adj[i][c]
                    # accordingly, we need to consider the cost of editing
                    # the children of node i_c with the children of j,
                    # plus the cost of deleting all other children of i.
                    del_options[c] = D_forest[i_c, j] + D_forest[i, n] - D_forest[i_c, n]
                del_cost = np.min(del_options)
                # Second, we could insert all children of j except for a single
                # subtree
                ins_options = np.zeros(n_j)
                for c in range(n_j):
                    j_c = y_adj[j][c]
                    # accordingly, we need to consider the cost of editing
                    # the children of node i to the children of j_c,
                    # plus the cost of inserting all other children of j.
                    ins_options[c] = D_forest[i, j_c] + D_forest[m, j] - D_forest[m, j_c]
                ins_cost = np.min(ins_options)
                # Third, we could replace, meaning that we optimally match all
                # children of i to all children of j. We use the Hungarian
                # algorithm for this purpose.
                # prepare a cost matrix for the Hungarian algorithm
                C = np.full((m_i + n_j, m_i + n_j), np.inf)
                for ci in range(m_i):
                    for cj in range(n_j):
                        # matching ci with cj means editing the ci'th
                        # child of i to the cj'th child of j.
                        C[ci, cj] = D_tree[x_adj[i][ci], y_adj[j][cj]]
                for c in range(m_i):
                    # matching c with n_j + c means deleting the
                    # c'th child of i
                    C[c, n_j + c] = D_tree[x_adj[i][c], n]
                for c in range(n_j):
                    # matching m_i + c with c means inserting the
                    # c'th child of j
                    C[m_i + c, c] = D_tree[m, y_adj[j][c]]
                C[m_i:, n_j:] = 0.
                # solve the linear sum assignment problem for C. The resulting
                # minimum cost is our replacement cost
                I, J = linear_sum_assignment(C)
                rep_cost = C[I, J].sum()
                # compute minimum across deletion, insertion, and replacement
                D_forest[i, j] = min3_(del_cost, ins_cost, rep_cost)

            # next, compute the unordered tree edit distance between the
            # subtrees rooted at i and j.
            # Again, we have three options.
            # First, we could delete node i and all subtrees except a
            # single one.
            if m_i == 0:
                del_cost = D_tree[m, j]
            else:
                del_options = np.zeros(m_i)
                for c in range(m_i):
                    i_c = x_adj[i][c]
                    # accordingly, we need to consider the cost of editing
                    # tree i_c into tree j plus the cost of deleting all other
                    # children of i.
                    del_options[c] = D_tree[i_c, j] + D_tree[i, n] - D_tree[i_c, n]
                del_cost = np.min(del_options)
            # Second, we could insert node j and all children of j except
            # for a single one.
            if n_j == 0:
                ins_cost = D_tree[i, n]
            else:
                ins_options = np.zeros(n_j)
                for c in range(n_j):
                    j_c = y_adj[j][c]
                    # accordingly, we need to consider the cost of editing
                    # tree i into tree j_c plus the cost o inserting all other
                    # children of j.
                    ins_options[c] = D_tree[i, j_c] + D_tree[m, j] - D_tree[m, j_c]
                ins_cost = np.min(ins_options)
            # Third, we could replace, meaning that we edit node i into
            # node j and all children of i into all children of j
            rep_cost = reps[i, j] + D_forest[i, j]
            # compute minimum across deletion, insertion, and replacement
            D_tree[i, j] = min3_(del_cost, ins_cost, rep_cost)

    # Once the recursion is complete, return the first element
    return D_tree[0, 0]

def min3_(x, y, z):
    if x < y:
        if x < z:
            return x
        else:
            return z
    else:
        if y < z:
            return y
        else:
            return z
