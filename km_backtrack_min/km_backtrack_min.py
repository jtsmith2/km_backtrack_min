# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem. Taken from scikit-learn. Based on original code by Brian Clapper,
# adapted to NumPy by Gael Varoquaux.
# Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
# Adaptation to Kuhn-Munkres with Backtracking and Minimum Assignment by Taylor Smith
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD

import numpy as np

def kmb_min_assign(cost_matrix, ability_max_vector, ability_min_vector, task_vector):
    """Extends km_backtrack module to allow for specifying the minimum number of task
    assignments to each agent before they are allowed to have unassigned slots.

    The method used is the Hungarian algorithm, also known as the Munkres or
    Kuhn-Munkres algorithm that has been modified to allow backtracking to 
    solve the many-to-many problem. See:

    Solving the Many to Many assignment problem by improving the Kuhn-Munkres algorithm with backtracking
    Haibin Zhub, Dongning Liua, , , Siqin Zhanga, Yu Zhuc, Luyao Tengd, Shaohua Tenga
    http://dx.doi.org/10.1016/j.tcs.2016.01.002

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    ability_max_vector : list
        An ability limit vector of m agents is La, where La[i] denotes that how many tasks can be 
        assigned to agent i at most (0<i<m)

    ability_min_vector : list
        A limit vector of m agents is Lm, where Lm[i] denotes the minimum number of tasks to be assigned
        to agent i. (0<i<m and 0<=Lm[i]<=La[i])

    task_vector : list
        A task range vector L is a vector of n tasks, where L[j] denotes that quantity of task j   
        must be assigned (0?j<n).

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted.

    Notes
    -----
    .. versionadded:: 0.1.0

    Examples
    --------
    >>> c = np.array([[3,0,1,2],[2,3,0,1],[3,0,1,2],[1,0,2,3]])
    >>> La = [2,2,2,2]
    >>> L = [2,2,2,2]
    >>> agents,tasks = kmb(c,La,L)
    >>> zip(agents,tasks)
    array([(0, 3), (0, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1)])
    >>> c[agents,tasks].sum()
    8

    References
    ----------
    1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
        *Naval Research Logistics Quarterly*, 2:83-97, 1955.

    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
        problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
        *J. SIAM*, 5(1):32-38, March, 1957.

    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    cost_matrix = np.asarray(cost_matrix)
    ability_max_vector = np.asarray(ability_max_vector)
    ability_min_vector = np.asarray(ability_min_vector)
    task_vector = np.asarray(task_vector)

    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                            % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _HungaryBacktrack(cost_matrix, task_vector, ability_max_vector, ability_min_vector)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    if transposed:
        marked = state.marked.T
    else:
        marked = state.marked
    assignments = np.where(marked == 1)

    for i,agent in enumerate(assignments[0]):
        assignments[0][i] = state.agent_row_lookup[agent]
    for j,task in enumerate(assignments[1]):
        if task < task_vector.sum():
            assignments[1][j] = state.task_column_lookup[task]
        else:
            assignments[1][j] = -1

    return assignments

# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z and mark related zeros as unavailable. 
    # Repeat for each element in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i] and state.available[i,j]:
            state._star(i, j)
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    covered_C *= (state.available).astype(int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered, available zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state._prime(row, col)
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int)) * (
                    state.available[:, col])
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state._unstar(path[i, 0], path[i, 1])
        else:
            state._star(path[i, 0], path[i, 1])

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        M = state.C.copy()
        maxval = np.max(M)
        M[np.where(state.available==0)] = maxval
        minval = np.min(M[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered]) #selects the smallest value that is greater than zero
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
        state.C[np.where(state.C<0)]=0
    return _step4

class _HungaryBacktrack(object):
        """State of the Hungarian algorithm.

        Parameters
        ----------
        cost_matrix : 2D matrix
            The cost matrix. Must have shape[1] >= shape[0].
        """

        def __init__(self, cost_matrix, L, La_max, La_min):
            self.C = cost_matrix.copy()
            self.L = L.copy()
            self.La_min = La_min.copy()
            self.La_max = La_max.copy()
            self.n, self.m = self.C.shape

            self._expand_cost_matrix()

            self.k = self.C.shape[0]
            
            self.available = np.ones((self.k,self.k), dtype=bool)    
            
            # Sets the 'no assignment' tasks as unavailable for all agents with a minimum required assignment
            self.available[:,-(self.k-self.L.sum()):] = 0  
            for agent, minimum in enumerate(self.La_min):
                if minimum == 0:
                    self.available[self.agent_rows[self.agent_row_lookup[agent]],-(self.k-self.L.sum()):] = 1
                    
            self.row_uncovered = np.ones(self.k, dtype=bool)
            self.col_uncovered = np.ones(self.k, dtype=bool)
            self.Z0_r = 0
            self.Z0_c = 0
            self.path = np.zeros((2*self.k, 2), dtype=int)
            self.marked = np.zeros((self.k, self.k), dtype=int)

        def __repr__(self):
            M = self.C.astype(np.string_)
            M[np.where(self.marked==1)] = np.char.add( M[np.where(self.marked==1)],'*')
            M[np.where(self.marked==2)] = np.char.add( M[np.where(self.marked==2)],"'")
            M[np.where(self.available==0)] = np.char.add( M[np.where(self.available==0)],'x')
            z = np.zeros((self.k,1),dtype=np.string_)
            M = np.hstack((z,M))
            z = np.zeros((1,self.k+1),dtype=np.string_)
            M = np.vstack((z,M))
            M[0,:] = ''
            M[:,0] = ''
            for i,uc in enumerate(self.row_uncovered):
                if uc == 0:
                    M[i+1,0] = 'c'
            for j,uc in enumerate(self.col_uncovered):
                if uc == 0:
                    M[0,j+1] = 'c'

            return str(M)

        def _clear_covers(self):
            """Clear all covered matrix cells"""
            self.row_uncovered[:] = True
            self.col_uncovered[:] = True

        def _expand_cost_matrix(self):
            """
            Expands matrix C into a KxK matrix where K is the sum of availble agent slots 
            for tasks (the sum of the elements of La).  It does this in 3 steps:
            1. Expand each row by repeating the columns of each task
            2. Expand each column by repeating the rows of each agent
            3. Filling in with columns of zeros to make the matrix KxK

            Example
            --------------
            L = [1,3]
            La = [1,2,2]
            C = [[2,1],
                 [2,1],
                 [1,2]]

            After Step 1 (first column is repeated once, 2nd column is repeated 3x)
            C = [[2,1,1,1],
                 [2,1,1,1],
                 [1,2,2,2]]

            After Step 2 (1st row is repeated once, 2nd and 3rd rows are repeated twice):
            C = [[2,1,1,1],
                 [2,1,1,1],
                 [2,1,1,1],
                 [1,2,2,2],
                 [1,2,2,2]]

            After Step 3 (a column of zeros is added to make the shape 5x5):
            C = [[2,1,1,1,0],
                 [2,1,1,1,0],
                 [2,1,1,1,0],
                 [1,2,2,2,0],
                 [1,2,2,2,0]]
            """
            self.C = np.repeat(self.C, self.L, axis=1)  #step 1
            self.C = np.repeat(self.C, self.La_max, axis=0)  #step 2
            zero_cols = np.ones((self.C.shape[0],self.C.shape[0]-self.C.shape[1]),dtype=int)*(np.max(self.C)+1)
            self.C = np.hstack((self.C,zero_cols)) #step 3

            self.agent_row_lookup = range(self.m)
            self.task_column_lookup = range(self.n)

            self.agent_row_lookup = np.repeat(self.agent_row_lookup,self.La_max)
            self.task_column_lookup = np.repeat(self.task_column_lookup,self.L)

            self.task_columns = {}
            self.agent_rows = {}

            for i,_ in enumerate(self.La_max): 
                self.agent_rows[i]=np.where(self.agent_row_lookup==i)
            for j,_ in enumerate(self.L):
                self.task_columns[j]=np.where(self.task_column_lookup==j)

        def _set_related_unavailable(self,row,col):
            """Sets 'related' cells (same agent, same task, just on different columns and rows) to 
            unavailable so that the same agent can't be assigned to the same task more than once.

            Parameters:
            row - row of agent
            col - column of task
            """
            if col < len(self.task_column_lookup):  #only run for actual tasks (not the extra zero padded columns)
                agent = self.agent_row_lookup[row]
                task = self.task_column_lookup[col]

                related_rows = np.delete(self.agent_rows[agent],np.where(self.agent_rows[agent]==row)[1]) #the related rows, excluding the current row
                related_cols = np.delete(self.task_columns[task],np.where(self.task_columns[task]==col)[1]) #the related cols, excluding the current col
                for i in related_rows:
                    for j in related_cols:
                        if self.marked[i,j]!=1:
                            self.available[i,j] = False

        def _make_all_available(self):
            self.available[:,:] = True
            
        def _set_related_available(self,row,col):

            if col < len(self.task_column_lookup):  #only run for actual tasks (not the extra zero padded columns)
                agent = self.agent_row_lookup[row]
                task = self.task_column_lookup[col]
                related_rows = self.agent_rows[agent] #the related rows
                related_cols = self.task_columns[task] #the related cols
                self.available[related_rows[0][0]:related_rows[0][-1]+1,related_cols[0][0]:related_cols[0][-1]+1] = True

        def _star(self,i,j):
            self.marked[i,j]=1
            self._set_related_unavailable(i,j)
            self._check_minimums(i)

        def _unstar(self,i,j):
            self.marked[i, j] = 0
            self._set_related_available(i, j)
            self._check_minimums(i)

        def _prime(self,i,j):
            self.marked[i,j]=2
            self._set_related_unavailable(i,j)

        def _check_minimums(self,agent):
            assigned = self.marked[self.agent_rows[self.agent_row_lookup[agent]],:-(self.k-self.L.sum())].sum()
            if assigned < self.La_min[self.agent_row_lookup[agent]]:
                self.available[self.agent_rows[self.agent_row_lookup[agent]],-(self.k-self.L.sum()):] = 0
            else:
                self.available[self.agent_rows[self.agent_row_lookup[agent]],-(self.k-self.L.sum()):] = 1



if __name__ == "__main__":
    c = np.array([[3,0,1,2],[2,3,0,1],[3,0,1,2],[1,0,2,3]])
    La_max = [3,3,3,3]
    La_min = [1,1,1,1]
    L = [2,2,2,2]
    c = np.power(c,1)

    agents,tasks = kmb_min_assign(c,La_max,La_min,L)
    cost = c[agents,tasks].sum()
    #for agent, task in zip(agents,tasks):
    #    if task < 0:
    #        cost += np.max(c)
    #    else:
    #        cost += c[agent,task]
    print "Cost:", cost
    print zip(agents,tasks)
