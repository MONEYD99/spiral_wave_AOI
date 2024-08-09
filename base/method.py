import numpy as np
from scipy.ndimage import zoom
from scipy.sparse import coo_matrix

# Reaction diffusion system
class difsys:
    '''
    2D media is expanded into 1D and multiplied by the connection matrix
    '''
    def __init__(self, N):
        self.N = N
        self.conn_matrix = None
        self.neighbor_list = None

    # Diffusion matrix (first-order approximation)
    def create_diffusion_matrix_4(self):
        N = self.N
        diff_matrix = np.zeros((N*N, N*N))
        neighbor_list = np.zeros((N*N, 5), dtype=int) # Dimension of input = 5
        for i in range(N*N):   
            # Periodic boundary
            # left  = (i - 1) if i % N != 0 else (i + N - 1)         
            # right = (i + 1) if (i + 1) % N != 0 else (i - N + 1)   
            # upper = (i - N) if i >= N else (i + N*(N-1))           
            # lower = (i + N) if i < N*(N-1) else (i - N*(N-1))      
            # No-flow boundary
            left  = (i - 1) if i % N != 0 else i        # The left neighbor is itself
            right = (i + 1) if (i + 1) % N != 0 else i  # The right neighbor is itself
            upper = (i - N) if i >= N else i            # The upper neighbor is itself
            lower = (i + N) if i < N*(N-1) else i       # The lower neighbor is itself
            # Node connection matrix
            diff_matrix[i, left ] = 1
            diff_matrix[i, right] = 1
            diff_matrix[i, upper] = 1
            diff_matrix[i, lower] = 1
            # Neighbor number table
            neighbor_list[i] = [left, right, lower, upper, i]
        self.conn_matrix = diff_matrix
        self.neighbor_list = neighbor_list

    # Diffusion matrix (second-order approximation)
    def create_diffusion_matrix_8(self):
        N = self.N
        diff_matrix = np.zeros((N*N, N*N))
        neighbor_list = np.zeros((N*N, 9), dtype=int)  # Dimension of input = 9

        for i in range(N*N):
            # No-flow boundary
            left  = (i - 1) if i % N != 0 else i        
            right = (i + 1) if (i + 1) % N != 0 else i  
            upper = (i - N) if i >= N else i            
            lower = (i + N) if i < N*(N-1) else i       
            # Next-nearest neighbor
            upper_left  = (i - N - 1) if (i >= N and i % N != 0) else i             
            upper_right = (i - N + 1) if (i >= N and (i + 1) % N != 0) else i       
            lower_left  = (i + N - 1) if (i < N*(N-1) and i % N != 0) else i        
            lower_right = (i + N + 1) if (i < N*(N-1) and (i + 1) % N != 0) else i  
            # Node connection matrix
            diff_matrix[i, left ] = 1
            diff_matrix[i, right] = 1
            diff_matrix[i, upper] = 1
            diff_matrix[i, lower] = 1
            diff_matrix[i, upper_left]  = 0.5
            diff_matrix[i, upper_right] = 0.5
            diff_matrix[i, lower_left]  = 0.5
            diff_matrix[i, lower_right] = 0.5
            # Neighbor number table
            neighbor_list[i] = [left, right, lower, upper, upper_left, upper_right, lower_left, lower_right, i]
        self.conn_matrix = diff_matrix
        self.neighbor_list = neighbor_list

    # Laplacian matrix
    def to_laplacian_matrix(self):
        degree_matrix = np.diag(np.sum(self.conn_matrix, axis=1))
        laplacian_matrix = self.conn_matrix - degree_matrix   # Electrical coupling form, other node - this node
        self.conn_matrix = laplacian_matrix

    # Sparse matrix
    def to_sparse_matrix(self):
        # Find the indices of non-zero elements
        nonzero_indices = np.nonzero(self.conn_matrix)
        rows = nonzero_indices[0]
        cols = nonzero_indices[1]
        # Get the values of non-zero elements
        values = self.conn_matrix[rows, cols]
        # Create a COO-format sparse matrix
        sparse_matrix = coo_matrix((values, (rows, cols)), shape=self.conn_matrix.shape)
        self.conn_matrix = sparse_matrix
    
    def __call__(self):
        # self.create_diffusion_matrix_4()
        self.create_diffusion_matrix_8()
        self.to_laplacian_matrix()
        self.to_sparse_matrix()
        
        return self.conn_matrix, self.neighbor_list


# Jacobian determinant method
class JCB:
    '''
    Teng-Chao Li et al.
    Jacobian-determinant method of identifying phase singularity during reentry.
    PHYSICAL REVIEW E 98, 062405 (2018) doi.org/10.1103/PhysRevE.98.062405
    '''
    def __init__(self, l, JCB_th):
        self.l = l
        self.JCB_th = JCB_th

    def cal(self, vtime1, vtime0):
        l = self.l
        # Calculate Jacobi
        v_time_x1 = (vtime1[1:l, 0:l-1] + vtime1[1:l, 1:l] - vtime1[0:l-1, 0:l-1] - vtime1[0:l-1, 1:l]) / 2
        v_time_y1 = (vtime1[0:l-1, 1:l] + vtime1[1:l, 1:l] - vtime1[0:l-1, 0:l-1] - vtime1[1:l, 0:l-1]) / 2
        v_time_x2 = (vtime0[1:l, 0:l-1] + vtime0[1:l, 1:l] - vtime0[0:l-1, 0:l-1] - vtime0[0:l-1, 1:l]) / 2
        v_time_y2 = (vtime0[0:l-1, 1:l] + vtime0[1:l, 1:l] - vtime0[0:l-1, 0:l-1] - vtime0[1:l, 0:l-1]) / 2
        jcb = np.abs(v_time_x1 * v_time_y2 - v_time_y1 * v_time_x2)  # Ignoring chirality
        vtime0 = vtime1.copy()                  # Sampling replacement

        jcb = zoom(jcb, (l/(l-1), l/(l-1)))     # Scale the matrix from 99 to 100 to fit the network
        jcb[jcb < self.JCB_th] = 0              # The JCB exceeding the threshold is identified as PS

        return jcb, vtime0


# Topological charge density method
class TCD:
    '''
    Yin-Jie He et al.
    Topological charge-density method of identifying phase singularities in cardiac fibrillation.
    PHYSICAL REVIEW E 104, 014213 (2021) doi.org/10.1103/PhysRevE.104.014213
    '''
    def __init__(self, l, v_mean):
        self.l = l
        self.v_mean = v_mean

    def cal_phase(self, vtime1, vtime0):
        ph = np.arctan2(vtime1 - self.v_mean, vtime0 - self.v_mean)
        vtime0 = vtime1.copy() # Sampling replacement

        return ph, vtime0

    def cal_density(self, ph):
        l = self.l
        ph_dx = np.mod(ph[:, 1:l] - ph[:,:l-1] + np.pi, 2 * np.pi) - np.pi  
        ph_dy = np.mod(ph[:l-1,:] - ph[1:l, :] + np.pi, 2 * np.pi) - np.pi
        ph_dxdy = ph_dx[1:l, :] - ph_dx[:l-1,:]
        ph_dydx = ph_dy[:,:l-1] - ph_dy[:, 1:l]
        temp_charge = ph_dydx - ph_dxdy
        charge = (temp_charge > 6).astype(int) - (temp_charge < -6).astype(int) # Return to plus or minus one
        charge[:,-1] = 0
        charge[-1,:] = 0
        charge[:, 0] = 0
        charge[0, :] = 0
        charge = np.hstack([(np.vstack([charge, np.zeros(l-1)])), np.zeros((l,1))]) # Add rows and columns to fit the network

        return charge

    def __call__(self, vtime1, vtime0):
        ph, vtime0  = self.cal_phase(vtime1, vtime0)
        tcd = self.cal_density(ph)

        return tcd, vtime0


# dynamic learning of synchronization technique
class DLS:
    '''
    Yong Wu et al.
    (1) Dynamic learning of synchronization in nonlinear coupled systems.  
    doi.org/10.48550/arXiv.2401.11691
    (2) Dynamic modulation of external excitation enhance synchronization in complex neuronal network. 
    Chaos Soliton Fract. 2024; 183: 114896. doi.org/10.1016/j.chaos.2024.114896
    '''
    def __init__(self, N=1, local=[1, 2], alpha=0.01, dt=0.01):
        self.dt = dt
        self.num = N        # term number of the polynomial expansion
        self.local = local  # The location of the node to be trained
        self.alpha = alpha  # hyperparameter of learning rate
        self.P = np.full((len(local), N), self.alpha)   # Initialization of P matrix

    # Recursive least squares algorithm
    def forward(self, w, input, error):
        local_input = input[self.local]     # shape: (len(self.local), N)
        local_error = error[self.local]     # shape: (len(self.local),)

        # Calculate Prs (only need to multiply diagonal by input)
        Prs = self.P * local_input  # Direct element-by-element multiplication

        # Calculate a
        as_ = 1.0 / (1.0 + np.sum(local_input * Prs, axis=1))

        # Update Ps
        P_updates = as_[:, np.newaxis] * (Prs ** 2)
        self.P -= P_updates

        # Update weights
        delta_w = (as_ * local_error)[:, np.newaxis] * Prs
        np.add.at(w, self.local, -delta_w)

    def train(self, re_factor, factor, mem, self_y=None):
        """
        self_y Supervised sequence (Custom value) DLS driver the system synchronizes with this sequence
        rev_factor, Parameters that need to be updated: such as light & current
        factor, Multiply with rev_factor
        mem, The state variable at time t+1
        """
        # Inputs from external factors(self.num, self.Inum)
        input = factor*self.dt

        if self_y is not None:
            yMean  = self_y                 # Supervised learning
        else:
            yMean = mem[self.local].mean()  # Unsupervised learning
        
        error_y = mem - yMean               # Least squares to find the difference

        self.forward(re_factor, input, error_y)

    def reset(self):
        self.P = np.full((len(self.local), self.num), self.alpha)


# Synchronization factor
class SynFactor:
    def __init__(self, Tn, num):
        self.Tn = Tn    # Count steps
        self.n = num    # Number of nodes
        self.count = 0  # count
        # Initialize the calculation process
        self.up1 = 0
        self.up2 = 0
        self.down1 = np.zeros(num)
        self.down2 = np.zeros(num)

    def __call__(self, x):
        F = np.mean(x)
        self.up1 += F*F/self.Tn
        self.up2 += F/self.Tn
        self.down1 += x*x/self.Tn
        self.down2 += x/self.Tn
        self.count += 1     # Count stack

    def return_syn(self):
        if self.count != self.Tn:
            print(f"Required count:{self.Tn}, Actual count:{self.count}") 
        down = np.mean(self.down1-self.down2**2)
        if down>-0.000001 and down<0.000001:
            return 1.
        up = self.up1-self.up2**2

        return up/down
    
    def reset(self):
        self.__init__(self.Tn, self.n)


# AES & AOI array
class Electrode:
    '''
    Qianming Ding et al.
    (1) Adaptive electric shocks control and elimination of spiral waves using dynamic learning based techniques.
    (2) Elimination of reentry spiral waves using adaptive optogenetical illumination based on dynamic learning techniques.
    '''
    def __init__(self, grid_size):
        self.grid_size = grid_size

    # Circular region of action
    def circle(self, center, radius):
        grid_size = self.grid_size
        x_c = center // grid_size
        y_c = center % grid_size
        indices_in_circle = []
        # Traverse the node, whether it is inside the circle
        for i in range(grid_size):
            for j in range(grid_size):
                if (i - x_c) ** 2 + (j - y_c) ** 2 <= radius ** 2:
                    index = i * grid_size + j
                    indices_in_circle.append(index)
        return indices_in_circle

    # Arrays of electrodes or LEDs
    def array(self, Len):
        N = self.grid_size
        # Create an N by N matrix
        matrix = np.arange(N * N).reshape(N, N)
        # Divide the matrix into M*M blocks and expand
        reshaped_matrix = matrix.reshape(N // Len, Len, N // Len, Len).swapaxes(1, 2).reshape(-1, Len * Len)
        return reshaped_matrix