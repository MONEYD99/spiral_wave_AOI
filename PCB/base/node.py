import numpy as np

# Luo-Rudy node
class LRnode:
    """
    Cardiac models, dimensional biophysical models.
    Reference: Luo CH, Rudy Y. 
    A Model of the Ventricular Cardiac Action Potential. 
    Circ. Res. 1991; 68(6): 1501-1526.  doi.org/10.1161/01.res.68.6.1501
    """
    def __init__(self, dt):
        self.dt = dt
        
        # Luo-Rudy parameters
        self.C = 1.0        # uF/cm^2
        self.Cao = 1.8      # mM

        # Channel conductance
        self.GNa = 16.0     # mS/cm^2
        self.Gsi = 0.02     # mS/cm^2
        self.GK = 0.423     # mS/cm^2
        self.GK1 = 0.6047   # mS/cm^2
        self.GKp = 0.0183   # mS/cm^2
        self.Gb = 0.03921   # mS/cm^2

        # Reversal potential
        self.ENa = 54.8     # mV
        # self.Esi = -82.3    # mV   钙离子反转电位动态变化
        self.EK = -77.6     # mV
        self.EK1 = -87.9    # mV
        self.EKp = -87.9    # mV
        self.Eb = -59.87    # mV

        # Initializes the state variable
        self.v   = None     # Membrane voltage
        self.m   = None     # Sodium activation gate
        self.h   = None     # Sodium inactivation gate
        self.j   = None     # Sodium slow inactivation gate
        self.d   = None     # Slow inward current activation gate
        self.f   = None     # Slow inward current inactivation gate
        self.X   = None     # Potassium activation gate
        self.Cai = None     # Intracellular calcium concentration

    def _initialize(self, init_LR):
        self.v   = init_LR[0].reshape(-1).astype(np.float64)
        self.m   = init_LR[1].reshape(-1).astype(np.float64)
        self.h   = init_LR[2].reshape(-1).astype(np.float64)
        self.j   = init_LR[3].reshape(-1).astype(np.float64)
        self.d   = init_LR[4].reshape(-1).astype(np.float64)
        self.f   = init_LR[5].reshape(-1).astype(np.float64)
        self.X   = init_LR[6].reshape(-1).astype(np.float64)
        self.Cai = init_LR[7].reshape(-1).astype(np.float64)

    def integral(self, inputs):
        """
        inputs: External currents
        """
        INa = self.calc_INa(self.v, self.m, self.h, self.j)
        Isi = self.calc_Isi(self.v, self.d, self.f, self.Cai)
        IK  = self.calc_IK(self.v, self.X)
        IK1 = self.calc_IK1(self.v)
        IKp = self.calc_IKp(self.v)
        Ib  = self.calc_Ib(self.v)

        I_ion = INa + Isi + IK + IK1 + IKp + Ib

        self.v += self.dt * (-(I_ion - inputs) / self.C)
        self.v = np.clip(self.v, -100, 20)  # Limit conditions to prevent errors

    def calc_INa(self, V, m, h, j):
        """
        Calculate fast sodium ion current
        """
        am = 0.32*( V + 47.13 ) / (1 - np.exp( -0.1 * ( V + 47.13 ) ) )
        bm = 0.08 * np.exp( -V / 11 )
        m_inf = am / (am + bm)
        m += (m_inf - m) * (am + bm) * self.dt  # Update m-gated variables

        ah = np.where(V < -40, # Voltage dependent ah
                      0.135 * np.exp( ( 80 + V ) / -6.8 ),
                      0.0)                      
        bh = np.where(V < -40, # Voltage dependent bh
                      3.56 * np.exp( 0.079 * V ) + 3.1 * 10**5 * np.exp( 0.35 * V ),
                      1 / ( 0.13 * ( 1 + np.exp( ( V + 10.66 ) / -11.1 ) ) ))   
        h_inf = ah / (ah + bh)
        h += (h_inf - h) * (ah + bh) * self.dt  # Update h-gated variables

        aj = np.where(V < -40, # Voltage dependent aj
                      ( -1.2714 * 10**5 * np.exp( 0.2444 * V) - 3.474 * 10**-5 * np.exp( -0.04391 * V))\
                    * ( V + 37.78 ) / ( 1 + np.exp( 0.311 * ( V + 79.23 ))),
                      0.0)                      
        bj = np.where(V < -40, # Voltage dependent bj
                      0.1212 * np.exp( -0.01052 * V) / ( 1 + np.exp( -0.1378 * ( V + 40.14 ))),
                      0.3 * np.exp( -2.535 * 10**-7 * V ) / ( 1 + np.exp( -0.1 * ( V + 32 )))) 
        j_inf = aj / (aj + bj)
        j += (j_inf - j) * (aj + bj) * self.dt  # Update j-gated variables

        INa = self.GNa * m**3 * h * j * (V - self.ENa)
        return INa
    
    def calc_Isi(self, V, d, f, Cai):
        """
        Calculate L-type calcium ion current
        """
        ad = 0.095 * np.exp( -0.01 * ( V - 5 ) ) / ( 1 + np.exp( -0.072 * ( V - 5 ) ) )
        bd = 0.07  * np.exp( -0.017 * ( V + 44 ) ) / ( 1 + np.exp( 0.05 * (V + 44 ) ) )
        d_inf = ad / (ad + bd)
        d += (d_inf - d) * (ad + bd) * self.dt  # Update d-gated variables
        
        af = 0.012 * np.exp( -0.008 * ( V + 28 ) ) / ( 1 + np.exp( 0.15 * ( V + 28 ) ) )
        bf = 0.0065 * np.exp( -0.02 * ( V + 30 ) ) / ( 1 + np.exp( -0.2 * ( V + 30 ) ) )
        f_inf = af / (af + bf)
        f += (f_inf - f) * (af + bf) * self.dt  # Update f-gated variables

        Esi = 7.7 - 13.0287 * np.log( Cai / self.Cao )  # Calculate the calcium reversal potential
        Isi = self.Gsi * d * f * (V - Esi)

        dCai = -10**-4 * Isi + 0.07 * ( 10**-4 - Cai )  # Updated Ca entry rate
        self.Cai += dCai * self.dt

        return Isi
    
    def calc_IK(self, V, X):
        """
        Calculate the time-dependent potassium ion current
        """
        aX = 0.0005 * np.exp(0.083 * (V + 50)) / (1 + np.exp(0.057 * (V + 50)))
        bX = 0.0013 * np.exp(-0.06 * (V + 20)) / (1 + np.exp(-0.04 * (V + 20)))
        X_inf = aX / (aX + bX)
        X += (X_inf - X) * (aX + bX) * self.dt  # Update the X-gated variable
        Xi = np.where(V < -100,                 # Voltage dependent Xi
                      2.837 * (np.exp(0.04 * (V + 77)) - 1) / ((V + 77) * np.exp(0.04 * (V + 35))),
                      1.0)
        
        IK = self.GK * X * Xi * (V - self.EK)   # Calculate the potassium ion current
        return IK

    def calc_IK1(self, V):
        """
        Calculate the time-invariant potassium ion current
        """
        aK1 = 1.02 / (1 + np.exp(0.2385 * (V - self.EK1 - 59.215)))
        bK1 = (0.49124 * np.exp(0.08032 * (V - self.EK1 + 5.476)) + \
               np.exp(0.06175 * (V - self.EK1 - 594.31))) /         \
              (1 + np.exp(-0.5143 * (V - self.EK1 + 4.753)))
        K1_inf = aK1 / (aK1 + bK1)
        IK1 = self.GK1 * K1_inf * (V - self.EK1)
        return IK1

    def calc_IKp(self, V):
        """
        Calculate platform potassium ion current
        """
        Kp = 1 / (1 + np.exp((7.488 - V) / 5.98))
        IKp = self.GKp * Kp * (V - self.EKp)
        return IKp

    def calc_Ib(self, V):
        """
        Calculate the background ion current
        """
        Ib = self.Gb * (V - self.Eb)
        return Ib

    def reset(self):
        """
        Reset model
        """
        self.v = None
        self.m = None
        self.h = None
        self.j = None
        self.d = None
        self.f = None
        self.X = None
        self.Cai = None

# Fenton-Karma node
class FKnode:
    """
    Dimensionless generalized cardiac model
    reference: Fenton FH, Karma A. 
    Vortex dynamics in three-dimensional continuous myocardium with fiber rotation: flament instability and fbrillation.
    Chaos 1998; 8: 20-47.  doi.org/10.1063/1.166311
    """
    def __init__(self, dt):
        self.dt = dt

        # FK model parameters
        self.v_c = 0.13
        self.v_v = 0.04
        self.tau_fi = 0.395
        self.tau_so1 = 9
        self.tau_so2 = 33.3
        self.tau_si = 29
        self.k = 15
        self.v_csi = 0.50

        # initializes the state variable
        self.v = None
        self.u = None
        self.w = None

    def _initialize(self, vuw):
        self.v = vuw[0].reshape(-1).astype(np.float64)
        self.u = vuw[1].reshape(-1).astype(np.float64)
        self.w = vuw[2].reshape(-1).astype(np.float64)

    def integral(self, inputs):
        """
        inputs: External currents
        """
        p1 = np.where(self.v <= self.v_c, 0.0, 1.0)
        p2 = np.where(self.v >= self.v_c, 0.0, 1.0)

        J_fi = -self.u * p1 * (1 - self.v) * (self.v - self.v_c) / self.tau_fi
        J_so = self.v * p2 / self.tau_so1 + p1 / self.tau_so2
        J_si = -self.w * (1 + np.tanh(self.k * (self.v - self.v_csi))) / (2 * self.tau_si)

        self.v += self.dt * -(J_fi + J_so + J_si - inputs)
        self.u += self.dt * self.u_fk()
        self.w += self.dt * self.w_fk()

    def u_fk(self):
        p1 = np.where(self.v <= self.v_c, 0.0, 1.0)
        p2 = np.where(self.v >= self.v_c, 0.0, 1.0)
        q1 = np.where(self.v <= self.v_v, 0.0, 1.0)
        q2 = np.where(self.v >= self.v_v, 0.0, 1.0)
        tau_v_minus = q1 * 9 + q2 * 8
        tau_v_plus = 3.33
        return (p2 * (1 - self.u) / tau_v_minus) - (p1 * self.u / tau_v_plus)

    def w_fk(self):
        p1 = np.where(self.v <= self.v_c, 0.0, 1.0)
        p2 = np.where(self.v >= self.v_c, 0.0, 1.0)
        tau_w_minus = 60
        tau_w_plus = 250
        return (p2 * (1 - self.w) / tau_w_minus) - (p1 * self.w / tau_w_plus)

    def reset(self):
        """
        Reset model
        """
        self.v = None
        self.u = None
        self.w = None

# Hodgkin–Huxley node
class HHnode:
    """
    Mammal Hodgkin-Huxley-type model
    reference: Yuguo Yu, Adam P Hill, David A. McCormick. 
    Warm Body Temperature Facilitates Energy Efficient Cortical Action Potentials.
    PLoS Comput Biol. 2012; 8(4): e1002456. doi.org/10.1371/journal.pcbi.1002456.
    """
    def __init__(self, dt):
        self.dt = dt

        # HH model parameters
        self.g_Na = 150.0
        self.g_K = 40.0
        self.g_l = 0.033
        self.V_Na = 60.0
        self.V_K = -90.0
        self.V_l = -70.0
        self.C = 0.75

        # initializes the state variable
        self.v = None
        self.m = None
        self.n = None
        self.h = None

    def _initialize(self, vmnh):
        self.v = vmnh[0].reshape(-1).astype(np.float64)
        self.m = vmnh[1].reshape(-1).astype(np.float64)
        self.n = vmnh[2].reshape(-1).astype(np.float64)
        self.h = vmnh[3].reshape(-1).astype(np.float64)

    def integral(self, inputs):
        """
        inputs: External currents
        """
        I_Na = self.g_Na * self.m**3 * self.h * (self.v - self.V_Na)
        I_K = self.g_K * self.n * (self.v - self.V_K)
        I_l = self.g_l * (self.v - self.V_l)

        # Update ion channel states using gating variables' equations
        dm = self._m_gate(self.v, self.m)
        dn = self._n_gate(self.v, self.n)
        dh = self._h_gate(self.v, self.h)

        # Update membrane potential
        self.v += self.dt * (inputs - I_Na - I_K - I_l) / self.C
        self.m += self.dt * dm
        self.n += self.dt * dn
        self.h += self.dt * dh
    
    def _m_gate(self, v, m):
        a_m = 0.182 * (v + 30.0) / (1 - np.exp(-(v + 30.0) / 8.0) + 1e-6)
        b_m = -0.124 * (v + 30.0) / (1 - np.exp((v + 30.0) / 8.0) + 1e-6)
        return a_m * (1 - m) - b_m * m

    def _n_gate(self, v, n):
        a_n = 0.01 * (v - 30.0) / (1 - np.exp(-(v - 30.0) / 9.0) + 1e-6)
        b_n = -0.002 * (v - 30.0) / (1 - np.exp((v - 30.0) / 9.0) + 1e-6)
        return a_n * (1 - n) - b_n * n

    def _h_gate(self, v, h):
        a_h = 0.028 * (v + 45.0) / (1 - np.exp(-(v + 45.0) / 6.0) + 1e-6)
        b_h = -0.0091 * (v + 70.0) / (1 - np.exp((v + 70.0) / 6.0) + 1e-6)
        return (-h + 1 / (1 + np.exp((v + 60.0) / 6.2))) * (a_h + b_h)

    def reset(self):
        """
        Reset model
        """
        self.v = None
        self.m = None
        self.n = None
        self.h = None

# FitzHugh-Nagumo node
class FHNnode:
    """
    Dimensionless neural models
    Fitzhugh R. 
    Impulses and Physiological States in Theoretical Models of Nerve Membrane. 
    Biophys J. 1961; 1(6): 445-466. doi.org/10.1016/s0006-3495(61)86902-6.
    """
    def __init__(self, dt):
        self.dt = dt  # Time step for the simulation
        self.a = 0.08  # 
        self.b = 0.8  # 
        self.c = 0.7  # 

        # Initialize state variables
        self.v = None  # Membrane potential
        self.w = None  # Recovery variable

    def _initialize(self, vw):
        """
        Initialize the state variables of the FHN model.
        :param v_init: Initial membrane potential
        :param w_init: Initial recovery variable
        """
        self.v = vw[0].reshape(-1).astype(np.float64)
        self.w = vw[1].reshape(-1).astype(np.float64)

    def integral(self, inputs):
        """
        Update the state of the FHN neuron model based on the current input and model parameters.
        :param I: Input current
        """
        # FHN model equations
        v_next = self.v + self.dt * (self.v - self.v ** 3 / 3 - self.w + inputs)
        w_next = self.w + self.dt * self.a * (self.v + self.c - self.b * self.w)

        self.v = v_next
        self.w = w_next

    def reset(self):
        """
        Reset the neuron's state variables to None.
        """
        self.v = None
        self.w = None