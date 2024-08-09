import numpy as np

# Channelrhodopsin-2
class CHR2:
    """
    Light sensitive channel
    Williams JC, Xu J, Lu Z, Klimas A, Chen X, Ambrosi CM, Cohen IS, Entcheva E. 
    Computational optogenetics: empirically-derived voltage- and light-sensitive Channelrhodopsin-2 model. 
    PLoS Comput. Biol. 2013; 9(9): e1003220. doi.org/10.1371/journal.pcbi.1003220
    """
    def __init__(self, node):
        self.node = node        # Inherited node class
        self.dt = self.node.dt  # time steps
        self.lambda_nm = 470    # Wave length               nm
        self.temperature = 37   # Body temperature          ℃
        self.irradiance = 0.0   # Llumination intensity     mW/mm^2
        self.initialize_params()        # Example Initialize the optical channel parameters
        self.update_temp_params(self.temperature)     # Initialize temperature-dependent parameters
        self.update_opto_params(self.irradiance)      # Initializes the light dependent parameters

    def initialize_params(self):
        self.C1 = np.ones_like(self.node.v)       # Initial value = 1.0
        self.C2 = np.zeros_like(self.node.v)      # Initial value = 0.0
        self.O1 = np.zeros_like(self.node.v)      # Initial value = 0.0
        self.O2 = np.zeros_like(self.node.v)      # Initial value = 0.0
        self.p =  np.zeros_like(self.node.v)      # Initial value = 0.0
        self.E_chr2 = 0.0       # ChR2 reversal potential
        self.gamma = 0.1        # Dark adaptation/light adaptation gated open probability ratio
        self.wloss = 1.3        # Light loss factor
        self.tauChR2 = 1.3      # ChR2 activation time constant
        self.gChR2 = 0.4        # Maximum conductance = 2.0 / 5

    def update_temp_params(self, temperature):
        # Original temperature dependent parameters
        self.original_params = {
            'Gd1': 0.075 + 0.043 * np.tanh((self.node.v + 20) / -20),   # Gating dynamic parameter 1
            'Gd2': 0.05,                                                # Gating dynamic parameter 2
            'epsilon1': 0.8535,                                         # Photon absorption efficiency 1
            'epsilon2': 0.14,                                           # Photon absorption efficiency 2
            'Gr': 0.0000434587 * np.exp(-0.0211539274 * self.node.v),   # Transition rate constant
            'e12_dark': 0.011,                                          # O1-O2 - ms^-1
            'e21_dark': 0.008                                           # O2-O1 - ms^-1
        }
        # Q10 temperature coefficients
        Q10 = {
            'Gd1': 1.97,
            'Gd2': 1.77,
            'epsilon1': 1.46,
            'epsilon2': 2.77,
            'Gr': 2.56,
            'e12_dark': 1.1,
            'e21_dark': 1.95
        }
        self.temperature = temperature
        temp_factor = (self.temperature - 22) / 10
        for params in self.original_params:         # Update temperature sensitive parameters
            setattr(self, params, self.original_params[params] * (Q10[params] ** temp_factor))

    def update_opto_params(self, irradiance):
        self.irradiance = irradiance                # Light intensity
        self.theta = 100 * self.irradiance
        self.e12 = self.e12_dark + 0.005 * np.log(1 + self.irradiance / 0.024)
        self.e21 = self.e21_dark + 0.004 * np.log(1 + self.irradiance / 0.024)
        self.F = 0.0006 * self.irradiance * self.lambda_nm / self.wloss     # Photon flux
        self.S = 0.5 * (1 + np.tanh(120 * (self.theta - 0.1)))              # photoadaptability

    def calc_opto_current(self):
        k1 = self.epsilon1 * self.F * self.p
        k2 = self.epsilon2 * self.F * self.p
        self.C1 += (self.Gr * self.C2 + self.Gd1 * self.O1 - k1 * self.C1) * self.dt
        self.C2 += (self.Gd2 * self.O2 - (k2 + self.Gr) * self.C2) * self.dt
        self.O1 += (k1 * self.C1 - (self.Gd1 + self.e12) * self.O1 + self.e21 * self.O2) * self.dt
        self.O2 += (k2 * self.C2 - (self.Gd2 + self.e21) * self.O2 + self.e12 * self.O1) * self.dt
        self.p  += ((self.S - self.p) / self.tauChR2) * self.dt

        G = (10.6408 - 14.6408 * np.exp(-self.node.v / 42.7671)) / self.node.v
        # Calculate the optical channel current
        Iopt = self.gChR2 * G * (self.O1 + self.gamma * self.O2) * (self.node.v - self.E_chr2)  
        return Iopt

    def __call__(self, irradiance):
        # The light intensity is positive
        irradiance = np.where(irradiance < 0.0, 0.0, irradiance)
        # Update illumination parameter
        self.update_opto_params(irradiance)
        # Return the photoinduced current
        return self.calc_opto_current()
    
    def reset(self):
        self.initialize_params()        # Example Initialize the optical channel parameters
        self.update_temp_params(37.)    # Reset the temperature-dependent parameters
        self.update_opto_params(0.0)    # Reset the light-dependence parameters
