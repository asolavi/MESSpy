import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from scipy.integrate import solve_ivp
import os
import sys 
import pickle
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir)))  
from core import constants as c
from scipy.optimize import newton, brentq, curve_fit
from scipy.interpolate import UnivariateSpline
import math


def cracker_reactor(NH3_massflowrate, T_i, p, n, reactor_volume, thermal_mode, constant_Q = False):
    """
    The function models the chemical reaction process for ammonia cracking.
    It calculates the changes in the molar flow rates of nitrogen, hydrogen, and ammonia, as well as
    the temperature, the NH3 conversion and the heat needed as a function of the reactor volume.

    Inputs:
        NH3_massflowrate: mass flow rate of NH3 at the cracker inlet [kg/s]
        T_i: initial temperature [K]
        p: reactor operating pressure [bar]
        n: number of points for the solver
        reactor_volume: volume of the reactor [m^3]
        
    Outputs:
        N2_molarflowrate_bed: molar flow rate of nitrogen inside the reactor [mol/s]
        H2_molarflowrate_bed:  molar flow rate of hydrogen inside the reactor [mol/s]
        NH3_molarflowrate_bed:  molar flow rate of ammonia inside the reactor [mol/s]
        T_bed: temperature inside the reactor [K]
        XNH3_bed: ammonia conversion inside the reactor [-]
        Q_bed: heat inside the reactor [kW]
    """
    
    def reactor_ode(V, y, mix_massflowrate, NH3_molarflowrate0):
        N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XNH3, Q = y  
        
        mix_molarflowrate = N2_molarflowrate + H2_molarflowrate + NH3_molarflowrate
        x_N2 = N2_molarflowrate / mix_molarflowrate
        x_H2 = H2_molarflowrate / mix_molarflowrate
        x_NH3 = NH3_molarflowrate / mix_molarflowrate

        # Resolution of the reaction kinetics at 1 bar # https://doi.org/10.3390/molecules28166006, assumption of ideal mixture
        "NH₃ --> 1/2N₂ + 3/2H₂"
        p_NH3 = p * x_NH3 * 100000
        p_H2 = p * x_H2 * 100000
        p_H2 = max(p_H2, 1e-20)
        
        a = -5.996
        b = 4.344 * 10**4
        c = -2.610 * 10**7
        k = np.exp(a + b / T + c / T**2)   # [mol*Pa/(s*m^2)]
        a = -6.181
        b = 2.849 * 10**4
        c = -1.287 * 10**7
        K = np.exp(a + b / T + c / T**2)   # [Pa^0.5]

        # Ammonia decomposition rate - Ni-based catalyst
        av = 2.19*10**5              # [m^2_cat / m^3_react]
        r = k * p_NH3**2 / (p_H2**1.5 + K * p_NH3)**2 * av     # ammonia decomposition rate [mol_NH3/(s*m^3)]

        # Molar flowrate variations for m^3 
        dN2_molarflowrate_dV = r / 2     # [mol/(s*m^3)], r/2 because every mole of NH3 decomposed, half of N2 is produced
        dH2_molarflowrate_dV = 3/2 * r   # [mol/(s*m^3)], 3/2r because every mole of NH3 decomposed, 3/2 of H2 is produced
        dNH3_molarflowrate_dV = -r       # [mol/(s*m^3)]
        # Ammonia conversion variation for m^3 (XNH3 = NH3_cracked / NH3_in)
        dXNH3_dV = r / NH3_molarflowrate0  # [1/m^3], r = ammonia decomposed 
        # Temperature variation for m^3
        cp_H2 = 29.66
        cp_N2 = 31.42
        cp_NH3 = 0.02815 * T + 28.64
        heat_of_reaction = 45900 + (1.5 * cp_H2 + 0.5 * cp_N2 - cp_NH3) * (T - 298)    
        if thermal_mode == 'adiabatic':
            dT_dV = -heat_of_reaction * r / (H2_molarflowrate * cp_H2 + N2_molarflowrate * cp_N2 + NH3_molarflowrate * cp_NH3) # [K/m^3]
            dQ_dV = 0  # adiabatic cracker
        elif thermal_mode == 'constant Q':
            dQ_dV = constant_Q / reactor_volume  # [kW/m^3]
            dT_dV = -heat_of_reaction * r / (H2_molarflowrate * cp_H2 + N2_molarflowrate * cp_N2 + NH3_molarflowrate * cp_NH3) + dQ_dV * 1000 / (H2_molarflowrate * cp_H2 + N2_molarflowrate * cp_N2 + NH3_molarflowrate * cp_NH3) # [K/m^3]
        elif thermal_mode == 'isothermal':
            dT_dV = 0  # isothermal, all the heat absorbed by the cracking is istantaneously furnished by the heat source
            dQ_dV = heat_of_reaction * r / 1000 # [kW/m^3]

        return [dN2_molarflowrate_dV, dH2_molarflowrate_dV, dNH3_molarflowrate_dV, dT_dV, dXNH3_dV, dQ_dV]
    
    # Inputs for the solution of the ODE problem
    N2_molarflowrate_i = 0
    H2_molarflowrate_i = 0
    NH3_molarflowrate_i = NH3_massflowrate / c.NH3MOLMASS     # [mol/s]
    NH3_molarflowrate0 = NH3_molarflowrate_i
    XNH3_i = 0
    Q_i = 0
    y0 = [N2_molarflowrate_i, H2_molarflowrate_i, NH3_molarflowrate_i, T_i, XNH3_i, Q_i]  # initial values 
    V_span = [0, reactor_volume]                                                          # interval of integration
    # Solution using BDF method
    mix_massflowrate = NH3_massflowrate
    sol = solve_ivp(reactor_ode, V_span, y0, method='BDF', args=(mix_massflowrate, NH3_molarflowrate0), t_eval=np.linspace(0, reactor_volume, n))
    N2_molarflowrate_bed = sol.y[0]
    H2_molarflowrate_bed = sol.y[1]
    NH3_molarflowrate_bed = sol.y[2]
    T_bed = sol.y[3]
    XNH3_bed = sol.y[4]
    Q_bed = sol.y[5]
        
    return T_bed, N2_molarflowrate_bed, H2_molarflowrate_bed, NH3_molarflowrate_bed, XNH3_bed, Q_bed


def profile_RHE(Hotfluid, n, T_hot_in, p_hot, T_cold_in, T_cold_out, p_cold, massflowrate):
    """
    The function models the feed-effluent heat exchanger in the design of the plant - 
    check that there is no reversal of the heat exchange flow. This is achieved by 
    discretizing the exchanger and progressively calculating the heat exchange and 
    temperature. It returns also some design parameters caalculated with the epsilon-NTU 
    method.
    
    """

    h_cold_in = PropsSI('H','P', p_cold * 100000,'T', T_cold_in,'Ammonia')	# enthalpy cold fluid HE inlet
    h_cold_out = PropsSI('H','P', p_cold * 100000,'T', T_cold_out,'Ammonia')	   # enthalpy cold fluid HE outlet
    Q = massflowrate * (h_cold_out - h_cold_in)
    
    Hotfluid.update(CP.PT_INPUTS, p_hot*100000, T_hot_in)
    h_hot_in = Hotfluid.hmass() 	   # enthalpy hot fluid HE inlet
     
    DQ = Q / (n-1)	             # discretization of Q
    dh_hot = DQ / massflowrate	# delta_h hot side
    dh_cold = DQ / massflowrate  # delta_h cold side
    
    # Initialization
    T_hot = np.zeros(n)
    h_hot = np.zeros(n)
    T_cold = np.zeros(n)
    h_cold = np.zeros(n)
    deltaT = np.ones(n)* (T_hot_in - T_cold_in)
    Q_vals = np.zeros(n)
    T_hot[0] = T_hot_in	
    h_hot[0] = h_hot_in	
    T_cold[0] = T_cold_in	
    h_cold[0] = h_cold_in 
    Q_HE = 0
    
    for i in range(1,n):
       # Hot fluid
       h_hot[i] = h_hot[i-1]-dh_hot
       T_hot[i] = T_ph(Hotfluid, p_hot, T_hot[i-1], h_hot[i]) 
       
       # Cold fluid
       h_cold[i] = h_cold[i-1]+dh_cold
       T_cold[i] = PropsSI('T','P', p_cold * 100000,'H', h_cold[i],'Ammonia')
       # Calculate the heat exchanged incrementally
       Q_HE += DQ
       Q_vals[i] = Q_HE
       
       
    # Calculate the temperature difference 
    for i in range(0,n):
        deltaT[i] = T_hot[i] - T_cold[n-i-1]
    pinch_point = min(deltaT)
    if deltaT[i] <= 0:
        return 0, 0, 0, 0, 0, 0, pinch_point

    cp_cold = (h_cold_out - h_cold_in) / (T_cold_out - T_cold_in)
    h_hot_out = h_hot[-1]
    T_hot_out = T_hot[-1]
    cp_hot = (h_hot_in - h_hot_out) / (T_hot_in - T_hot_out)
    C_min = massflowrate * min(cp_hot, cp_cold)
    C_max = massflowrate * max(cp_hot, cp_cold)
    epsilon = Q / (C_min * (T_hot_in - T_cold_in))
    
    # Epsilon-NTU calculation - necessary for off-design operation
    C = C_min / C_max
    U = 60   # [W / (m^2 * K)] https://doi.org/10.1016/j.ijhydene.2024.05.308
    # Counterflow heat exchanger
    NTU = 1 / (C - 1) * np.log((epsilon - 1) / (epsilon * C - 1))
    A = NTU * C_min / U

    return T_hot, T_cold, Q_vals, A, U, epsilon, pinch_point


def T_ph(AS, p, T, h_ref):       
    """
    The function iterates to find the temperature given the pressure p and the enthalpy h_ref, using the secant method.
    
    Inputs:
        AS: thermodinamic state 
        p: pressure [bar]
        T: initial temperature [K]
        h_ref: reference enthalpy [J/kg]

    Output:
        T_found: temperature found [K]
    """
    p=p*100000
    
    def h(T):
        AS.update(CP.PT_INPUTS, p, T)
        return AS.hmass()
    
    def f_T(T):
        return h(T) - h_ref
    
    T_found = newton(f_T, x0=T, tol=1e-2)
    
    return T_found


def RHE_plot(T_hot, T_cold, Q):
    
    plt.figure(dpi = 400)    
    plt.plot(Q/1000, T_hot, label="Hot Fluid", color="red")
    plt.plot(Q/1000, T_cold[::-1], label="Cold Fluid", color="blue")
    mid_index = len(Q) // 2
    dT_hot_dQ = np.gradient(T_hot, Q)  
    dT_cold_dQ = np.gradient(T_cold[::-1], Q)  
    dx_hot = Q[mid_index+1] - Q[mid_index] 
    dy_hot = dT_hot_dQ[mid_index] * dx_hot  
    dx_cold = Q[mid_index+1] - Q[mid_index] 
    dy_cold = dT_cold_dQ[mid_index] * dx_cold  
    plt.arrow(Q[mid_index]/1000, T_hot[mid_index], dx_hot/1000, dy_hot, 
      head_width=4, head_length=8, fc='red', ec='red')
    plt.arrow(Q[mid_index]/1000, T_cold[::-1][mid_index], -dx_cold/1000, -dy_cold, 
              head_width=4, head_length=8, fc='blue', ec='blue')
    plt.xlabel("Heat exchanged [kW]", fontsize=12)
    plt.ylabel("Temperature [K]", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


class cracker: 
    def __init__(self, parameters, T_in, p_in, location_name, path, file_structure, file_general, H2_FC = False, Npower_CCGT = False, eff_CCGT = False, timestep=False):
        """
        Create a general NH3 cracker object.
        
        Inputs:
            parameters: dictionary
                'Treact': reaction temperature [K]
                'reactor conversion': ammonia moles cracked / ammonia moles entering the cracker [-]
                'Nflowrate': nominal hydrogen flowrate produced by the cracker [kg/s]
                'number of modules': number of cracker modules, each sized at the nominal flowrate
                                    If not specified, it is computed based on the downstream fuel cell or CCGT size [-]
                'thermal mode': one of 'adiabatic', 'isothermal', 'constant Q'; defines the thermal behavior of the reactor
                'Tout': required only if 'thermal mode' is 'constant Q', defines the outlet temperature [K]
                'heat source': heating source used if heat is required; one of 'electric', 'ammonia combustion'
            T_in: temperature of ammonia entering the reactor [K]
            p_in: pressure of ammonia entering the reactor [bar]
                    
        Outputs: cracker object able to:
            consume ammonia and electricity and produce hydrogen .use(hydrogen, ammonia) 
    
        """
        # For module_consumption_simple function
        # If the spline fit has already been saved as file.pkl for the same cracker, this file is used, otherwise is calculated
        
        check = True # True if no parameters are changed from the old simulation
            
        # Directory for storing previous simulation data
        directory = './consumption'

        # Checking if the previous simulation exists
        if os.path.exists(path + f"{directory}/cracker_{file_structure}_{location_name}.pkl"):
            with open(path + f"{directory}/cracker_{file_structure}_{location_name}.pkl", 'rb') as f:
                ps_parameters = pickle.load(f)  # Load previous simulation parameters
                par_to_check = ['Treact', 'reactor conversion', 'Nflowrate', 'thermal mode', 'heat source']
                if ps_parameters["parameters"].get('thermal mode') == 'constant Q':
                    par_to_check.append('Tout')
                for par in par_to_check:
                    if par in ps_parameters["parameters"] and par in parameters:  # Some parameters haven't to be defined
                        if ps_parameters["parameters"][par] != parameters[par]:
                            check = False
                if T_in != ps_parameters["additional_data"]['T_in']:
                    check = False
                if p_in != ps_parameters["additional_data"]['p_in']:
                    check = False                  

        else:
            check = False
            
        if check == False:
            for file_name in os.listdir(os.path.join(path, 'consumption/calculations')):
                if 'cracker' in file_name:  
                    file_path = os.path.join(path, 'consumption/calculations', file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        name_serie = f"cracker_{location_name}_{file_general}_{file_structure}.pkl"
        if check and os.path.exists(path + '/consumption/calculations/' + name_serie): 
            with open(path + '/consumption/calculations/' + name_serie, 'rb') as f:
              splines = pickle.load(f)
              self.spline_electricity = splines["spline_electricity"]
              self.spline_ammonia = splines["spline_ammonia"]
              self.spline_electricity_NH3 = splines["spline_electricity_NH3"]
              self.spline_hydrogen_NH3 = splines["spline_hydrogen_NH3"]
        else:
            check = False
        
        self.check = check
        self.path = path
        self.name_serie = name_serie
        self.location_name = location_name
        self.file_structure = file_structure
        self.file_general = file_general
        self.directory = directory
        self.parameters = parameters
        
                                                 
        self.T_react = parameters['Treact']                                 # [K] reaction temperature 
        self.p_react = 1                                                    # [bar] reaction pressure 
        self.NH3_conversion = parameters['reactor conversion']              # [-] moles of ammonia cracked / total moles of ammonia entering
        self.PSA_recovery_rate = 0.79                                       # [-] H2 out / H2 in https://doi.org/10.1016/j.cej.2024.151875
        self.H2_massflowrate = parameters['Nflowrate']                      # [kg/s] hydrogen flow rate required at the outlet of the entire system (single module = cracker + adsorber + PSA)
        self.H2_massflowrate_reactor = self.H2_massflowrate / self.PSA_recovery_rate        # [kg/s] gross hydrogen flow rate that must be produced by the cracker module
        self.n_modules = parameters['number of modules']                    # [-] number of cracker modules
        if H2_FC != False:     # the number of cracker modules (total H2 flow rate) is the maximum acceptable by the FC if is the only component downstream, obtainable from the FC nominal power 
            self.n_modules = math.ceil(H2_FC / self.H2_massflowrate)                                  
            # print(f"The cracker will be handled using {self.n_modules} modules.")
        if Npower_CCGT != False:          # the nominal flowrate of hydrogen is the maximum acceptable by the CCGT if is the only component downstream, obtainable from the CCGT nominal power 
            self.n_modules = math.ceil((Npower_CCGT /(eff_CCGT * c.LHV_H2) / 3600) / self.H2_massflowrate)                     # [-] (kW/kWh/kg_H2/3600) / [kg_H2/s]    
            # print(f"The cracker will be handled using {self.n_modules} modules.")
        self.min_load = 0.3                                                 # hypotesized module minimum load = 30%, as the ammonia synthesis reactor
        self.thermal_mode = parameters['thermal mode']
        self.Tout = parameters.get('Tout', False)
        if self.thermal_mode == 'constant Q':
            if self.Tout == False:
                raise ValueError("Missing required parameter: 'Tout'.\n"
            "When 'thermal mode' is set to 'constant Q', a value for 'Tout' "
            "(the temperature at the end of the cracking) must be provided in the parameter dictionary.")
        
        self.heat_source = parameters['heat source']
        self.eta_combustion = 0.75                             # used if heat source = ammonia combustion
        self.calculation_mode = parameters.get('calculation mode', None)
        self.timestep = timestep if timestep else c.timestep  # [min] simulation timestep
        
        # Creation of the thermodinamic state for the mixture N2/H2/NH3 with Peng-Robinson cubic EOS        
        self.mix_H2N2NH3 = CP.AbstractState('PR',"Nitrogen&Hydrogen&Ammonia") 
        self.mix_H2N2NH3.set_binary_interaction_double(0,1,"kij",-0.036)   # from UNISIM
        self.mix_H2N2NH3.set_binary_interaction_double(0,2,"kij",0.222)    # from UNISIM
        self.mix_H2N2NH3.set_binary_interaction_double(1,2,"kij",0.000)    # from UNISIM
        self.mix_H2N2NH3.specify_phase(CP.iphase_gas)
        
        # Initialization of the thermodynamic points (calculations for the single module)
        self.p_points = np.zeros(10)  # [bar]
        self.T_points = np.zeros(10)  # [K]
        self.h_points = np.zeros(10)  # [J/kg]
        self.m_points = np.zeros(10)  # [kg/s]
        self.points=[]                # name of the point

        
            
        # Ammonia module cracker calculations (Ni-based catalyst)
        def Q_required_solver(m_NH3, V):  # helper to solve for constant Q required to reach target Tout if thermal mode = constant Q
            def Q_func(Q_guess):
                T, *_ = cracker_reactor(m_NH3, self.T_react, self.p_react, 50, V, self.thermal_mode, Q_guess)
                return T[-1] - self.Tout  
            return brentq(Q_func, 0, 1e10, xtol=1e-2)

        # NH3 conversion (and hydrogen flowrate) specified → compute required NH3 feed and reactor volume
        self.NH3_massflowrate_reactor = (2 / 3 * self.H2_massflowrate_reactor / c.H2MOLMASS / self.NH3_conversion * c.NH3MOLMASS)
        def reactor_dimensioning(V):   # Function to find reactor volume that achieves target NH3 conversion
            Q = Q_required_solver(self.NH3_massflowrate_reactor, V) if self.thermal_mode == 'constant Q' else False
            T, _, _, _, XNH3, _ = cracker_reactor(self.NH3_massflowrate_reactor, self.T_react, self.p_react, 50, V, self.thermal_mode, Q)
            return XNH3[-1] - self.NH3_conversion
        # Solve for required reactor volume
        self.reactor_volume = brentq(reactor_dimensioning, 0.000001, 1000, xtol=1e-5, maxiter=100)


            
        # Final reactor run at solved conditions
        self.Q_required = Q_required_solver(self.NH3_massflowrate_reactor, self.reactor_volume) if self.thermal_mode == 'constant Q' else False
        self.T_cracker, self.N2_molarflowrate_cracker, self.H2_molarflowrate_cracker, self.NH3_molarflowrate_cracker, self.XNH3, self.Q_cracker = cracker_reactor(self.NH3_massflowrate_reactor, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, self.Q_required)
        self.Q_cracker_design = self.Q_cracker[-1]
            
            
            
        # Inlet conditions of NH3
        self.T_in = T_in  # [K] inlet temperature (from tank)
        self.p_in = p_in  # [bar] inlet pressure (from tank)
        i = 0
        self.points.append('NH3 inlet')
        self.T_points[i] = self.T_in
        self.p_points[i] = self.p_in
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        self.m_points[i] = self.NH3_massflowrate_reactor
        
        
        
        # NH3 lamination
        i += 1
        self.points.append('NH3 lamination')
        self.p_points[i] = self.p_react
        self.m_points[i] = self.m_points[i-1]
        self.h_points[i] = self.h_points[i-1]
        self.T_points[i] = PropsSI('T','P', self.p_points[i] * 100000,'H', self.h_points[i-1],'Ammonia')
        
        
        
        # NH3 heating with ambient water
        i += 1
        self.points.append('NH3 heating - ambient water')
        self.p_points[i] = self.p_points[i-1]
        self.m_points[i] = self.m_points[i-1]
        if self.T_points[i-1] < (c.AMBTEMP - 10):
            self.T_points[i] = c.AMBTEMP - 10    # fluid temperature after heating with water at ambient temperature, DT = 10 K
        else:
            self.T_points[i] = self.T_points[i-1]
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        
        self.cp_cold_design = PropsSI('C','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        self.mu_cold_design = PropsSI('viscosity','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia') 
        self.k_cold_design = PropsSI('conductivity','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        
        
        
        #Cracker inlet
        i += 1
        self.points.append('Cracker inlet')
        self.T_points[i] = self.T_react
        self.p_points[i] = self.p_points[i-1]
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T',self.T_points[i],'Ammonia')
        self.m_points[i] = self.m_points[i-1]
        h_reactor_in = self.h_points[i]
        
        
        
        # Cracker outlet
        i += 1
        self.points.append('Cracker outlet')
        self.T_points[i] = self.T_cracker[-1]
        self.p_points[i] = self.p_points[i-1]
        molarfractions = [self.N2_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1]), self.H2_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1]), self.NH3_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1])]
        self.mix_H2N2NH3.set_mole_fractions(molarfractions)
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i] * 100000, self.T_points[i])
        self.h_points[i] = self.mix_H2N2NH3.hmass()
        self.m_points[i] = self.m_points[i-1]
        
        if self.heat_source == 'electric':
            self.electric_heating_cracker = self.Q_cracker[-1]
            self.NH3_massflowrate_furnace = 0
        if self.heat_source == 'ammonia combustion':
            self.NH3_massflowrate_furnace = self.Q_cracker[-1] / (self.eta_combustion * c.LHV_NH3) / 3600   # [kg/s]
            self.electric_heating_cracker = 0
        
        components = ["N2", "H2", "NH3"]
        self.cp_hot_design = self.mix_H2N2NH3.cpmass()
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molarfractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molarfractions[i] * 100000, components[i]) for i in range(len(components))]
        self.mu_hot_design = 1 / sum([molarfractions[i] / viscosities[i] for i in range(len(components))]) # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        self.k_hot_design = np.dot(molarfractions, conductivities)  # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        
        # RHE  
        # Calculation of the heat that can be recovered from the cooling of the reactor products
        check_pinch_point = False
        delta_T = 10
        while check_pinch_point == False:
          T_hot_RHE, T_cold_RHE, Q_RHE, self.A_RHE, self.U_RHE_design, self.epsilon_RHE, pinch_point = profile_RHE(self.mix_H2N2NH3, 50, self.T_points[i], self.p_points[i], self.T_points[i-2], self.T_points[i]-delta_T, self.p_points[i-2], self.m_points[i])
          if pinch_point < 20:   # minimum pinch point imposed = 20 K
              delta_T += 5
          else:
              check_pinch_point = True
        # RHE_plot(T_hot_RHE, T_cold_RHE, Q_RHE)
        if T_cold_RHE[-1] >= self.T_react:
           self.electric_heating_cracker_inlet = 0 # the heat is all taken from the RHE 
        else:
            h_cold_out_RHE = PropsSI('H','P', self.p_react * 100000,'T', T_cold_RHE[-1],'Ammonia')
            self.reactor_in_specific_heat=(h_reactor_in-h_cold_out_RHE)/1000   # [kJ/kg]
            self.electric_heating_cracker_inlet = self.reactor_in_specific_heat * self.m_points[i]
    
         
      
        # NH3 removal unit (adsorber) and PSA electrical consumption ?
        
        self.electricity_design = (self.electric_heating_cracker + self.electric_heating_cracker_inlet) * self.n_modules
        self.NH3_consumed_design = (self.NH3_massflowrate_reactor + self.NH3_massflowrate_furnace) * self.n_modules
        

    def module_consumption(self, hydrogen = False, ammonia = False):
        """
        The consumption function calculates the module energy consumption in 
        the operating range and the ammonia needs for the hydrogen required. 
            
        Inputs:
            hydrogen: total hydrogen mass flowrate to be cracked from NH3 [kg/s]
            ammonia: total ammonia needed [kg/s]
            
        Outputs:
            electricity: electricity consumption [kW]  
            ammonia: ammonia consumed [kg/s]
            hydrogen: hydrogen produced [kg/s]
        """
        
        if hydrogen is not False and ammonia is not False:
            raise ValueError("Ambiguous input: specify either 'hydrogen' (target H₂ output) or 'ammonia' (available NH₃ input), not both.")
        
        # Initialization of the thermodynamic points (calculations for the single module)
        self.p_points = np.zeros(10)  # [bar]
        self.T_points = np.zeros(10)  # [K]
        self.h_points = np.zeros(10)  # [J/kg]
        self.m_points = np.zeros(10)  # [kg/s]
        self.points=[]   # name of the point

        
            
        # Ammonia module cracker calculations (Ni-based catalyst)
        # Either hydrogen (target output) or ammonia (input) must be specified
        
        # Case 1: Thermal mode is 'constant Q' → NH3 conversion fixed
        if self.thermal_mode == 'constant Q':     
        
            def Q_func(Q_guess):              
                _, _, _, _, XNH3, _ = cracker_reactor(reactor_ammonia, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, Q_guess)
                return XNH3[-1] - self.NH3_conversion  
            
            if hydrogen:
                # Case: hydrogen target is known
                reactor_hydrogen = hydrogen / self.PSA_recovery_rate
                reactor_ammonia = 2 / 3 * reactor_hydrogen / c.H2MOLMASS / self.NH3_conversion * c.NH3MOLMASS
                self.Q_required = brentq(Q_func, 0, self.Q_cracker_design*1.5, xtol=1e-2)
            
            elif ammonia and self.heat_source == 'electric':
                # Case: ammonia feed is fixed, heat provided electrically
                reactor_ammonia = ammonia
                self.Q_required = brentq(Q_func, 0, self.Q_cracker_design*1.5, xtol=1e-2)
                
            elif ammonia and self.heat_source == 'ammonia combustion':
                # Case: ammonia feed is fixed, heat provided by ammonia combustion
                def NH3_func(NH3_guess, Q_guess):
                    NH3_furnace = Q_guess / (self.eta_combustion * c.LHV_NH3) / 3600
                    return NH3_guess + NH3_furnace - ammonia
            
                def Q_func_combustion(Q_guess):
                    f_a = NH3_func(0, Q_guess)
                    f_b = NH3_func(ammonia, Q_guess)
                    if f_a * f_b > 0:
                        return 1e20
                    reactor_ammonia_local = brentq(lambda NH3_guess: NH3_func(NH3_guess, Q_guess), 0, ammonia, xtol=1e-5)
                    _, _, _, _, XNH3, _ = cracker_reactor(reactor_ammonia_local, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, Q_guess)
                    self.reactor_ammonia_temp = reactor_ammonia_local
                    self.furnace_ammonia_temp = ammonia - reactor_ammonia_local
                    return XNH3[-1] - self.NH3_conversion
            
                self.Q_required = brentq(Q_func_combustion, 0, self.Q_cracker_design*1.5, xtol=1e-2)
                reactor_ammonia = self.reactor_ammonia_temp
                furnace_ammonia = self.furnace_ammonia_temp

        # Case 2: Adiabatic or isothermal → Q is determined by the system
        else:
            self.Q_required = False
            
            if hydrogen:
                # Case: hydrogen target is known
                # Find ammonia flowrate that gives the desired hydrogen production
                reactor_hydrogen = hydrogen / self.PSA_recovery_rate
                
                def ammonia_flowrate1(NH3_guess):
                    _, _, H2_molarflowrate, *_ = cracker_reactor(NH3_guess, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, self.Q_required)
                    H2_massflowrate = H2_molarflowrate[-1] * c.H2MOLMASS
                    return H2_massflowrate - reactor_hydrogen
                
                # Solve for NH3 feed rate that yields target H2 production
                reactor_ammonia = brentq(ammonia_flowrate1, 1e-10, 1e10, xtol=1e-5, maxiter=100)
                
            elif (ammonia and self.heat_source == 'electric') or (ammonia and self.thermal_mode == 'isothermal'):
                # Case: ammonia feed is fixed, heat provided electrically or no heat needed
                reactor_ammonia = ammonia
                
            elif ammonia and self.heat_source == 'ammonia combustion' and self.thermal_mode == 'adiabatic':
                # Case: ammonia feed is fixed, heat provided by ammonia combustion
                    
                def ammonia_flowrate2(NH3_guess):
                    _, _, _, _, _, Q = cracker_reactor(NH3_guess, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, self.Q_required)
                    NH3_furnace = Q[-1] / (self.eta_combustion * c.LHV_NH3) / 3600
                    return NH3_guess + NH3_furnace - ammonia
            
                reactor_ammonia = brentq(ammonia_flowrate2, 1e-10, ammonia, xtol=1e-5)
                
        self.T_cracker, self.N2_molarflowrate_cracker, self.H2_molarflowrate_cracker, self.NH3_molarflowrate_cracker, self.XNH3, self.Q_cracker = cracker_reactor(reactor_ammonia, self.T_react, self.p_react, 50, self.reactor_volume, self.thermal_mode, self.Q_required) 
        if hydrogen == False:
            reactor_hydrogen = self.H2_molarflowrate_cracker[-1] * c.H2MOLMASS
            hydrogen = reactor_hydrogen * self.PSA_recovery_rate 
            
        # Inlet conditions of NH3
        i = 0
        self.points.append('NH3 inlet')
        self.T_points[i] = self.T_in
        self.p_points[i] = self.p_in
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        self.m_points[i] = reactor_ammonia
        
        
        
        # NH3 lamination
        i += 1
        self.points.append('NH3 lamination')
        self.p_points[i] = self.p_react
        self.m_points[i] = self.m_points[i-1]
        self.h_points[i] = self.h_points[i-1]
        self.T_points[i] = PropsSI('T','P', self.p_points[i] * 100000,'H', self.h_points[i-1],'Ammonia')
        
        
        
        # NH3 heating with ambient water
        i += 1
        self.points.append('NH3 heating - ambient water')
        self.p_points[i] = self.p_points[i-1]
        self.m_points[i] = self.m_points[i-1]
        if self.T_points[i-1] < (c.AMBTEMP - 10):
            self.T_points[i] = c.AMBTEMP - 10    # fluid temperature after heating with water at ambient temperature, DT = 10 K
        else:
            self.T_points[i] = self.T_points[i-1]
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        
        T_cold_in_RHE = self.T_points[i]
        h_cold_in_RHE = self.h_points[i] 
        k_cold = PropsSI('conductivity','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia')
        mu_cold = PropsSI('viscosity','P', self.p_points[i] * 100000,'T', self.T_points[i],'Ammonia') 
        
        
        
        #Cracker inlet
        i += 1
        self.points.append('Cracker inlet')
        self.T_points[i] = self.T_react
        self.p_points[i] = self.p_points[i-1]
        self.h_points[i] = PropsSI('H','P', self.p_points[i] * 100000,'T',self.T_points[i],'Ammonia')
        self.m_points[i] = self.m_points[i-1]
        h_reactor_in = self.h_points[i]
        
        
        
        # Cracker outlet
        i += 1
        self.points.append('Cracker outlet')
        self.T_points[i] = self.T_cracker[-1]
        self.p_points[i] = self.p_points[i-1]
        molarfractions = [self.N2_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1]), self.H2_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1]), self.NH3_molarflowrate_cracker[-1] / (self.N2_molarflowrate_cracker[-1]+self.H2_molarflowrate_cracker[-1]+self.NH3_molarflowrate_cracker[-1])]
        self.mix_H2N2NH3.set_mole_fractions(molarfractions)
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i] * 100000, self.T_points[i])
        self.h_points[i] = self.mix_H2N2NH3.hmass()
        self.m_points[i] = self.m_points[i-1]
        
        if self.heat_source == 'electric':
            self.electric_heating_cracker = self.Q_cracker[-1]
            furnace_ammonia = 0
        if self.heat_source == 'ammonia combustion':
            furnace_ammonia = self.Q_cracker[-1] / (self.eta_combustion * c.LHV_NH3) / 3600   # [kg/s]
            self.electric_heating_cracker = 0
            
        components = ["N2", "H2", "NH3"]
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molarfractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molarfractions[i] * 100000, components[i]) for i in range(len(components))]
        mu_hot = 1 / sum([molarfractions[i] / viscosities[i] for i in range(len(components))])  # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        k_hot = np.dot(molarfractions, conductivities)   # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        T_hot_in_RHE = self.T_points[i]
        h_hot_in_RHE = self.h_points[i]


        
        # RHE  
        # Calculation of the heat that can be recovered - cooling of the products after the reactor - area from the design
        h_cold_out_RHE = None
        T_hot_out_RHE = None
        T_cold_out_RHE = None
        h_mixing = None
        T_mixing = None
        def recuperative_heat_exchanger(massflowrate_RHE):
            nonlocal h_cold_out_RHE, T_hot_out_RHE, T_cold_out_RHE, h_mixing, T_mixing
            tol = 1e-6  
            max_iter = 100 
            iter_count = 0 
            T_hot_out_guess = T_hot_in_RHE - 200
            T_cold_out_guess = T_cold_in_RHE + 200
            
            while iter_count < max_iter:
                iter_count += 1
                T_hot_out_RHE = T_hot_out_guess
                self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_react * 100000, T_hot_out_RHE)
                h_hot_out_RHE = self.mix_H2N2NH3.hmass()
                cp_hot = (h_hot_in_RHE - h_hot_out_RHE) / (T_hot_in_RHE - T_hot_out_RHE)
                T_cold_out_RHE = T_cold_out_guess
                h_cold_out_RHE = PropsSI('H','P', self.p_react * 100000,'T', T_cold_out_RHE,'Ammonia')
                cp_cold = (h_cold_out_RHE - h_cold_in_RHE) / (T_cold_out_RHE - T_cold_in_RHE)
                C_min = min(cp_cold * massflowrate_RHE, cp_hot * self.m_points[i])
                C_max = max(cp_cold * massflowrate_RHE, cp_hot * self.m_points[i])
                self.cp_min = min(cp_cold, cp_hot)
                C = C_min / C_max
                F1 = (k_hot / self.k_hot_design)**0.7 * (mu_hot / self.mu_hot_design)**(-0.5) * (self.m_points[i] / self.NH3_massflowrate_reactor)**0.8 * (cp_hot / self.cp_hot_design)**0.3
                F2 = (k_cold / self.k_cold_design)**0.7 * (mu_cold / self.mu_cold_design)**(-0.5) * (massflowrate_RHE / self.NH3_massflowrate_reactor)**0.8 * (cp_cold / self.cp_cold_design)**0.3
                F = 2 * F1 * F2 / (F1 + F2)
                U = self.U_RHE_design * F
                NTU = U * self.A_RHE / C_min
                self.epsilon_RHE = (1 - np.exp(-NTU * (1 - C))) / (1 - C * np.exp(-NTU * (1 - C)))
                self.Q_RHE= self.epsilon_RHE * C_min * (T_hot_in_RHE - T_cold_in_RHE)
                h_cold_out_RHE = h_cold_in_RHE + self.Q_RHE / massflowrate_RHE
                T_cold_out_new = PropsSI('T','P', self.p_react * 100000,'H', h_cold_out_RHE,'Ammonia') 
                massflowrate_bypass = self.m_points[i] - massflowrate_RHE
                h_mixing = (h_cold_out_RHE * massflowrate_RHE + h_cold_in_RHE * massflowrate_bypass) / self.m_points[i]
                T_mixing = PropsSI('T','P', self.p_react * 100000,'H', h_mixing,'Ammonia')    
                h_hot_out_RHE = h_hot_in_RHE - self.Q_RHE / self.m_points[i]
                T_hot_out_new = T_ph(self.mix_H2N2NH3, self.p_react, T_hot_in_RHE, h_hot_out_RHE)

                if abs(T_hot_out_new - T_hot_out_guess) < tol and abs(T_cold_out_new - T_cold_out_guess) < tol:
                    T_hot_out_RHE = T_hot_out_new
                    T_cold_out_RHE = T_cold_out_new
                    break

                T_hot_out_guess = T_hot_out_new
                T_cold_out_guess = T_cold_out_new

            return T_mixing - self.T_react
        
        try:
            # To reach the right inlet temperature to the cracker reactor a bypass of the RHE of a part of the cold flow can be implemented
            # If passing the entire flow rate of cold fluid through the RHE heats up too much
            massflowrate_RHE = brentq(recuperative_heat_exchanger, 0.01*self.m_points[i], self.m_points[i], xtol=1e-5, maxiter=100)
            self.bypass_RHE = (self.m_points[i]-massflowrate_RHE) / self.m_points[i] *100
            # No heat needed 
            self.electric_heating_cracker_inlet = 0
        except ValueError: 
            if recuperative_heat_exchanger(0.01*self.m_points[i]) < 0 and recuperative_heat_exchanger(self.m_points[i]) < 0:
                # If passing the entire flow rate of cold fluid through the RHE it remains too cold, a electric resistance is used to heat the reactants up to the target temperature
                massflowrate_RHE = self.m_points[i]
                self.bypass_RHE = 0
                recuperative_heat_exchanger(self.m_points[i])
                self.reactor_in_specific_heat=(h_reactor_in-h_cold_out_RHE)/1000   # [kJ/kg]
                self.electric_heating_cracker_inlet = self.reactor_in_specific_heat * self.m_points[i]
    
         
      
        # NH3 removal unit (adsorber) and PSA?
        
        
        electricity = self.electric_heating_cracker + self.electric_heating_cracker_inlet
        ammonia = reactor_ammonia + furnace_ammonia
        
        return(ammonia, electricity, hydrogen)
        
   
    def module_consumption_simple(self):
        """
           Calculates the relationship between hydrogen produced and the corresponding electricity consumption or the corresponding
           ammonia consumption. The spline is fitted with the load (hydrogen production / nominal hydrogen value). 
           The resulting electricity consumption and ammonia consumption from the spline is normalized and must be multiplied by the 
           flowrate of hydrogen to recover the actual consumption. 
           Alternatively it calculates the relationship between ammonia consumed and the corresponding electricity consumption or 
           the corresponding hydrogen production. The spline is fitted with the load (ammonia consumption / nominal ammonia value). 
           The resulting electricity consumption and hydrogen production from the spline is normalized and must be multiplied by 
           the flowrate of ammonia to recover the actual consumption or production. 
        """
        if self.check == True:
            spline_electricity = self.spline_electricity
            spline_ammonia = self.spline_ammonia
            spline_electricity_NH3 = self.spline_electricity_NH3
            spline_hydrogen_NH3 = self.spline_hydrogen_NH3
            
        else:
        
            # SPLINE from produced hydrogen 
            hydrogen = np.linspace(self.min_load * self.H2_massflowrate, self.H2_massflowrate, 50)
            
            load = []                       # normalized hydrogen load
            electricity_consumption = []    # electricity per kg of hydrogen
            NH3_consumption = []            # NH3 per kg of hydrogen
            
            for hyd in hydrogen:
                ammonia, electricity, _ = self.module_consumption(hydrogen=hyd)
                load.append(hyd / self.H2_massflowrate)
                electricity_consumption.append(electricity / hyd)
                NH3_consumption.append(ammonia / hyd)
            
            load = np.array(load)
            electricity_consumption = np.array(electricity_consumption)
            NH3_consumption = np.array(NH3_consumption)
            
            spline_electricity = UnivariateSpline(load, electricity_consumption, k=3)
            spline_ammonia = UnivariateSpline(load, NH3_consumption, k=3)
            
            # SPLINE from ammonia consumed
            NH3_min, *_ = self.module_consumption(hydrogen=self.H2_massflowrate*self.min_load)
            NH3_max, *_ = self.module_consumption(hydrogen=self.H2_massflowrate)
            NH3_in = np.linspace(NH3_min, NH3_max, 50)
            
            load_NH3 = []                  # normalized ammonia input load
            electricity_per_NH3 = []       # electricity per kg of ammonia
            hydrogen_per_NH3 = []          # hydrogen produced per kg of ammonia
            
            for NH3 in NH3_in:
                ammonia_used, electricity, hydrogen_out = self.module_consumption(ammonia=NH3)
                load_NH3.append(NH3 / NH3_max)
                electricity_per_NH3.append(electricity / NH3)
                hydrogen_per_NH3.append(hydrogen_out / NH3)
            
            load_NH3 = np.array(load_NH3)
            electricity_per_NH3 = np.array(electricity_per_NH3)
            hydrogen_per_NH3 = np.array(hydrogen_per_NH3)
            
            spline_electricity_NH3 = UnivariateSpline(load_NH3, electricity_per_NH3, k=3)
            spline_hydrogen_NH3 = UnivariateSpline(load_NH3, hydrogen_per_NH3, k=3)
            
            splines = {
                "spline_electricity": spline_electricity,
                "spline_ammonia": spline_ammonia,
                "spline_electricity_NH3": spline_electricity_NH3,
                "spline_hydrogen_NH3": spline_hydrogen_NH3
            }
            
            with open(self.path + '/consumption/calculations/' + self.name_serie, 'wb') as f:
                pickle.dump(splines, f)
            
            data_to_save = {
                "parameters": self.parameters,
                "additional_data": {"T_in": self.T_in, "p_in": self.p_in}
            }
            with open(self.path + f"{self.directory}/cracker_{self.file_structure}_{self.location_name}.pkl", 'wb') as f:
                pickle.dump(data_to_save, f)
            
        return spline_electricity, spline_ammonia, spline_electricity_NH3, spline_hydrogen_NH3
      
 
    def use(self, hyd, available_NH3):
        """
        The use function calculates the energy consumption in the operating range.
        
        Inputs:
            hyd: (>0) hydrogen mass flowrate needed [kg/s]
            available_NH3: ammonia available from the tank [kg/s]
            
        Outputs:
            ammonia: ammonia consumed [kg/s] 
            hydrogen: hydrogen produced [kg/s]
            electricity: electricity consumed [kW] 
        """
        hyd = abs(hyd)
        
        
        if self.calculation_mode == 'simple':
            electricity_consumption_simple = self.module_consumption_simple()[0]
            ammonia_consumption_simple = self.module_consumption_simple()[1]
            electricity_consumption_simple_NH3 = self.module_consumption_simple()[2]
            hydrogen_production_simple_NH3 = self.module_consumption_simple()[3]
    
        if round(hyd, 15) < round(self.H2_massflowrate * self.min_load, 15):
            # Below minimum load: run one module at minimum load
            hyd = self.H2_massflowrate * self.min_load
            self.n_modules_active = 1
    
        else:
            # Compute number of full-load modules
            self.n_modules_full = int(hyd // self.H2_massflowrate)
            remaining_hyd = hyd - self.n_modules_full * self.H2_massflowrate
    
            if remaining_hyd > 0:
                if round(remaining_hyd, 15) < round(self.H2_massflowrate * self.min_load, 15):
                    # Cannot operate below min load → round up
                    remaining_hyd = self.H2_massflowrate * self.min_load
                self.n_modules_active = self.n_modules_full + 1
            else:
                remaining_hyd = 0
                self.n_modules_active = self.n_modules_full
    
            if self.n_modules_active > self.n_modules:
                print(self.n_modules_active, self.n_modules)
                # Limit to installed capacity
                print(f"WARNING: H₂ demand ({hyd:.3f} kg/s) exceeds total installed cracker capacity "
                      f"({self.n_modules * self.H2_massflowrate:.3f} kg/s).")
                self.n_modules_active = self.n_modules
                self.n_modules_full = self.n_modules
                remaining_hyd = 0
    
        # Compute consumption and production based on full and partial modules
        ammonia = 0
        electricity = 0
        hydrogen = 0
        
        if hyd <= self.H2_massflowrate:
            # One module case
            if self.calculation_mode == 'simple':
                electricity = hyd * electricity_consumption_simple(hyd/self.H2_massflowrate)
                ammonia = hyd * ammonia_consumption_simple(hyd/self.H2_massflowrate)
                hydrogen = hyd
            else:
                ammonia, electricity, hydrogen = self.module_consumption(hydrogen = hyd)
        else:
            # Multiple modules
            if self.n_modules_full > 0:
                if self.calculation_mode == 'simple':
                    full_electricity = self.H2_massflowrate * electricity_consumption_simple(1)
                    full_ammonia = self.H2_massflowrate * ammonia_consumption_simple(1)
                    full_hydrogen = self.H2_massflowrate
                else:
                    full_ammonia, full_electricity, full_hydrogen = self.module_consumption(hydrogen = self.H2_massflowrate)
                ammonia += self.n_modules_full * full_ammonia
                electricity += self.n_modules_full * full_electricity
                hydrogen += self.n_modules_full * full_hydrogen 
            if remaining_hyd > 0:
                if self.calculation_mode == 'simple':
                    partial_electricity = remaining_hyd * electricity_consumption_simple(remaining_hyd/self.H2_massflowrate)
                    partial_ammonia = remaining_hyd * ammonia_consumption_simple(remaining_hyd/self.H2_massflowrate)
                    partial_hydrogen = remaining_hyd
                else:
                    partial_ammonia, partial_electricity, partial_hydrogen = self.module_consumption(hydrogen = remaining_hyd)
                ammonia += partial_ammonia
                electricity += partial_electricity
                hydrogen += partial_hydrogen
                
        # Check to see if there is enough ammonia available
        if ammonia > available_NH3:
            NH3 = available_NH3
            NH3_min = self.H2_massflowrate * self.min_load * ammonia_consumption_simple(self.min_load)  # min load NH3 consumption per module
            NH3_max = self.H2_massflowrate * ammonia_consumption_simple(1)                              # full load NH3 consumption per module
        
            if round(NH3, 15) < round(NH3_min, 15):
                # Not enough even for one module at min load → full shut down
                ammonia = 0
                hydrogen = 0
                electricity = 0
                self.n_modules_active = 0
        
            else:
                # Determine number of modules at full load
                self.n_modules_full = int(NH3 // NH3_max)
                remaining_ammonia = NH3 - self.n_modules_full * NH3_max
        
                if remaining_ammonia > 0:
                    if round(remaining_ammonia, 15) < round(NH3_min, 15):
                        # Not enough for second module → shut down
                        remaining_ammonia = 0
                        self.n_modules_active = self.n_modules_full
                    else:
                        self.n_modules_active = self.n_modules_full + 1
                else:
                    remaining_ammonia = 0
                    self.n_modules_active = self.n_modules_full
        
                if self.n_modules_active > self.n_modules:
                    print(f"WARNING: NH₃ input ({ammonia:.3f} kg/s) exceeds total installed cracker capacity "
                          f"({self.n_modules * NH3_max:.3f} kg/s).")
                    self.n_modules_active = self.n_modules
                    self.n_modules_full = self.n_modules
                    remaining_ammonia = 0
        
                # Compute consumption based on full and partial modules
                ammonia = 0
                electricity = 0
                hydrogen = 0
                
                if self.n_modules_active > 0:
                    if self.n_modules_full > 0:
                        if self.calculation_mode == 'simple':
                           full_electricity = NH3_max * electricity_consumption_simple_NH3(1)
                           full_ammonia = NH3_max 
                           full_hydrogen = NH3_max * hydrogen_production_simple_NH3(1)
                        else:
                            full_ammonia, full_electricity, full_hydrogen = self.module_consumption(ammonia = NH3_max)
                        ammonia += self.n_modules_full * full_ammonia
                        electricity += self.n_modules_full * full_electricity
                        hydrogen += self.n_modules_full * full_hydrogen
            
                    if remaining_ammonia > 0:
                        if self.calculation_mode == 'simple':
                            partial_electricity = remaining_ammonia * electricity_consumption_simple_NH3(remaining_ammonia / NH3_max)
                            partial_ammonia = remaining_ammonia 
                            partial_hydrogen = remaining_ammonia * hydrogen_production_simple_NH3(remaining_ammonia / NH3_max)
                        else:
                            partial_ammonia, partial_electricity, partial_hydrogen = self.module_consumption(ammonia = remaining_ammonia)
                        ammonia += partial_ammonia
                        electricity += partial_electricity
                        hydrogen += partial_hydrogen
    
        return(-ammonia, hydrogen, -electricity)
    
    
    def reactor_plot(self):
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.xlim(0, self.reactor_volume)
            line1, = plt.plot(np.linspace(0, self.reactor_volume, 50), self.NH3_molarflowrate_cracker, linestyle='-', color='lightcoral', label = 'NH$_{3}$', linewidth = 2.5)
            line2, = plt.plot(np.linspace(0, self.reactor_volume, 50), self.N2_molarflowrate_cracker, linestyle='-', color='orange', label = 'N$_{2}$', linewidth = 2.5)
            line3, = plt.plot(np.linspace(0, self.reactor_volume, 50), self.H2_molarflowrate_cracker, linestyle='-', color='green', label = 'H$_{2}$', linewidth = 2.5)
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("Molar flow rate [mol/s]")
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.xlim(0, self.reactor_volume)
            plt.plot(np.linspace(0, self.reactor_volume, 50), self.XNH3*100, linestyle='-', color='lightcoral', linewidth = 2.5)
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("NH$_{3}$ conversion [%]")
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.xlim(0, self.reactor_volume)
            plt.plot(np.linspace(0, self.reactor_volume, 50), self.T_cracker, linestyle='-', color='blue', linewidth = 2.5)
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("T [K]")
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.xlim(0, self.reactor_volume)
            plt.plot(np.linspace(0, self.reactor_volume, 50), self.Q_cracker, linestyle='-', color='red', linewidth = 2.5)
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("Q [kW]")
            plt.grid(True)
            plt.show()
    
    
    def tech_cost(self,tech_cost):
        """
        Inputs:
            tech_cost: dict
                'cost per unit': [€/(t_H2/day)]
                'OeM': operation and maintenance costs, percentage on initial investment [%]
                'refund': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of the initial investment [%]
                    'years': after how many years it will be replaced

        Ouputs:
            self.cost: dict
                'total cost': [€]
                'OeM': [€]
                'refund': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of the initial investment [%]
                    'years': after how many years it will be replaced
        """
        tech_cost = {key: value for key, value in tech_cost.items()}
           
        size = self.H2_massflowrate *  3600 * 24 / 1000 # [t_H2/day]
        
        if tech_cost['cost per unit'] == 'default price correlation':
            capacity = np.array([10, 20, 40, 60, 100, 160, 200, 300, 350])   # [tpd H2]
            cost = np.array([720, 560, 460, 400, 350, 300, 280, 250, 240])   # [k$/tpd]
            def cost_model(x, a, b):
                return a * x**b
            popt, _ = curve_fit(cost_model, capacity, cost)
            a_fit, b_fit = popt
            def fitted_cost(capacity_input):
                return cost_model(capacity_input, a_fit, b_fit)
            C = fitted_cost(size) * size * self.n_modules  # [k$]
            
            # capacity_range = np.linspace(10, 350, 500)
            # cost_model = fitted_cost(capacity_range)
            # plt.figure(figsize=(10, 6))
            # plt.plot(capacity_range, cost_model)
            # plt.scatter(capacity, cost)
            # plt.xlabel('Plant capacity [tpd of H$_{2}$]')
            # plt.ylabel('Capital cost [k$/tpd]')
            # plt.ylim(200, 800)
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
            
            exchange_rate = 0.95
            CEPCI_2021 = 708.8
            CEPCI_2024 = 800
            C = C * exchange_rate * 1000      # [€]
            C = C * CEPCI_2024 / CEPCI_2021
            tech_cost['total cost'] = C
            
        else:
            tech_cost['total cost'] = tech_cost['cost per unit'] * size * self.n_modules
            
        tech_cost['OeM'] = tech_cost['OeM'] * tech_cost['total cost'] / 100
    
        self.cost = tech_cost
        
"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    inp_test = {'Treact'    : 793,
                'Nflowrate' : 0.01,
                'number of modules': 1,
                'reactor conversion': 0.995,
                'thermal mode': 'constant Q',
                'Tout': 870,
                'heat source': 'ammonia combustion'}
    tech_cost = {"cost per unit": "default price correlation",
            "OeM": 3,
            "refund": { "rate": 0, "years": 0},
            "replacement": {"rate": 'calculated', "years": 20}}

    sim_steps   = 50      # [-] number of steps to be considered for the simulation - usually a time horizon of 1 year minimum is considered
    timestep    = 60      # [min] selected timestep for the simulation 
    location_name = 'electricity_consumer'
    path = r'I:\Drive condivisi\Mattia_Valentina\Ammonia\EFC_2025\Constant demand\input_dev_ammonia' 
    file_structure = 'studycase'
    file_general = 'general'
    
    cracker_test=cracker(inp_test, 288.15, 18, location_name, path, file_structure, file_general, timestep = 60) 
    cracker_test.reactor_plot()
    # ammonia, electricity, hydrogen = cracker_test.module_consumption(hydrogen =inp_test['Nflowrate']*0.5)
    
    
#%% # Cracker validation
    
    NH3_massflowrate = 1*10**-5 * c.NH3MOLMASS
    T_react = 800
    P_react = 1
    reactor_volume = 0.1 * (5*10**-3)**2 * 3.14
    
    T_cracker, N2_molarflowrate_cracker, H2_molarflowrate_cracker, NH3_molarflowrate_cracker, XNH3, Q = cracker_reactor(NH3_massflowrate, T_react, P_react, 50, reactor_volume, 'adiabatic') 
    z = np.array([0, 0.001, 0.005,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]) * (5*10**-3)**2 * 3.14
    T = np.array([800, 775, 750, 720, 690, 680, 674, 662.5, 659, 653, 650, 648, 635])
    NH3 = np.array([1, 0.96, 0.9, 0.85, 0.82, 0.80, 0.78, 0.77, 0.76, 0.755, 0.75, 0.745, 0.74])
    
    def logfun(z, a, b, c):
        return a * np.log(b*z + 1) + c
    
    # Logaritmica
    popt_Tlog, _ = curve_fit(logfun, z, T, p0=[-100, 50, 800], maxfev=10000)
    T_log = logfun(z, *popt_Tlog)
    popt_NH3log, _ = curve_fit(logfun, z, NH3, p0=[-0.1, 50, 1], maxfev=10000)
    NH3_log = logfun(z, *popt_NH3log)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,7), dpi=150)

    # Primo grafico: Temperatura
    ax1.set_xlim(0, reactor_volume)
    ax1.plot(np.linspace(0, reactor_volume, 50), T_cracker, linestyle='-', color='lightcoral',
             linewidth=2.5, label="This model")
    ax1.plot(z, T_log, linestyle='--', color='lightcoral', linewidth=2.5,
             label="Barat")
    ax1.set_xlabel("Reactor volume [m$^3$]")
    ax1.set_ylabel("T [K]")
    ax1.set_ylim(600, 800)
    ax1.grid(True)
    ax1.legend()
    
    # Secondo grafico: NH3
    ax2.set_xlim(0, reactor_volume)
    ax2.plot(np.linspace(0, reactor_volume, 50),
             NH3_molarflowrate_cracker/(NH3_molarflowrate_cracker
                                        + H2_molarflowrate_cracker
                                        + N2_molarflowrate_cracker),
             linestyle='-', color='cornflowerblue', linewidth=2.5,
             label="This model")
    ax2.plot(z, NH3_log, linestyle='--', color='cornflowerblue', linewidth=2.5,
             label="Barat")
    ax2.set_xlabel("Reactor volume [m$^3$]")
    ax2.set_ylabel("NH$_{3}$ conversion [-]")
    ax2.set_ylim(0.7, 1)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    
#%% # Module specific consumption at variable load
    
    import matplotlib.pyplot as plt
    
    cracker_test=cracker(inp_test, 288.15, 18, location_name, path, file_structure, file_general, timestep = 60)
    
    load = np.arange(cracker_test.min_load, 1.01, 0.05)
    SEC = np.zeros(len(load))
    SEC_simple = np.zeros(len(load))
    NH3_consumption = np.zeros(len(load))
    NH3_consumption_simple = np.zeros(len(load))
    for i, l in enumerate(load):
        NH3_consumption[i], SEC[i], _ = cracker_test.module_consumption(hydrogen = inp_test['Nflowrate']*l) 
        NH3_consumption[i] /= inp_test['Nflowrate']*l
        SEC[i] /= (inp_test['Nflowrate']*l*3600)      # [kWh/kg]
        SEC_simple[i] = cracker_test.module_consumption_simple()[0](l) / 3600
        NH3_consumption_simple[i] = cracker_test.module_consumption_simple()[1](l) 
        
    plt.figure(figsize=(8, 5), dpi=150)
    plt.xlim(30, 100)
    plt.plot(load * 100, SEC, linestyle='-', color='blue', label = "Consumption", linewidth = 2.5)
    plt.plot(load * 100, SEC_simple, linestyle='--', color='black', label = "Consumption simple", linewidth = 2.5)
    plt.xlabel("Load [%]")
    plt.ylabel("Specific consumption [kWh/kg$_{\mathrm{H_{\mathrm{2}}}}$]")
    plt.grid(True)
    plt.show()
        
        
    plt.figure(figsize=(8, 5), dpi=150)
    plt.xlim(30, 100)
    plt.plot(load * 100, NH3_consumption, linestyle='-', color='blue', label = "Consumption", linewidth = 2.5)
    # plt.plot(load * 100, NH3_consumption_simple, linestyle='--', color='black', label = "Consumption simple", linewidth = 2.5)
    plt.xlabel("Load [%]")
    plt.ylabel("Specific ammonia consumption [kg$_{\mathrm{NH_{\mathrm{3}}}}$/kg$_{\mathrm{H_{\mathrm{2}}}}$]")
    plt.grid(True)
    plt.show()
