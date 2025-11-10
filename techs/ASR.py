import numpy as np
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir))) 
import CoolProp.CoolProp as CP
from functools import lru_cache
from scipy.integrate import solve_ivp
from scipy.optimize import newton, brentq
from scipy.signal import savgol_filter
from scipy.interpolate import SmoothBivariateSpline, make_interp_spline
import pandas as pd
import pickle  
from core import constants as c
import matplotlib.pyplot as plt



def T_ps(AS, p, T, s_ref):          
    """
    The function iterates to find the temperature given the pressure p and the entropy s_ref, using the secant method.
    
    Inputs:
        AS: thermodinamic state
        p: pressure [bar]
        T: initial temperature [K]
        s_ref: reference entropy [J/(kg*K)]

    Output:
        T_found: temperature found [K]
    """
    p=p*100000
    
    @lru_cache(maxsize=None)
    def s(T):
        AS.update(CP.PT_INPUTS, p, T)
        return AS.smass()
    
    def f_T(T):
        return s(T) - s_ref

    T_found = newton(f_T, x0=T, tol=1e-2)
    
    return T_found


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
    
    @lru_cache(maxsize=None)
    def h(T):
        AS.update(CP.PT_INPUTS, p, T)
        return AS.hmass()
    
    def f_T(T):
        return h(T) - h_ref
    
    T_found = newton(f_T, x0=T, tol=1e-2)
    
    return T_found
    

def x_cond (mix_H2N2NH3, Ammonia, T, p, N2_mf, H2_mf):
    """
    The function calculates the molar fraction of the components (N2, H2, NH3) in the residual gas after ammonia condensation.

    Inputs:
        AS : thermodinamic state
        T : condenser final temperature [K]
        p : total pressure of the mixture [bar]
        N2_mf : molar fraction of nitrogen inside a mixture of H2 and N2 in stoichiometric ratio [-]
        H2_mf : molar fraction of hydrogen inside a mixture of H2 and N2 in stoichiometric ratio [-]

    Outputs:
        N2_molarfraction : molar fraction of nitrogen in the gas phase after ammonia condensation [-]
        H2_molarfraction : molar fraction of hydrogen in the gas phase after ammonia condensation [-]
        NH3_molarfraction : molar fraction of ammonia in the gas phase after condensation [-]
"""
    p=p*100000 
    
    Ammonia.update(CP.PT_INPUTS, p, T)
    liq_fugacity_coefficient=Ammonia.fugacity_coefficient(0)   # fugacity coefficient for liquid ammonia
    
    # Ref: Smith, Joseph Mauk, et al. Introduction to chemical engineering thermodynamics. Singapore: McGraw-Hill, 1949. 
    # Fugacities of NH3 inside the gas and the liquid are the same for the vapour-liquid equilibrium.
    # Fugacity = molar fraction * fugacity coefficient * total pressure. 
    # Equalizing the fugacities the results is: NH3 molar fraction inside the gas = liquid fugacity coefficient / vapour NH3 fugacity coefficient
    p_satNH3=CP.PropsSI('P', 'T', T, 'Q', 0, 'PR::Ammonia')
    NH3_molarfraction = p_satNH3 / p    # initial guess of molar fraction of ammonia inside the gas mixture - Raoult's Law
    N2H2_molarfraction = 1 - NH3_molarfraction
    N2_molarfraction = N2H2_molarfraction * N2_mf
    H2_molarfraction = N2H2_molarfraction * H2_mf
    mix_H2N2NH3.set_mole_fractions([N2_molarfraction, H2_molarfraction, NH3_molarfraction])
    mix_H2N2NH3.update(CP.PT_INPUTS, p, T)
    
    tolerance = 1e-5
    max_iterations = 100
   
    for iteration in range(max_iterations):
        NH3vap_fugacity_coefficient = mix_H2N2NH3.fugacity_coefficient(2)   # fugacity coefficient for ammonia inside the mixture, it represents the ratio between the fugacity and the partial pressure (p_tot*x)
        new_x_NH3 = liq_fugacity_coefficient / NH3vap_fugacity_coefficient 

        if abs(new_x_NH3 - NH3_molarfraction) < tolerance:
            break
        if (iteration+1==max_iterations):
            print("Maximum iterations reached without convergence in the x_cond function (ASR.py)")

        NH3_molarfraction = new_x_NH3
        N2H2_molarfraction=1-NH3_molarfraction
        N2_molarfraction=N2H2_molarfraction*N2_mf
        H2_molarfraction=N2H2_molarfraction*H2_mf
        mix_H2N2NH3.set_mole_fractions([N2_molarfraction, H2_molarfraction, NH3_molarfraction])
        mix_H2N2NH3.update(CP.PT_INPUTS, p,T)

    return N2_molarfraction, H2_molarfraction, NH3_molarfraction  
        
    
def reactor_bed(mix_H2N2NH3, mix_massflowrate, N2_molarflowrate_i, H2_molarflowrate_i, NH3_molarflowrate_i, N2_molarflowrate0, T_i, XN2_i, p, n, reactor_bed_volume):
    """
    The function models the chemical reaction process in an ammonia synthesis reactor bed.
    It calculates the changes in the molar flow rates of nitrogen, hydrogen, and ammonia, as well as
    the temperature and nitrogen conversion, as a function of the reactor volume.

    Inputs:
        self.mix_H2N2NH3: mixture object representing the components H2, N2, and NH3
        mix_massflowrate: mass flow rate of the mixture [kg/s]
        N2_molarflowrate_i: initial molar flow rate of nitrogen - entering the bed [mol/s]
        H2_molarflowrate_i: initial molar flow rate of hydrogen - entering the bed [mol/s]
        NH3_molarflowrate_i: initial molar flow rate of ammonia - entering the bed [mol/s]
        N2_molarflowrate0: nitrogen molar flow rate entering the reactor (first bed) [mol/s]
        T_i: initial temperature [K]
        XN2_i: initial nitrogen conversion [-]
        p: reactor operating pressure [bar]
        n: number of points for the solver
        reactor_bed_volume: volume of the reactor bed [m^3]
        
    Outputs:
        N2_molarflowrate_bed: molar flow rate of nitrogen inside the reactor bed [mol/s]
        H2_molarflowrate_bed:  molar flow rate of hydrogen inside the reactor bed [mol/s]
        NH3_molarflowrate_bed:  molar flow rate of ammonia inside the reactor bed [mol/s]
        T_bed: temperature inside the reactor bed [K]
        XN2_bed: nitrogen conversion inside the reactor bed [-]
        p_bed: pressure inside the reactor bed [bar]
    """
     
    R = 1.987  # gas constant [cal/(mol*K)]
    p_atm=p * 0.986923 # conversion in [atm]
    # Coefficients for the correction of the conversion rate (eta) - depending on the pressure
    b0 = np.polyval(c.poly_coefficients['b0'], p_atm)
    b1 = np.polyval(c.poly_coefficients['b1'], p_atm)
    b2 = np.polyval(c.poly_coefficients['b2'], p_atm)
    b3 = np.polyval(c.poly_coefficients['b3'], p_atm)
    b4 = np.polyval(c.poly_coefficients['b4'], p_atm)
    b5 = np.polyval(c.poly_coefficients['b5'], p_atm)
    b6 = np.polyval(c.poly_coefficients['b6'], p_atm)
    
    def reactor_ode(V, y, mix_massflowrate, N2_molarflowrate0):
        N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XN2 = y  
        
        mix_molarflowrate = N2_molarflowrate + H2_molarflowrate + NH3_molarflowrate
        x_N2 = N2_molarflowrate / mix_molarflowrate
        x_H2 = H2_molarflowrate / mix_molarflowrate
        x_NH3 = NH3_molarflowrate / mix_molarflowrate
        mix_H2N2NH3.set_mole_fractions([x_N2, x_H2, x_NH3])
        mix_H2N2NH3.update(CP.PT_INPUTS, p * 100000, T)

        # Resolution of the reaction kinetics
        "N₂ + 3H₂ --> 2NH₃"
        k = np.exp (2.303 * 14.7102 - 39075 / (R * T))  # reaction rate constant [kmol_NH3/(h*m^3)] (Singh and Saraf, 1979)
        logK_eq = -2.691122 * np.log10(T) - 5.519265e-5 * T + 1.848863e-7 * T**2 + 2001.6 / T + 2.6899  # equilibrium constant [-], Arrhenius-type equation (Gillespie and Beattie, 1930)
        K_eq = 10**logK_eq  # reaction equilibrium constant
        
        # Fugacity coefficients of the mixture components
        fugacity_coefficient_N2 = mix_H2N2NH3.fugacity_coefficient(0)           #0.93431737 + 0.3101804e-3 * T + 0.295895e-3 * p - 0.270729e-6 * T**2 + 0.4775207e-6 * p**2                   
        fugacity_coefficient_H2 = mix_H2N2NH3.fugacity_coefficient(1)           #np.exp(np.exp(-3.8402 * T**0.125 + 0.541) * p - np.exp(-0.1263 * T**0.5 - 15.980) * p**2 + 300 * (np.exp(-0.011901 * T - 5.941)) * (np.exp(-p/300) - 1))   
        fugacity_coefficient_NH3 = mix_H2N2NH3.fugacity_coefficient(2)          #0.1438996 + 0.2028538e-2 * T - 0.4487672e-3 * p - 0.1142945e-5 * T**2 + 0.2761216e-6 * p**2                                                                

        # Activities = fugacity/reference pressure (reference pressure = 1 atm) (Lewis and Randall, 1923)
        # Fugacity = molar fraction * fugacity coefficient * p tot
        a_N2 = x_N2 * fugacity_coefficient_N2 * p_atm
        a_H2 = x_H2 * fugacity_coefficient_H2 * p_atm
        a_NH3 = x_NH3 * fugacity_coefficient_NH3 * p_atm

        # Ammonia production rate - modified form of Temkin equation (Dyson and Simon, 1968)
        alpha = 0.57  # kinetic fitting parameter [−], variable from 0.5 to 0.75
        r = k * (K_eq**2 * a_N2 * (a_H2**3 / a_NH3**2)**alpha - (a_NH3**2 / a_H2**3)**(1 - alpha))  # ammonia production rate [kmol_NH3/(h*m^3)]
        r = r * 1000 / 3600  # conversion in [mol_NH3/(s*m^3)]
        # Correction with the effectiveness factor eta - effect of resistance to transfer of mass or heat (Dyson and Simon, 1968)
        eta = b0 + b1 * T +  b2 * XN2 + b3 * T**2 + b4 * XN2**2 + b5 * T**3 + b6 * XN2**3  
        r = r * eta

        # Molar flowrate variations for m^3 
        dN2_molarflowrate_dV = -r / 2   # [mol/(s*m^3)], -r/2 because every mole of NH3 produced, half of N2 is consumed
        dH2_molarflowrate_dV = -3/2 * r   # [mol/(s*m^3)], -3/2r because every mole of NH3 produced, 3/2 of H2 is consumed
        dNH3_molarflowrate_dV = r  # [mol/(s*m^3)]
        # Nitrogen based conversion variation for m^3 (XN2 = (N2_in - N2_out) / N2_in)
        dXN2_dV = r / 2 / N2_molarflowrate0  # [1/m^3], r/2 = nitrogen consumed
        # Temperature variation for m^3
        heat_of_reaction = abs(4.184 * ((-0.5426 - 840.609 / T - 4.59734e8 / T**3) * p_atm - 5.34685 * T - 0.2525e-3 * T**2 + 1.69197e-6 * T**3 - 9157.09))  # [J/mol_NH3] (Gillespie and Beattie, 1930)     
        cp_mixture = mix_H2N2NH3.cpmass()    # [J/kg*K]
        dT_dV = heat_of_reaction * r / (mix_massflowrate * cp_mixture) # [K/m^3]

        return [dN2_molarflowrate_dV, dH2_molarflowrate_dV, dNH3_molarflowrate_dV, dT_dV, dXN2_dV]
    
    # Inputs for the solution of the ODE problem
    y0 = [N2_molarflowrate_i, H2_molarflowrate_i, NH3_molarflowrate_i, T_i, XN2_i]  # initial values 
    V_span = [0, reactor_bed_volume]            # interval of integration
    # Solution using method BDF - for stiff problems
    sol = solve_ivp(reactor_ode, V_span, y0, method='BDF', args=(mix_massflowrate, N2_molarflowrate0), t_eval=np.linspace(0, reactor_bed_volume, n))
    N2_molarflowrate_bed = sol.y[0]
    H2_molarflowrate_bed = sol.y[1]
    NH3_molarflowrate_bed = sol.y[2]
    T_bed = sol.y[3]
    XN2_bed = sol.y[4]
        
    return T_bed, N2_molarflowrate_bed, H2_molarflowrate_bed, NH3_molarflowrate_bed, XN2_bed
    

def profile_RHE(Hotfluid, Coldfluid, n, massflowrate, T_hot_in, p_hot, T_cold_in, T_cold_out, p_cold):
    """
    The function models the feed-effluent heat exchanger in the design of the plant - 
    check that there is no reversal of the heat exchange flow. this is achieved by 
    discretizing the exchanger and progressively calculating the heat exchange and 
    temperature. It returns also some design parameters caalculated with the epsilon-NTU 
    method.
    
    """
    
    Coldfluid.update(CP.PT_INPUTS, p_cold*100000, T_cold_in)
    h_cold_in = Coldfluid.hmass()	# enthalpy cold fluid HE inlet
    Coldfluid.update(CP.PT_INPUTS, p_cold*100000, T_cold_out)
    h_cold_out = Coldfluid.hmass()	# enthalpy cold fluid HE outlet
    Q = massflowrate * (h_cold_out - h_cold_in)
    
    Hotfluid.update(CP.PT_INPUTS, p_hot*100000, T_hot_in)
    h_hot_in = Hotfluid.hmass()	 # enthalpy hot fluid HE inlet
     
    DQ = Q / (n-1)	# discretization of Q
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
       h_hot[i]=h_hot[i-1]-dh_hot
       T_hot[i]=T_ph(Hotfluid, p_hot, T_hot[i-1], h_hot[i])
       
       # Cold fluid
       h_cold[i]= h_cold[i-1]+dh_cold
       T_cold[i] = T_ph(Coldfluid, p_cold, T_cold[i-1], h_cold[i])
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
    U = 536   # [W / (m^2 * K)]
    # Counterflow heat exchanger
    NTU = 1 / (C - 1) * np.log((epsilon - 1) / (epsilon * C - 1))
    A = NTU * C_min / U

    return T_hot, T_cold, Q_vals, A, U, epsilon, pinch_point
    

def profile_RHE_off_design(Hotfluid, Coldfluid, n, massflowrate_hot, T_hot_in, p_hot, massflowrate_cold, T_cold_in, T_cold_out, p_cold):
    
    Coldfluid.update(CP.PT_INPUTS, p_cold*100000, T_cold_in)
    h_cold_in = Coldfluid.hmass()	# enthalpy cold fluid HE inlet
    Coldfluid.update(CP.PT_INPUTS, p_cold*100000, T_cold_out)
    h_cold_out = Coldfluid.hmass()	# enthalpy cold fluid HE outlet
    Q = massflowrate_cold * (h_cold_out - h_cold_in)
    
    Hotfluid.update(CP.PT_INPUTS, p_hot*100000, T_hot_in)
    h_hot_in = Hotfluid.hmass()	 # enthalpy hot fluid HE inlet
     
    DQ = Q / (n-1)	# discretization of Q
    dh_hot = DQ / massflowrate_hot	# delta_h hot side
    dh_cold = DQ / massflowrate_cold  # delta_h cold side
    
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
       h_hot[i]=h_hot[i-1]-dh_hot
       T_hot[i]=T_ph(Hotfluid, p_hot, T_hot[i-1], h_hot[i])
       
       # Check that ammonia is not starting to condense
       if T_hot[i] < 405.56:  # critical temperature
           x = Hotfluid.get_mole_fractions()  
           P_tot = p_hot * 100000
           x_NH3 = x[2]  
           Hotfluid.update(CP.PT_INPUTS, P_tot, T_hot[i])
           fugacity_coefficient_NH3 = Hotfluid.fugacity_coefficient(2)  
           fugacity_NH3 = x_NH3 * fugacity_coefficient_NH3 * P_tot
           P_sat_NH3 = CP.PropsSI("P", "T", T_hot[i], "Q", 0, "Ammonia") 
           Ammonia = CP.AbstractState('PR', 'Ammonia')
           Ammonia.specify_phase(CP.iphase_liquid)
           Ammonia.update(CP.PT_INPUTS, P_sat_NH3, T_hot[i])
           fugacity_coefficient_NH3_sat = Ammonia.fugacity_coefficient(0)  
           fugacity_NH3_sat = P_sat_NH3 * fugacity_coefficient_NH3_sat  
           if fugacity_NH3 > fugacity_NH3_sat:
                 print(f"NH3 condensation detected at the pass {i} at T={T_hot[i]:.2f} K. The profile_RHE function considers just gas phase in the RHE.")
       
       # Cold fluid
       h_cold[i]= h_cold[i-1]+dh_cold
       T_cold[i] = T_ph(Coldfluid, p_cold, T_cold[i-1], h_cold[i])
       # Calculate the heat exchanged incrementally
       Q_HE += DQ
       Q_vals[i] = Q_HE
       
    # Calculate the temperature difference 
    for i in range(0,n):
        deltaT[i] = T_hot[i] - T_cold[n-i-1]
        if deltaT[i] <= 0:
           print("Warning: negative pinch point inside the feed-effluent heat exchanger")
           
    return T_hot, T_cold, Q_vals


def RHE_plot(T_hot, T_cold, Q):
    
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
          head_width=15, head_length=15, fc='red', ec='red')
        plt.arrow(Q[mid_index]/1000, T_cold[::-1][mid_index], -dx_cold/1000, -dy_cold, 
                  head_width=15, head_length=15, fc='blue', ec='blue')
        plt.xlabel("Heat exchanged [kW]", fontsize=12)
        plt.ylabel("Temperature [K]", fontsize=12)
        plt.title("RHE T-Q diagram", fontsize = 18, pad=15)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    

def part_load_COP(PLR):
    """
    Calculation of the part_load_COP (scaling factor for the COP) on the given load fraction (PLR = partial load ratio) - 
    from chillers COP-PLR second-order polynomial model.
    """
    # Ref: Acerbi, Federica, Mirco Rampazzo, and Giuseppe De Nicolao. "An exact algorithm for the optimal chiller loading problem and its application to the optimal chiller sequencing problem." Energies 13.23 (2020): 6372.
    a = 0.9000  
    b = 1.8432
    c = -1.4188
    part_load_COP = (a+b*PLR+c*PLR**2) / (a+b+c)

    # loads = np.linspace(0.2, 1, 50)
    # COP = np.zeros(len(loads))
    # i = 0
    # for PLR in loads:
    #     COP[i] = (a+b*PLR+c*PLR**2) / (a+b+c) 
    #     i +=1
    # plt.figure(figsize=(8, 5), dpi=150)
    # plt.plot(loads*100, COP, color='lightcoral', label = '$COP_{part} = COP_{off-design} / COP_{design}$', linewidth=2.2)
    # plt.xlabel('Load fraction [%]')
    # plt.ylabel('$COP_{part}$ [-]')
    # plt.title('COP at partial load')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    return part_load_COP


def part_load_eta(massflowrate_fraction):
    """
    Calculation of the part_load_eta (scaling factor for the isentropic efficiency) for a reciprocating 
    compressor based on the given massflowrate fraction.
    Uses a second-order polynomial fit based on literature curves.
    """
    # Ref: Wang, L., et al. "Performance comparison of capacity control methods for reciprocating compressors." IOP Conference Series: Materials Science and Engineering. Vol. 90. No. 1. IOP Publishing, 2015.
    m_fraction = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    part_load_eta_values = np.array([0.82, 0.86, 0.9, 0.93, 0.96, 0.99, 1])
    coefficients = np.polyfit(m_fraction, part_load_eta_values, 2)
    model = np.poly1d(coefficients)
    part_load_eta = model(massflowrate_fraction)
    
    # m_fractions = np.linspace(0.3, 1, 50)
    # eta = np.zeros(len(m_fractions))
    # i = 0
    # for mf in m_fractions:
    #     eta[i] = model(mf)
    #     i +=1 
    # plt.figure(figsize=(8, 5), dpi=150)
    # plt.plot(m_fractions*100, eta, label=r"$\eta_{part} = \eta_{iso, off-design} / \eta_{iso, design}$", color='lightcoral', linewidth=2.2)
    # plt.xlabel('Massflowrate fraction [%]')
    # plt.ylabel('$\eta_{part}$ [-]')
    # plt.title('Compressor partial load efficiency')
    # plt.legend()
    # plt.grid(True)
    # plt.ylim(0, 1.1)
    # plt.show()
    
    return part_load_eta


def part_load_eta_loop(head_fraction, massflowrate_fraction):
    """
    Calculation of the part_load_eta (scaling factor for the isentropic efficiency) for a screw 
    compressor based on the given massflowrate fraction and head fraction.
    Uses a third-order polynomial fit based on literature curves.
    """
    # Ref: Brasz, Joost JJ. "Comparison of part-load efficiency characteristics of screw and centrifugal compressors." (2006).
    data_points = np.array([
        [0.28, 1, 0.5], [0.38, 1, 0.6], [0.51, 1, 0.7], [0.64, 1, 0.8], [0.81, 1, 0.9], [1, 1, 1], # Percentage head 100%
        [0.27, 0.8, 0.5], [0.37, 0.8, 0.6], [0.5, 0.8, 0.7], [0.65, 0.8, 0.8], [0.84, 0.8, 0.9],   # Percentage head 80%
        [0.28, 0.6, 0.5], [0.38, 0.6, 0.6], [0.5, 0.6, 0.7], [0.72, 0.6, 0.8], [1, 0.6, 0.9],   # Percentage head 60%
        [0.35, 0.4, 0.5], [0.57, 0.4, 0.6], [0.83, 0.4, 0.7],   # Percentage head 40%
        [1, 0.2, 0.5]])  # Percentage head 20%

    m_fraction = data_points[:, 0]  
    h_fraction = data_points[:, 1]   
    part_load_eta_values = data_points[:, 2] 
    
    X = np.column_stack((
        np.ones_like(h_fraction),              
        h_fraction,                            
        m_fraction,
        h_fraction**2,                     
        m_fraction**2,
        h_fraction * m_fraction,
        h_fraction**3,                         
        m_fraction**3,
        h_fraction**2 * m_fraction,
        h_fraction * m_fraction**2))
    coefficients = np.linalg.lstsq(X, part_load_eta_values, rcond=None)[0]
    
    
    def model(head_fraction, massflowrate_fraction):
        return (
            coefficients[0] +
            coefficients[1] * head_fraction +
            coefficients[2] * massflowrate_fraction +
            coefficients[3] * head_fraction**2 +
            coefficients[4] * massflowrate_fraction**2 +
            coefficients[5] * head_fraction * massflowrate_fraction +
            coefficients[6] * head_fraction**3 +
            coefficients[7] * massflowrate_fraction**3 +
            coefficients[8] * head_fraction**2 * massflowrate_fraction +
            coefficients[9] * head_fraction * massflowrate_fraction**2)
    part_load_eta_loop = model(head_fraction, massflowrate_fraction)

    # m_fraction_grid, h_fraction_grid = np.meshgrid(np.linspace(0.27, 1, 50),np.linspace(0, 1, 50))
    # efficiency_pred = model(h_fraction_grid, m_fraction_grid)
    # plt.figure(figsize=(8, 5), dpi=150)
    # desired_efficiency_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # contour = plt.contour(m_fraction_grid*100, h_fraction_grid*100, efficiency_pred, levels=desired_efficiency_levels, cmap="tab20b")
    # plt.clabel(contour, inline=True, fmt=r"$\eta_{part}=%.2f$") 
    # plt.title("Compressor partial load efficiency")
    # plt.xlabel("Massflowrate fraction [%]")
    # plt.ylabel("Head fraction [%]")
    # plt.text(0.5, 0.9, r"$\eta_{part} = \eta_{iso, off-design} / \eta_{iso, design}$", horizontalalignment='left', verticalalignment='center', color='black', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.5'))
    # # plt.scatter(m_fraction*100, h_fraction*100, c=part_load_eta_values, cmap="viridis", edgecolor='k') 
    # plt.grid(True)
    # plt.show()
    
    return part_load_eta_loop


class ASR:  

    def __init__(self, parameters, timestep_number, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, buffer_info=False, timestep=False):       
        """
            Create a general ASR object - adiabatic reactor with three beds - calculation of design conditions.
            Ammonia synthesis subcicle:
                - initialization of the components variables and properties
                - compression of the mixture of reactants in the main compressor or in two separate compressors if the 
                  exit pressures from PSA and electrolyzer are not the same
                - calculation of the adiabatic reactor with the chemical kinetics
                - recovery of heat inside a recuperative heat exchanger
                - energy expended to cool down the mixture after the reactor and extract the liquid ammonia    
                - compression of the recycled gases

            Inputs:
                timestep_number: number of the simulation timesteps
                T_p_hyd[0]: temperature of the hydrogen after the electrolyzer [K]
                T_p_hyd[1]: pressure of the hydrogen after the electrolyzer [bar]
                T_p_nitro[0]: temperature of the nitrogen after the PSA [K]
                T_p_nitro[1]: pressure of the nitrogen after the PSA [bar]
                buffer_info[0]: temperature of the hydrogen inside buffer [K]
                buffer_info[1]: maximum pressure of the hydrogen inside the buffer [bar]
                buffer_info[2]: minimum pressure of the hydrogen inside the buffer [bar]
                parameters: dictionary  
                    'Preact': pressure inside the reactor [bar]
                    'Treact': list with the inlet temperatures of the three beds [K]
                    'reactor beds volume': list with the volumes of the three beds [m^3]
                    'max_ammonia': maximum flow rate of ammonia producible [kg/s]
                    'comp efficiency': compressors politropic efficiency [-]
                    'nstages feed compressor': number of stages of the feed compressor
                     or
                    'nstages N2 compressor': number of stages of the N2 compressor
                    'nstages H2 compressor': number of stages of the H2 compressor
                    'Tcondenser': temperature of the condenser used to separate the liquid ammonia [K]
                    'cooling COP': refrigeration cycle COP [-]
                    'calculation mode': if 'simple' is chosen, the result of the electricity consumed 
                                         comes from the interpolation with a fifth degree polynomial of the consumes 
                                         inside the operational range of the reactor (max ammonia - min ammonia)
                        
            Outputs: ASR object able to:
                consume N2, H2 and electricity and produce ammonia .use(step, hyd, hyd_buffer)
        """
        # For consumption_simple function
        # If the spline fit has already been saved as file.pkl for the same ASR, this file is used, otherwise is calculated
        
        check = True # True if no parameters are changed from the old simulation
            
        # # Directory for storing previous simulation data
        directory = './consumption'

        # Checking if the previous simulation exists
        if os.path.exists(path + f"{directory}/ASR_{file_structure}_{location_name}.pkl"):
            with open(path + f"{directory}/ASR_{file_structure}_{location_name}.pkl", 'rb') as f:
                ps_parameters = pickle.load(f)  # Load previous simulation parameters
                par_to_check = ['comp efficiency', 'cooling COP', 'Treact', 'Preact', 'Tcondenser', 'nstages N2 compressor', 'nstages H2 compressor']
                for par in par_to_check:
                    if par in ps_parameters["parameters"] and par in parameters:  # Some parameters haven't to be defined
                        if ps_parameters["parameters"][par] != parameters[par]:
                            check = False
                if T_p_hyd != [ps_parameters["additional_data"]['T_H2'], ps_parameters["additional_data"]['p_H2']]:
                    check = False
                if T_p_nitro != [ps_parameters["additional_data"]['T_N2'], ps_parameters["additional_data"]['p_N2']]:
                    check = False
                if buffer_info != False:
                    if buffer_info != ps_parameters["additional_data"]['buffer_info']:
                        check = False

        else:
            check = False
            
        if check == False:
            for file_name in os.listdir(path + '/consumption/calculations'):
                file_path = os.path.join(path + '/consumption/calculations', file_name)
                if os.path.isfile(file_path):  
                    os.remove(file_path)
        if parameters['reactor beds volume'] == 'automatic dimensioning':
           name_serie = f"ASR_{parameters['reactor beds volume']}_{location_name}_{file_general}_{file_structure}.pkl"
        else:
            name_serie = f"ASR_{parameters['max_prod']}_{parameters['reactor beds volume']}_{location_name}_{file_general}_{file_structure}.pkl"
        if check and os.path.exists(path + '/consumption/calculations/' + name_serie): 
            with open(path + '/consumption/calculations/' + name_serie, 'rb') as f:
              splines = pickle.load(f)
              self.spline_1 = splines["spline_1"]
              self.spline_2 = splines["spline_2"]
              self.spline_3 = splines["spline_3"]
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
        
        self.cache_loop_massflowrate = {}
        self.cache_T_beds = {}
        self.cache_molarflowrate_in = {}
        self.cache_molarflowrate_out = {}
        
        self.cost = False # will be updated with tec_cost()
        
        self.only_renewables = parameters.get('only_renewables', False)
        self.strategy = parameters.get('strategy', None)
        self.p_react = parameters.get('Preact')
        self.T_react = parameters.get('Treact')
        self.T_in_Ibed = self.T_react[0]
        self.T_in_IIbed = self.T_react[1]
        self.T_in_IIIbed = self.T_react[2]
        self.max_ammonia = parameters.get('max_prod')
        self.max_hydrogen = self.max_ammonia * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS)
        self.max_nitrogen = self.max_ammonia * c.N2MOLMASS / (2*c.NH3MOLMASS)
        self.T_cond = parameters.get('Tcondenser')
        self.COP_design = parameters.get('cooling COP')
        self.reactor_beds_volume = parameters.get('reactor beds volume')
        if self.reactor_beds_volume == 'automatic dimensioning':
            self.reactor_beds_volume = self.reactor_dimensioning()
        self.reactor_Ibed_volume = self.reactor_beds_volume[0]
        self.reactor_IIbed_volume = self.reactor_beds_volume[1]
        self.reactor_IIIbed_volume = self.reactor_beds_volume[2]
        self.eta_pol = parameters.get('comp efficiency')
        self.nstages_feed_compressor = parameters.get('nstages feed compressor', None)
        self.nstages_N2_compressor = parameters.get('nstages N2 compressor', None)
        self.nstages_H2_compressor = parameters.get('nstages H2 compressor', None) 
        self.calculation_mode = parameters.get('calculation mode', None)
        
        self.min_load = 0.3     # minimum load = 30 %
        self.min_ammonia = self.min_load * self.max_ammonia 
        self.min_hydrogen = self.min_ammonia * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS)
        self.min_nitrogen = self.min_ammonia * c.N2MOLMASS / (2*c.NH3MOLMASS)
        self.max_ramp_rate = 0.3   # load maximum hourly ramp rate = 30 %
        self.load = np.zeros(timestep_number)
        self.T_IC = c.AMBTEMP + 10    # [K] fluid temperature after cooling with water at ambient temperature, DT = 10 K
        self.eta_motor = 0.95       # assumed efficiency of electric motor driving the compressors
        self.loop_pressure_drop = 0.06*self.p_react      # [bar] assumed pressure drop inside the loop, 6% of the reaction pressure
        self.buffer_info = buffer_info
        
        #timestep
        self.timestep = timestep if timestep else c.timestep  # [min] simulation timestep 
        # Operational period  
        self.state = parameters["state"]  # when not in operational period, off or minimum load                      
        self.operational_period = parameters["operational_period"]
        initial_day, final_day = self.operational_period.split(',')    #extract inital and final operational days
        initial_day = pd.to_datetime(initial_day, format = '%d-%m')
        final_day = pd.to_datetime(final_day, format = '%d-%m')
        initial_day = initial_day.day_of_year
        final_day = final_day.day_of_year
        self.initial_hour = (initial_day - 1) * 24  
        self.final_hour = final_day * 24
        # Control on hydrogen buffer level
        self.buffer_control = parameters.get('buffer control', None)


        # Creation of the thermodinamic state for pure fluids with Peng-Robinson cubic EOS
        self.Nitrogen = CP.AbstractState('PR', "Nitrogen")
        self.Nitrogen.specify_phase(CP.iphase_gas)
        self.Hydrogen = CP.AbstractState('PR', "Hydrogen")
        self.Hydrogen.specify_phase(CP.iphase_gas)
        self.Ammonia = CP.AbstractState('PR', 'Ammonia')
        self.Ammonia.specify_phase(CP.iphase_liquid)
        # Creation of the thermodinamic state for the mixture N2/H2 with Peng-Robinson cubic EOS
        self.mix_H2N2 = CP.AbstractState('PR',"Nitrogen&Hydrogen")  
        self.mix_H2N2.set_binary_interaction_double(0,1,"kij",-0.036)    # from UNISIM
        self.mix_H2N2.specify_phase(CP.iphase_gas)  
        # Creation of the thermodinamic state for the mixture N2/H2/NH3 with Peng-Robinson cubic EOS        
        self.mix_H2N2NH3 = CP.AbstractState('PR',"Nitrogen&Hydrogen&Ammonia") 
        self.mix_H2N2NH3.set_binary_interaction_double(0,1,"kij",-0.036)   # from UNISIM
        self.mix_H2N2NH3.set_binary_interaction_double(0,2,"kij",0.222)    # from UNISIM
        self.mix_H2N2NH3.set_binary_interaction_double(1,2,"kij",0.000)    # from UNISIM
        self.mix_H2N2NH3.specify_phase(CP.iphase_gas)
        
        # Inlet conditions of N2 and H2
        self.T_N2=T_p_nitro[0]  # inlet temperature for nitrogen (from PSA)
        self.p_N2=T_p_nitro[1]  # inlet pressure for nitrogen (from PSA)
        self.Nitrogen.update(CP.PT_INPUTS, self.p_N2*100000, self.T_N2)
        self.T_H2=T_p_hyd[0]   # inlet temperature for hydrogen (from electrolyzer)
        self.p_H2=T_p_hyd[1]   # inlet pressure for hydrogen (from electrolyzer)
        self.Hydrogen.update(CP.PT_INPUTS, self.p_H2*100000, self.T_H2)
        self.cv_N2=self.Nitrogen.cvmass()
        self.cv_H2=self.Hydrogen.cvmass()
        self.cp_N2=self.Nitrogen.cpmass()
        self.cp_H2=self.Hydrogen.cpmass()
        self.h_N2=self.Nitrogen.hmass()
        self.h_H2=self.Hydrogen.hmass()
        
        # Calculations for the design conditions
        self.N2_massflowrate=self.max_nitrogen
        self.H2_massflowrate=self.max_hydrogen
        self.N2_molarflowrate=self.N2_massflowrate/c.N2MOLMASS  # number of nitrogen moles per second
        self.H2_molarflowrate=self.H2_massflowrate/c.H2MOLMASS  # number of hydrogen moles per second
        self.N2_molarfraction=self.N2_molarflowrate/(self.N2_molarflowrate+self.H2_molarflowrate)  # nitrogen molar fraction
        self.H2_molarfraction=self.H2_molarflowrate/(self.N2_molarflowrate+self.H2_molarflowrate)  # hydrogen molar fraction
        self.N2_massfraction=self.N2_massflowrate/(self.N2_massflowrate+self.H2_massflowrate)      # nitrogen mass fraction
        self.H2_massfraction=self.H2_massflowrate/(self.N2_massflowrate+self.H2_massflowrate)      # hydrogen mass fraction
        
        # Initialization of the thermodynamic points
        self.p_points = np.zeros(21)  # [bar]
        self.T_points = np.zeros(21)  # [K]
        self.h_points = np.zeros(21)  # [J/kg]
        self.m_points = np.zeros(21)  # [kg/s]
        self.points=[]   # name of the point
        
         

        # Compression up to p_react, two compressors, one for hydrogen and one for nitrogen
            
        # First compressor for nitrogen
        if self.nstages_N2_compressor==None:
             raise ValueError('Please add the number of the stages of the N2 compressor for the ASR. Note that N2 and H2 arrive from PSA and electrolyzer at a different pressure.')
        self.beta_N2 = self.p_react/self.p_N2  # compression ratio
        self.beta_stage_N2=self.beta_N2**(1/self.nstages_N2_compressor)  # stage compression ratio
                  
        # Nitrogen inlet conditions point 0
        i=0
        self.points.append('N2 inlet')
        self.T_points[i]=self.T_N2
        self.p_points[i]=self.p_N2
        self.h_points[i]=self.h_N2
        self.m_points[i]=self.N2_massflowrate
        
        
        
        # Compression stages with interrefrigeration
        self.gamma_N2 = self.cp_N2/self.cv_N2
        self.eps_N2 = (self.gamma_N2-1)/self.gamma_N2
        self.eta_iso_N2_design = (self.beta_stage_N2**self.eps_N2 - 1)/(self.beta_stage_N2**(self.eps_N2/self.eta_pol)-1)
        self.N2_compressor_specific_work=np.zeros(self.nstages_N2_compressor)
        self.N2_compressor_work=np.zeros(self.nstages_N2_compressor)
        self.IC_specific_heat_N2=np.zeros(self.nstages_N2_compressor)            
        
        for n in range(self.nstages_N2_compressor):
            
            # Interrefrigeration,it starts before the first stage if the gas is already hot
            if self.T_points[i]>self.T_IC:
                i+=1
                self.points.append('N2 Cooler')
                self.T_points[i]=self.T_IC
                self.p_points[i]=self.p_points[i-1] 
                self.Nitrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
                self.h_points[i]=self.Nitrogen.hmass()
                self.m_points[i]=self.m_points[i-1]
                
                self.IC_specific_heat_N2[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] heat removed
            
            # Compression
            i+=1
            self.points.append('N2 Compressor')
            self.p_points[i]=self.p_points[i-1]*self.beta_stage_N2
            s_iso=self.Nitrogen.smass()
            T_iso=T_ps(self.Nitrogen, self.p_points[i], self.T_points[i-1], s_iso)
            self.Nitrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
            h_iso=self.Nitrogen.hmass()
            self.h_points[i]=(h_iso - self.h_points[i-1]) / self.eta_iso_N2_design + self.h_points[i-1]
            self.T_points[i]=T_ph(self.Nitrogen, self.p_points[i], T_iso, self.h_points[i])
            self.m_points[i]=self.m_points[i-1]
            
            self.N2_compressor_specific_work[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
            self.N2_compressor_work[n]=self.N2_compressor_specific_work[n]*self.m_points[i]/self.eta_motor   # [kW]
                
        T_after_compression_N2=self.T_points[i]
        h_after_compression_N2=self.h_points[i]
            
            
            
        # Second compressor for hydrogen
        if self.nstages_H2_compressor==None:
             raise ValueError('Please add the number of the stages of the H2 compressor for the ASR. Note that N2 and H2 arrive from PSA and electrolyzer at a different pressure.')
        self.beta_H2 = self.p_react/self.p_H2  # compression ratio
        self.beta_stage_H2=self.beta_H2**(1/self.nstages_H2_compressor)  # stage compression ratio
        
        # Hydrogen inlet conditions
        i+=1
        self.points.append('H2 inlet')
        self.T_points[i]=self.T_H2
        self.p_points[i]=self.p_H2
        self.h_points[i]=self.h_H2
        self.m_points[i]=self.H2_massflowrate
       
        # Compression stages with interrefrigeration
        self.gamma_H2 = self.cp_H2/self.cv_H2
        self.eps_H2 = (self.gamma_H2-1)/self.gamma_H2
        self.eta_iso_H2_design = (self.beta_stage_H2**self.eps_H2 - 1)/(self.beta_stage_H2**(self.eps_H2/self.eta_pol)-1)
        self.H2_compressor_specific_work=np.zeros(self.nstages_H2_compressor)
        self.H2_compressor_work=np.zeros(self.nstages_H2_compressor)
        self.IC_specific_heat_H2=np.zeros(self.nstages_H2_compressor)
        
        for n in range(self.nstages_H2_compressor):
            
            # Interrefrigeration,it starts before the first stage if the gas is already hot
            if self.T_points[i]>self.T_IC:
                i+=1
                self.points.append('H2 Cooler')
                self.T_points[i]=self.T_IC
                self.p_points[i]=self.p_points[i-1]
                self.Hydrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
                self.h_points[i]=self.Hydrogen.hmass()
                self.m_points[i]=self.m_points[i-1]
                
                self.IC_specific_heat_H2[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] heat removed
         
            # Compression
            i+=1
            self.points.append('H2 Compressor')
            self.p_points[i]=self.p_points[i-1]*self.beta_stage_H2
            s_iso=self.Hydrogen.smass()
            T_iso=T_ps(self.Hydrogen, self.p_points[i], self.T_points[i-1], s_iso)
            self.Hydrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
            h_iso=self.Hydrogen.hmass()
            self.h_points[i]=(h_iso - self.h_points[i-1]) / self.eta_iso_H2_design + self.h_points[i-1]
            self.T_points[i]=T_ph(self.Hydrogen, self.p_points[i], T_iso, self.h_points[i])
            self.m_points[i]=self.m_points[i-1]
            
            self.H2_compressor_specific_work[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
            self.H2_compressor_work[n]=self.H2_compressor_specific_work[n]*self.m_points[i]/self.eta_motor   # [kW]
                
        T_after_compression_H2=self.T_points[i]
        h_after_compression_H2=self.h_points[i]


 
        # Mixing of the reactants
        i+=1
        self.points.append('Mixing')
        self.mix_H2N2.set_mole_fractions([self.N2_molarfraction, self.H2_molarfraction])
        self.p_points[i]=self.p_points[i-1]
        self.h_points[i]=(h_after_compression_N2 * self.N2_massflowrate + h_after_compression_H2 * self.H2_massflowrate) / (self.N2_massflowrate + self.H2_massflowrate)
        self.T_points[i]=T_ph(self.mix_H2N2, self.p_points[i], (T_after_compression_H2+T_after_compression_N2)/2 , self.h_points[i])
        self.m_points[i]=self.N2_massflowrate+self.H2_massflowrate
        
        h_fresh_feed=self.h_points[i]
        T_fresh_feed=self.T_points[i]
            
            
            
        # Convergence of the recycle loop using Brent method
        # Fresh reactants
        new_N2_massflowrate = self.N2_massflowrate
        new_H2_massflowrate = self.H2_massflowrate
        # Calculations for the condensation step
        self.N2_loop_molarfraction, self.H2_loop_molarfraction, self.NH3_loop_molarfraction = x_cond(self.mix_H2N2NH3, self.Ammonia, self.T_cond, self.p_react, self.N2_molarfraction, self.H2_molarfraction)
        self.NH3_loop_massfraction = self.NH3_loop_molarfraction * c.NH3MOLMASS / (self.N2_loop_molarfraction * c.N2MOLMASS + self.H2_loop_molarfraction * c.H2MOLMASS + self.NH3_loop_molarfraction * c.NH3MOLMASS)

        def loop_equations(massflowrate_loop):
            
            # Composition of the flowrate inside the loop (thanks to the calculations of the condensation - VLE equilibirum)
            ammonia_loop = massflowrate_loop * self.NH3_loop_massfraction
            nitro_loop = (massflowrate_loop - ammonia_loop)  * self.N2_massfraction
            hyd_loop = (massflowrate_loop - ammonia_loop) * self.H2_massfraction
            
            # Reactor inlet flowrates
            massflowrate_reactor = new_N2_massflowrate + new_H2_massflowrate + massflowrate_loop
            massflowrate_N2_in = new_N2_massflowrate + nitro_loop
            self.molarflowrate_N2_in = massflowrate_N2_in / c.N2MOLMASS
            massflowrate_H2_in = new_H2_massflowrate + hyd_loop
            self.molarflowrate_H2_in = massflowrate_H2_in / c.H2MOLMASS
            massflowrate_NH3_in = ammonia_loop  
            self.molarflowrate_NH3_in = massflowrate_NH3_in / c.NH3MOLMASS
            
            # Reactor model - calculations of the three beds
            XN2_Ibed = 0
            self.T_Ibed, self.N2_molarflowrate_Ibed, self.H2_molarflowrate_Ibed, self.NH3_molarflowrate_Ibed, self.XN2_Ibed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, self.molarflowrate_N2_in, self.molarflowrate_H2_in, self.molarflowrate_NH3_in, self.molarflowrate_N2_in, self.T_in_Ibed, XN2_Ibed, self.p_react, 50, self.reactor_Ibed_volume)
            self.T_IIbed, self.N2_molarflowrate_IIbed, self.H2_molarflowrate_IIbed, self.NH3_molarflowrate_IIbed, self.XN2_IIbed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, self.N2_molarflowrate_Ibed[-1], self.H2_molarflowrate_Ibed[-1], self.NH3_molarflowrate_Ibed[-1], self.molarflowrate_N2_in, self.T_in_IIbed, self.XN2_Ibed[-1], self.p_react, 50, self.reactor_IIbed_volume)
            self.T_IIIbed, self.N2_molarflowrate_IIIbed, self.H2_molarflowrate_IIIbed, self.NH3_molarflowrate_IIIbed, self.XN2_IIIbed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, self.N2_molarflowrate_IIbed[-1], self.H2_molarflowrate_IIbed[-1], self.NH3_molarflowrate_IIbed[-1], self.molarflowrate_N2_in, self.T_in_IIIbed, self.XN2_IIbed[-1], self.p_react, 50, self.reactor_IIIbed_volume)
            
            # Reactor outlet flowrates
            self.molarflowrate_NH3_out=self.NH3_molarflowrate_IIIbed[-1]    
            massflowrate_NH3_out=self.molarflowrate_NH3_out*c.NH3MOLMASS
            self.molarflowrate_N2_out=self.N2_molarflowrate_IIIbed[-1]
            massflowrate_N2_out=self.molarflowrate_N2_out*c.N2MOLMASS
            self.molarflowrate_H2_out=self.H2_molarflowrate_IIIbed[-1]
            massflowrate_H2_out=self.molarflowrate_H2_out*c.H2MOLMASS

            # Loop flowrates
            nitro_loop = massflowrate_N2_out
            hyd_loop = massflowrate_H2_out
            new_loop = (nitro_loop+hyd_loop)/(1-self.NH3_loop_massfraction)
            ammonia_loop = new_loop * self.NH3_loop_massfraction 
            
            return new_loop - massflowrate_loop  
        
        self.loop_massflowrate = brentq(loop_equations, self.m_points[i], 20*self.m_points[i],  xtol=1e-5, maxiter=100)
        if np.max(self.T_Ibed) > 800 or np.max(self.T_IIbed) > 800 or np.max(self.T_IIIbed) > 800:
            print("The temperature inside the bed reaches values ​​beyond the maximum limit of 800 K")
        self.loop_massflowrate_design = self.loop_massflowrate
        #print(f'The loop mass flowrate is {round(self.loop_massflowrate/(self.N2_massflowrate+self.H2_massflowrate),2)} times of the make up gas') 
        
        
        def find_equilibrium(molarflowrate_N2_in, molarflowrate_H2_in, molarflowrate_NH3_in, T_in_bed, XN2_bed, initial_volume):
            volume_increase_step=initial_volume/50
            tol=1e-6
            max_iter=200
            reactor_volume = initial_volume
            for iteration in range(max_iter):
                massflowrate_reactor=self.loop_massflowrate+self.N2_massflowrate+self.H2_massflowrate
                _, N2_molarflowrate_Ibed, H2_molarflowrate_Ibed, NH3_molarflowrate_Ibed, XN2 = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, molarflowrate_N2_in, molarflowrate_H2_in, molarflowrate_NH3_in, self.molarflowrate_N2_in, T_in_bed, XN2_bed, self.p_react, 50, reactor_volume)
                if XN2[-1] - XN2[-2] < tol:
                    NH3_concentration_equilibrium = NH3_molarflowrate_Ibed[-1] / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1])
                    return NH3_concentration_equilibrium, XN2[-1]
                reactor_volume += volume_increase_step
                
        # NH3_concentration_equilibrium_Ibed, XN2_equilibrium_Ibed = find_equilibrium(self.molarflowrate_N2_in, self.molarflowrate_H2_in, self.molarflowrate_NH3_in, self.T_in_Ibed, 0, self.reactor_Ibed_volume)
        # NH3_concentration_Ibed = self.NH3_molarflowrate_Ibed[-1] / (self.N2_molarflowrate_Ibed[-1] + self.H2_molarflowrate_Ibed[-1] + self.NH3_molarflowrate_Ibed[-1])
        # NH3_concentration_equilibrium_IIbed, XN2_equilibrium_IIbed = find_equilibrium(self.N2_molarflowrate_Ibed[-1], self.H2_molarflowrate_Ibed[-1], self.NH3_molarflowrate_Ibed[-1], self.T_in_IIbed, self.XN2_Ibed[-1], self.reactor_IIbed_volume)
        # NH3_concentration_IIbed = self.NH3_molarflowrate_IIbed[-1] / (self.N2_molarflowrate_IIbed[-1] + self.H2_molarflowrate_IIbed[-1] + self.NH3_molarflowrate_IIbed[-1])
        # NH3_concentration_equilibrium_IIIbed, XN2_equilibrium_IIIbed = find_equilibrium(self.N2_molarflowrate_IIbed[-1], self.H2_molarflowrate_IIbed[-1], self.NH3_molarflowrate_IIbed[-1], self.T_in_IIIbed, self.XN2_IIbed[-1], self.reactor_IIIbed_volume)
        # NH3_concentration_IIIbed = self.NH3_molarflowrate_IIIbed[-1] / (self.N2_molarflowrate_IIIbed[-1] + self.H2_molarflowrate_IIIbed[-1] + self.NH3_molarflowrate_IIIbed[-1])
        # print(NH3_concentration_Ibed/NH3_concentration_equilibrium_Ibed, NH3_concentration_IIbed/NH3_concentration_equilibrium_IIbed, NH3_concentration_IIIbed/NH3_concentration_equilibrium_IIIbed)
        # print(self.XN2_Ibed[-1]/XN2_equilibrium_Ibed, self.XN2_IIbed[-1]/XN2_equilibrium_IIbed, self.XN2_IIIbed[-1]/XN2_equilibrium_IIIbed)
        
    
        # Loop
        i+=1
        self.points.append('Loop')
        self.p_points[i]= self.p_react - self.loop_pressure_drop  
        self.T_points[i]=self.T_cond
        self.mix_H2N2NH3.set_mole_fractions([self.N2_loop_molarfraction, self.H2_loop_molarfraction, self.NH3_loop_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.loop_massflowrate
        
        
        
        # Loop compressor
        self.beta_loop=self.p_react/self.p_points[i]
        self.Cp_mix= self.mix_H2N2NH3.cpmass()
        self.Cv_mix=self.mix_H2N2NH3.cvmass()
        self.gamma = self.Cp_mix/self.Cv_mix
        self.eps = (self.gamma-1)/self.gamma
        self.eta_iso_loop_design = (self.beta_loop**self.eps - 1)/(self.beta_loop**(self.eps/self.eta_pol)-1)
        i+=1
        self.points.append('Loop compressor')
        self.p_points[i]=self.p_points[i-1]*self.beta_loop
        s_iso=self.mix_H2N2NH3.smass()
        T_iso=T_ps(self.mix_H2N2NH3, self.p_points[i], self.T_points[i-1], s_iso)  
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
        h_iso=self.mix_H2N2NH3.hmass()
        self.h_points[i]=(h_iso - self.h_points[i-1]) /self.eta_iso_loop_design + self.h_points[i-1]
        self.T_points[i]=T_ph(self.mix_H2N2NH3, self.p_points[i], T_iso, self.h_points[i])
        self.m_points[i]=self.m_points[i-1]
                        
        self.specific_work_loop = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
        self.loop_compressor_work = self.specific_work_loop*self.m_points[i]/self.eta_motor  # [kW]
        self.loop_compressor_is_head_design = h_iso - self.h_points[i-1]
        
        

        # Mixing of the fresh feed with the loop flowrate
        i+=1
        self.points.append('Mixing') 
        self.m_points[i]=self.N2_massflowrate+self.H2_massflowrate+self.loop_massflowrate
        self.NH3_reactor_in_molarfraction=self.molarflowrate_NH3_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.N2_reactor_in_molarfraction=self.molarflowrate_N2_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.H2_reactor_in_molarfraction=self.molarflowrate_H2_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.mix_H2N2NH3.set_mole_fractions([self.N2_reactor_in_molarfraction, self.H2_reactor_in_molarfraction, self.NH3_reactor_in_molarfraction])
        self.p_points[i]=self.p_points[i-1]
        self.h_points[i]=(h_fresh_feed*(self.N2_massflowrate+self.H2_massflowrate)+self.h_points[i-1]*self.loop_massflowrate) / (self.N2_massflowrate+self.H2_massflowrate+self.loop_massflowrate)
        self.T_points[i]=T_ph(self.mix_H2N2NH3, self.p_points[i], (T_fresh_feed+self.T_points[i-1])/2 , self.h_points[i])
        
        self.cp_cold_design = self.mix_H2N2NH3.cpmass()
        components = ["H2", "N2", "NH3"]
        molar_fractions = [self.N2_reactor_in_molarfraction, self.H2_reactor_in_molarfraction, self.NH3_reactor_in_molarfraction]
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        self.mu_cold_design = 1 / sum([molar_fractions[i] / viscosities[i] for i in range(len(components))]) # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        self.k_cold_design = np.dot(molar_fractions, conductivities)   # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        
        
        
        # Heating up to T_react
        i+=1
        self.points.append('RHE + Heater')
        self.p_points[i]=self.p_points[i-1] 
        self.T_points[i]=self.T_in_Ibed
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1]
        
        h_reactor_in = self.h_points[i]
    
        
        
        # This heat can be recovered from the cooling of the reactor products
        # Creation of the thermodinamic state for the hot mixture N2/H2/NH3 with Peng-Robinson cubic EOS        
        self.mix_H2N2NH3_aux = CP.AbstractState('PR',"Nitrogen&Hydrogen&Ammonia") 
        self.mix_H2N2NH3_aux.set_binary_interaction_double(0,1,"kij",-0.036)   # from UNISIM
        self.mix_H2N2NH3_aux.set_binary_interaction_double(0,2,"kij",0.222)    # from UNISIM
        self.mix_H2N2NH3_aux.set_binary_interaction_double(1,2,"kij",0.000)    # from UNISIM
        self.mix_H2N2NH3_aux.specify_phase(CP.iphase_gas)
        self.NH3_reactor_out_molarfraction=self.molarflowrate_NH3_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.N2_reactor_out_molarfraction=self.molarflowrate_N2_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.H2_reactor_out_molarfraction=self.molarflowrate_H2_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.mix_H2N2NH3_aux.set_mole_fractions([self.N2_reactor_out_molarfraction, self.H2_reactor_out_molarfraction, self.NH3_reactor_out_molarfraction])

        # Calculation of the heat that can be recovered  
        check_pinch_point = False
        delta_T = 10
        while check_pinch_point == False:
          T_hot_RHE, T_cold_RHE, Q_RHE, self.A_RHE, self.U_RHE_design, self.epsilon_RHE, pinch_point = profile_RHE(self.mix_H2N2NH3_aux, self.mix_H2N2NH3, 50, self.m_points[i], self.T_IIIbed[-1], self.p_react, self.T_points[i-1], self.T_IIIbed[-1]-delta_T, self.p_react)
          if pinch_point < 20:
              delta_T += 5
          else:
              check_pinch_point = True
        # RHE_plot(T_hot_RHE, T_cold_RHE, Q_RHE)
        if T_cold_RHE[-1] >= self.T_in_Ibed:
           self.electric_resistance_heat = 0 # the heat is all taken from the RHE 
        else:
            h_cold_out_RHE = self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, T_cold_RHE[-1])
            self.reactor_in_specific_heat=(h_reactor_in-h_cold_out_RHE)/1000   # [kJ/kg]
            self.electric_resistance_heat = self.reactor_in_specific_heat * self.m_points[i]


        
        # Reactor products
        i+=1
        self.points.append('Reactor')
        self.p_points[i]=self.p_react
        self.T_points[i]=self.T_IIIbed[-1]  # adiabatic reactor
        self.mix_H2N2NH3.set_mole_fractions([self.N2_reactor_out_molarfraction, self.H2_reactor_out_molarfraction, self.NH3_reactor_out_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1]
        self.massflowrate_reactor_design = self.m_points[i]
        
        self.cp_hot_design = self.mix_H2N2NH3.cpmass()
        molar_fractions = [self.N2_reactor_out_molarfraction, self.H2_reactor_out_molarfraction, self.NH3_reactor_out_molarfraction]
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        self.mu_hot_design = 1 / sum([molar_fractions[i] / viscosities[i] for i in range(len(components))]) # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        self.k_hot_design = np.dot(molar_fractions, conductivities)  # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture



        # Ammonia condensation
         
        # First part - cooled in part with the flowrate entering the reactor to recover heat 
        i+=1
        self.points.append('RHE')
        self.p_points[i]=self.p_points[i-1] 
        self.T_points[i]=T_hot_RHE[-1]   # This is not the real temperature during design conditions, because the RHE is overdimensioned, and some massflowrate will bypass it, the real temperature is inside consumption method
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1]
        
        
        
        # Second part - with the ambient (water)
        i+=1
        self.points.append('Condenser 1')
        self.p_points[i]=self.p_points[i-1]
        self.T_points[i]=self.T_IC
        self.m_points[i]=self.m_points[i-1]
        
        # Gas phase
        N2gas_Icondenser_molarfraction, H2gas_Icondenser_molarfraction, NH3gas_Icondenser_molarfraction = x_cond (self.mix_H2N2NH3, self.Ammonia, self.T_points[i], self.p_points[i], self.N2_molarfraction, self.H2_molarfraction)
        self.mix_H2N2NH3.set_mole_fractions([N2gas_Icondenser_molarfraction, H2gas_Icondenser_molarfraction, NH3gas_Icondenser_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        h_gas_Icondenser=self.mix_H2N2NH3.hmass()
        
        # Liquid phase
        self.N2_reactor_out_molarflowrate=self.molarflowrate_N2_out
        gas_reactor_out_molarflowrate=self.N2_reactor_out_molarflowrate/self.N2_reactor_out_molarfraction  # calculation of the total molar flowrate from the N2 molar flowrate and it's fraction
        gas_Icondenser_molarflowrate=self.N2_reactor_out_molarflowrate/N2gas_Icondenser_molarfraction  # the molar flowrate of N2 inside the gas remains constant, it doesnt'condens
        NH3liq_Icondenser_molarflowrate=self.NH3_reactor_out_molarfraction*gas_reactor_out_molarflowrate-NH3gas_Icondenser_molarfraction*gas_Icondenser_molarflowrate
        NH3liq_Icondenser_massflowrate=NH3liq_Icondenser_molarflowrate*c.NH3MOLMASS
        self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        h_liq_Icondenser=self.Ammonia.hmass()       
        
        gas_Icondenser_massflowrate=self.m_points[i]-NH3liq_Icondenser_massflowrate
        self.h_points[i]=(NH3liq_Icondenser_massflowrate*h_liq_Icondenser+gas_Icondenser_massflowrate*h_gas_Icondenser)/self.m_points[i]
        
        self.Icondenser_specific_heat=(self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg]

        
        
        # Third part - with a refrigeration cycle
        if self.T_cond < self.T_IC:
            i+=1
            self.points.append('Condenser 2')
            self.p_points[i]=self.p_points[i-1]
            self.T_points[i]=self.T_cond
            self.m_points[i]=self.m_points[i-1]
            
            # The gas phase is the one inside the loop 
            self.mix_H2N2NH3.set_mole_fractions([self.N2_loop_molarfraction, self.H2_loop_molarfraction, self.NH3_loop_molarfraction])
            self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
            h_gas_IIcondenser=self.mix_H2N2NH3.hmass()
            
            # Liquid phase
            self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
            h_liq_IIcondenser=self.Ammonia.hmass()
            NH3liq_IIcondenser_massflowrate=self.m_points[i]-self.loop_massflowrate
            
            self.h_points[i]=(NH3liq_IIcondenser_massflowrate*h_liq_IIcondenser+self.loop_massflowrate*h_gas_IIcondenser)/self.m_points[i]
            
            self.IIcondenser_specific_heat=(self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg])
            self.IIcondenser_heat_design = self.IIcondenser_specific_heat * self.m_points[i]
            self.refrigerationcycle_work = -self.IIcondenser_specific_heat*self.m_points[i]/self.COP_design/self.eta_motor # [kW]
        else:
            self.refrigerationcycle_work = 0

        
        
        # Separator
        i+=1
        self.points.append('Separator-product')
        self.p_points[i]=self.p_points[i-1] 
        self.T_points[i]=self.T_points[i-1]
        self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.Ammonia.hmass()
        self.m_points[i]=self.m_points[i-1]-self.loop_massflowrate    
       
         
        
        # Calculation of the total electricity consumed
        if self.p_N2==self.p_H2: 
                    
            self.electricity_design=self.refrigerationcycle_work+np.sum(self.feed_compressor_work)+self.loop_compressor_work
                        
        if self.p_N2!=self.p_H2:
          
            self.electricity_design=self.refrigerationcycle_work+np.sum(self.N2_compressor_work)+np.sum(self.H2_compressor_work)+self.loop_compressor_work

   

    def reactor_dimensioning(self):
        
        # To obtain the same conversion at the end of the beds as https://doi.org/10.4236/aces.2018.83009
        
        if self.p_react != 150:
            raise ValueError("Automatic dimensioning is possible just for reactor operating pressure of 150 bar")
        if self.T_react != [663.15,693.15, 683.15]:
            raise ValueError("Automatic dimensioning is possible just for the inlet bed temperature values [663.15,693.15, 683.15] K")
        if self.T_cond != 263.15:
            raise ValueError("Automatic dimensioning is possible just for the condenser temperature value 263.15 K")
            
        size_ref = 1 # [kg/s], nominal ammonia production reference case
        reactor_beds_volume_ref = [0.9, 1.25, 1.55] # [m^3]
        
        scaling_factor = self.max_ammonia / size_ref
        reactor_beds_volume = [volume * scaling_factor for volume in reactor_beds_volume_ref]
        
        return reactor_beds_volume



    def consumption(self, hyd, hyd_buffer=False, p_buffer=False):
        """
        The consumption function calculates the energy consumption in the operating range. If there is a part of hydrogen coming from the 
        pressurized tank, it is subtracted from the total massflowrate of hydrogen compressed. In every operating point the recycle massflowrate 
        changes due to different nitrogen conversion. Moreover the off-design operation of compressors, refrigeration cycle and feed-effluent 
        heat exchanger is taken into account with modified efficiencies. A cache is used in the convergence of the loop if it has already been 
        calculated for that value of hydrogen.
            
        Inputs:
            hyd: (>0) total hydrogen mass flowrate [kg/s]
            hyd_buffer: (>0) part of hydrogen mass flowrate from the buffer [kg/s]
            p_buffer: pressure of the hydrogen coming from the buffer [bar]
            
        Outputs:
            electricity: electricity consumption [kW]  
        """
            
        self.H2_massflowrate = hyd
        self.N2_massflowrate = hyd * c.N2MOLMASS / (3 * c.H2MOLMASS)
        
        # Initialization of the thermodynamic points
        self.p_points = np.zeros(21)  # [bar]
        self.T_points = np.zeros(21)  # [K]
        self.h_points = np.zeros(21)  # [J/kg]
        self.m_points = np.zeros(21)  # [kg/s]
        self.points=[]   # name of the point
                      
        # Nitrogen inlet conditions point 0
        i=0
        self.points.append('N2 inlet')
        self.T_points[i]=self.T_N2
        self.p_points[i]=self.p_N2
        self.h_points[i]=self.h_N2
        self.m_points[i]=self.N2_massflowrate
        self.Nitrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        
        
        
        # Compression stages with interrefrigeration
        self.N2_compressor_specific_work=np.zeros(self.nstages_N2_compressor)
        self.N2_compressor_work=np.zeros(self.nstages_N2_compressor)
        self.IC_specific_heat_N2=np.zeros(self.nstages_N2_compressor)            
        
        for n in range(self.nstages_N2_compressor):
            
            # Interrefrigeration,it starts before the first stage if the gas is already hot
            if self.T_points[i]>self.T_IC:
                i+=1
                self.points.append('N2 Cooler')
                self.T_points[i]=self.T_IC
                self.p_points[i]=self.p_points[i-1] 
                self.Nitrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
                self.h_points[i]=self.Nitrogen.hmass()
                self.m_points[i]=self.m_points[i-1]
                
                self.IC_specific_heat_N2[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] heat removed
            
            # Compression
            i+=1
            self.points.append('N2 Compressor')
            self.m_points[i]=self.m_points[i-1]
            self.eta_iso_N2 = part_load_eta(self.m_points[i]/self.max_nitrogen) * self.eta_iso_N2_design  # reduced efficiency at part - load
            self.p_points[i]=self.p_points[i-1]*self.beta_stage_N2
            s_iso=self.Nitrogen.smass()
            T_iso=T_ps(self.Nitrogen, self.p_points[i], self.T_points[i-1], s_iso)
            self.Nitrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
            h_iso=self.Nitrogen.hmass()
            self.h_points[i]=(h_iso - self.h_points[i-1]) / self.eta_iso_N2 + self.h_points[i-1]
            self.T_points[i]=T_ph(self.Nitrogen, self.p_points[i], T_iso, self.h_points[i])
            
            self.N2_compressor_specific_work[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
            self.N2_compressor_work[n]=self.N2_compressor_specific_work[n]*self.m_points[i]/self.eta_motor   # [kW]
                
        T_after_compression_N2=self.T_points[i]
        h_after_compression_N2=self.h_points[i]
            
            
        
        # Hydrogen inlet conditions
        i+=1
        self.points.append('H2 inlet')
        self.T_points[i]=self.T_H2
        self.p_points[i]=self.p_H2
        self.h_points[i]=self.h_H2
        if hyd_buffer != False:
           self.m_points[i]=self.H2_massflowrate - hyd_buffer
        else:
            self.m_points[i]=self.H2_massflowrate
        self.Hydrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
       
        # Compression stages with interrefrigeration
        self.H2_compressor_specific_work=np.zeros(self.nstages_H2_compressor)
        self.H2_compressor_work=np.zeros(self.nstages_H2_compressor)
        self.IC_specific_heat_H2=np.zeros(self.nstages_H2_compressor)
        
        if self.buffer_info != False and self.buffer_info[2] < self.p_H2:
            raise ValueError("The minimum pressure of the hydrogen buffer must be at least the same as the one of the electrolyzer")
        
        check = False
        for n in range(self.nstages_H2_compressor):
            
            if (hyd_buffer != False or hyd_buffer > 0) and p_buffer < round(self.p_points[i]*self.beta_stage_H2, 2) and check == False:
                check = True
                self.m_points[i] = self.m_points[i] + hyd_buffer
                # Inlet of H2 from the buffer if the pressure is less than the one at the exit of this stage
                T_buffer = self.buffer_info[0]
                self.Hydrogen.update(CP.PT_INPUTS, p_buffer*100000, T_buffer)
                h_inlet = self.Hydrogen.hmass()
                # Lamination of H2 at the inlet pressure of the stage, it needs to be the nominal one
                h_lamination = h_inlet
                # Mixing of H2 from the buffer and from the electrolyzer 
                hyd_el = self.H2_massflowrate - hyd_buffer  # hydrogen from the electrolyzer
                self.h_points[i]=(self.h_points[i] * hyd_el + h_lamination * hyd_buffer) / self.H2_massflowrate
                self.T_points[i]=T_ph(self.Hydrogen, self.p_points[i], self.T_points[i], self.h_points[i])
                
            
            # Interrefrigeration, it starts before the first stage if the gas is already hot
            if self.T_points[i]>self.T_IC:
                i+=1
                self.points.append('H2 Cooler')
                self.T_points[i]=self.T_IC
                self.p_points[i]=self.p_points[i-1]
                self.Hydrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
                self.h_points[i]=self.Hydrogen.hmass()
                self.m_points[i]=self.m_points[i-1]
                
                self.IC_specific_heat_H2[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] heat removed
         
            # Compression
            i+=1
            self.points.append('H2 Compressor')
            self.m_points[i]=self.m_points[i-1]
            self.eta_iso_H2 = part_load_eta(self.m_points[i]/self.max_hydrogen) * self.eta_iso_H2_design  # reduced efficiency at part - load
            self.p_points[i]=self.p_points[i-1]*self.beta_stage_H2
            s_iso=self.Hydrogen.smass()
            T_iso=T_ps(self.Hydrogen, self.p_points[i], self.T_points[i-1], s_iso)
            self.Hydrogen.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
            h_iso=self.Hydrogen.hmass()
            self.h_points[i]=(h_iso - self.h_points[i-1]) / self.eta_iso_H2 + self.h_points[i-1]
            self.T_points[i]=T_ph(self.Hydrogen, self.p_points[i], T_iso, self.h_points[i])
            
            self.H2_compressor_specific_work[n] = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
            self.H2_compressor_work[n]=self.H2_compressor_specific_work[n]*self.m_points[i]/self.eta_motor
                
        T_after_compression_H2=self.T_points[i]
        h_after_compression_H2=self.h_points[i]


 
        # Mixing of the reactants
        i+=1
        self.points.append('Mixing')
        self.mix_H2N2.set_mole_fractions([self.N2_molarfraction, self.H2_molarfraction])
        self.p_points[i]=self.p_points[i-1]
        if (hyd_buffer != False or hyd_buffer) and p_buffer >= self.p_react:
            # Inlet of H2 from the buffer if it skips the compression
            T_buffer = self.buffer_info[0]
            self.Hydrogen.update(CP.PT_INPUTS, p_buffer*100000, T_buffer)
            h_inlet = self.Hydrogen.hmass()
            # Lamination of H2 at p_react
            h_lamination = h_inlet
            # Mixing of fresh reactants 
            hyd_el = self.H2_massflowrate - hyd_buffer  # hydrogen from the electrolyzer
            self.h_points[i]=(h_after_compression_N2 * self.N2_massflowrate + h_after_compression_H2 * hyd_el + h_lamination * hyd_buffer) / (self.N2_massflowrate + self.H2_massflowrate)
            self.T_points[i]=T_ph(self.mix_H2N2, self.p_react, (T_after_compression_H2+T_after_compression_N2)/2, self.h_points[i])
        else:
            self.h_points[i]=(h_after_compression_N2 * self.N2_massflowrate + h_after_compression_H2 * self.H2_massflowrate) / (self.N2_massflowrate + self.H2_massflowrate)
            self.T_points[i]=T_ph(self.mix_H2N2, self.p_points[i], (T_after_compression_H2+T_after_compression_N2)/2 , self.h_points[i])
        self.m_points[i]=self.N2_massflowrate+self.H2_massflowrate
        
        h_fresh_feed=self.h_points[i]
        T_fresh_feed=self.T_points[i] 
            

            
        # Convergence of the recycle loop using Brent method
        # Fresh reactants
        new_N2_massflowrate = self.N2_massflowrate
        new_H2_massflowrate = self.H2_massflowrate
        fresh_feed_massflowrate = new_N2_massflowrate + new_H2_massflowrate
        
        # If the loop has already been calculated for this value of hydrogen, it takes directly the results inside the cache
        hyd_key = format(hyd, ".4g")
        if hyd_key in self.cache_loop_massflowrate:
            self.loop_massflowrate = self.cache_loop_massflowrate[hyd_key]
            self.T_Ibed, self.T_IIbed, self.T_IIIbed = self.cache_T_beds[hyd_key] 
            self.molarflowrate_N2_in, self.molarflowrate_H2_in, self.molarflowrate_NH3_in = self.cache_molarflowrate_in[hyd_key]
            self.molarflowrate_N2_out, self.molarflowrate_H2_out, self.molarflowrate_NH3_out = self.cache_molarflowrate_out[hyd_key]
            
        else:
            def loop_equations(massflowrate_loop):
                
                # Composition of the flowrate inside the loop (thanks to the calculations of the condensation - VLE equilibirum)
                ammonia_loop = massflowrate_loop * self.NH3_loop_massfraction
                nitro_loop = (massflowrate_loop - ammonia_loop)  * self.N2_massfraction
                hyd_loop = (massflowrate_loop - ammonia_loop) * self.H2_massfraction
                
                # Reactor inlet flowrates
                massflowrate_reactor = new_N2_massflowrate + new_H2_massflowrate + massflowrate_loop
                massflowrate_N2_in = new_N2_massflowrate + nitro_loop
                self.molarflowrate_N2_in = massflowrate_N2_in / c.N2MOLMASS
                massflowrate_H2_in = new_H2_massflowrate + hyd_loop
                self.molarflowrate_H2_in = massflowrate_H2_in / c.H2MOLMASS
                massflowrate_NH3_in = ammonia_loop  
                self.molarflowrate_NH3_in = massflowrate_NH3_in / c.NH3MOLMASS
                
                # Reactor model - calculations of the three beds
                XN2_Ibed = 0
                self.T_Ibed, N2_molarflowrate_Ibed, H2_molarflowrate_Ibed, NH3_molarflowrate_Ibed, self.XN2_Ibed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, self.molarflowrate_N2_in, self.molarflowrate_H2_in, self.molarflowrate_NH3_in, self.molarflowrate_N2_in, self.T_in_Ibed, XN2_Ibed, self.p_react, 50, self.reactor_Ibed_volume)
                self.T_IIbed, N2_molarflowrate_IIbed, H2_molarflowrate_IIbed, NH3_molarflowrate_IIbed, self.XN2_IIbed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, N2_molarflowrate_Ibed[-1], H2_molarflowrate_Ibed[-1], NH3_molarflowrate_Ibed[-1], self.molarflowrate_N2_in, self.T_in_IIbed, self.XN2_Ibed[-1], self.p_react, 50, self.reactor_IIbed_volume)
                self.T_IIIbed, N2_molarflowrate_IIIbed, H2_molarflowrate_IIIbed, NH3_molarflowrate_IIIbed, self.XN2_IIIbed = reactor_bed(self.mix_H2N2NH3, massflowrate_reactor, N2_molarflowrate_IIbed[-1], H2_molarflowrate_IIbed[-1], NH3_molarflowrate_IIbed[-1], self.molarflowrate_N2_in, self.T_in_IIIbed, self.XN2_IIbed[-1], self.p_react, 50, self.reactor_IIIbed_volume)
                
                # Reactor outlet flowrates
                self.molarflowrate_NH3_out=NH3_molarflowrate_IIIbed[-1]    
                massflowrate_NH3_out=self.molarflowrate_NH3_out*c.NH3MOLMASS
                self.molarflowrate_N2_out=N2_molarflowrate_IIIbed[-1]
                massflowrate_N2_out=self.molarflowrate_N2_out*c.N2MOLMASS
                self.molarflowrate_H2_out=H2_molarflowrate_IIIbed[-1]
                massflowrate_H2_out=self.molarflowrate_H2_out*c.H2MOLMASS

                # Loop flowrates
                nitro_loop = massflowrate_N2_out
                hyd_loop = massflowrate_H2_out
                new_loop = (nitro_loop+hyd_loop)/(1-self.NH3_loop_massfraction)
                ammonia_loop = new_loop * self.NH3_loop_massfraction  
                
                return new_loop - massflowrate_loop

            self.loop_massflowrate = brentq(loop_equations, fresh_feed_massflowrate, 7*fresh_feed_massflowrate,  xtol=1e-5, maxiter=100) 
            self.cache_loop_massflowrate[hyd_key] = self.loop_massflowrate
            self.cache_T_beds[hyd_key] = self.T_Ibed, self.T_IIbed, self.T_IIIbed
            self.cache_molarflowrate_in[hyd_key] = self.molarflowrate_N2_in, self.molarflowrate_H2_in, self.molarflowrate_NH3_in
            self.cache_molarflowrate_out[hyd_key] = self.molarflowrate_N2_out, self.molarflowrate_H2_out, self.molarflowrate_NH3_out

        if (np.max(self.T_Ibed) or np.max(self.T_IIbed) or np.max(self.T_IIIbed)) > 800:
            print("The temperature inside the bed reaches values ​​beyond the maximum limit of 800 K")        


        
        # Loop
        self.loop_pressure_drop_scaled = self.loop_pressure_drop * ((self.loop_massflowrate+self.N2_massflowrate+self.H2_massflowrate) / self.massflowrate_reactor_design)**2
        i+=1
        self.points.append('Loop')
        self.p_points[i]= self.p_react - self.loop_pressure_drop_scaled
        self.T_points[i]=self.T_cond
        self.mix_H2N2NH3.set_mole_fractions([self.N2_loop_molarfraction, self.H2_loop_molarfraction, self.NH3_loop_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.loop_massflowrate
        
        
        
        # Loop compressor
        i+=1
        self.points.append('Loop compressor')
        self.p_points[i] = self.p_react
        self.m_points[i]=self.m_points[i-1]
        s_iso=self.mix_H2N2NH3.smass()
        T_iso=T_ps(self.mix_H2N2NH3, self.p_points[i], self.T_points[i-1], s_iso)  
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, T_iso)
        h_iso=self.mix_H2N2NH3.hmass()
        self.eta_iso_loop = part_load_eta_loop((h_iso - self.h_points[i-1])/self.loop_compressor_is_head_design, self.m_points[i]/self.loop_massflowrate_design) * self.eta_iso_loop_design
        self.h_points[i]=(h_iso - self.h_points[i-1]) / self.eta_iso_loop + self.h_points[i-1]
        self.T_points[i]=T_ph(self.mix_H2N2NH3, self.p_points[i], T_iso, self.h_points[i])
                        
        self.specific_work_loop = (self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg] 
        self.loop_compressor_work = self.specific_work_loop*self.m_points[i]/self.eta_motor  # [kW]
        


        # Mixing of the fresh feed with the loop flowrate
        i+=1
        self.points.append('Mixing') 
        self.m_points[i]=self.N2_massflowrate+self.H2_massflowrate+self.loop_massflowrate
        self.NH3_reactor_in_molarfraction=self.molarflowrate_NH3_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.N2_reactor_in_molarfraction=self.molarflowrate_N2_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.H2_reactor_in_molarfraction=self.molarflowrate_H2_in/(self.molarflowrate_NH3_in+self.molarflowrate_N2_in+self.molarflowrate_H2_in)
        self.mix_H2N2NH3.set_mole_fractions([self.N2_reactor_in_molarfraction, self.H2_reactor_in_molarfraction, self.NH3_reactor_in_molarfraction])
        self.p_points[i]=self.p_points[i-1]
        self.h_points[i]=(h_fresh_feed*(self.N2_massflowrate+self.H2_massflowrate)+self.h_points[i-1]*self.loop_massflowrate) / (self.N2_massflowrate+self.H2_massflowrate+self.loop_massflowrate)
        self.T_points[i]=T_ph(self.mix_H2N2NH3, self.p_points[i], (T_fresh_feed+self.T_points[i-1])/2 , self.h_points[i])
        
        components = ["H2", "N2", "NH3"]
        molar_fractions = [self.N2_reactor_in_molarfraction, self.H2_reactor_in_molarfraction, self.NH3_reactor_in_molarfraction]
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        mu_cold = 1 / sum([molar_fractions[i] / viscosities[i] for i in range(len(components))]) # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        k_cold = np.dot(molar_fractions, conductivities)   # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        T_cold_in_RHE = self.T_points[i]
        h_cold_in_RHE = self.h_points[i]
        
        
        
        # Heating up to T_react
        i+=1
        self.points.append('RHE + Heater')
        self.p_points[i]=self.p_points[i-1] 
        self.T_points[i]=self.T_in_Ibed
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1]
        h_reactor_in=self.h_points[i]
        
        
        
        # Reactor products
        i+=1
        self.points.append('Reactor')
        self.p_points[i]=self.p_points[i-1]
        self.T_points[i]=self.T_IIIbed[-1]  # adiabatic reactor
        self.NH3_reactor_out_molarfraction=self.molarflowrate_NH3_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.N2_reactor_out_molarfraction=self.molarflowrate_N2_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.H2_reactor_out_molarfraction=self.molarflowrate_H2_out/(self.molarflowrate_NH3_out+self.molarflowrate_N2_out+self.molarflowrate_H2_out)
        self.mix_H2N2NH3.set_mole_fractions([self.N2_reactor_out_molarfraction, self.H2_reactor_out_molarfraction, self.NH3_reactor_out_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1]
        
        molar_fractions = [self.N2_reactor_out_molarfraction, self.H2_reactor_out_molarfraction, self.NH3_reactor_out_molarfraction]
        viscosities = [CP.PropsSI('VISCOSITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        conductivities = [CP.PropsSI('CONDUCTIVITY', 'T', self.T_points[i], 'P', self.p_points[i] * molar_fractions[i] * 100000, components[i]) for i in range(len(components))]
        mu_hot = 1 / sum([molar_fractions[i] / viscosities[i] for i in range(len(components))])  # viscosity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        k_hot = np.dot(molar_fractions, conductivities)   # conductivity of the mixture not available, calculated as the viscosity for an ideal gas mixture
        T_hot_in_RHE = self.T_points[i]
        h_hot_in_RHE = self.h_points[i]


        
        # Ammonia condensation
         
        # First part - cooled in part with the flowrate entering the reactor to recover heat 
        # Calculation of the heat that can be recovered - cooling of the products after the reactor - area from the design
        h_cold_out_RHE = None
        T_hot_out_RHE = None
        T_cold_out_RHE = None
        h_mixing = None
        T_mixing = None
        self.mix_H2N2NH3_aux.set_mole_fractions([self.N2_reactor_in_molarfraction, self.H2_reactor_in_molarfraction, self.NH3_reactor_in_molarfraction])
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
                self.mix_H2N2NH3_aux.update(CP.PT_INPUTS, self.p_react * 100000, T_cold_out_RHE)
                h_cold_out_RHE = self.mix_H2N2NH3_aux.hmass()
                cp_cold = (h_cold_out_RHE - h_cold_in_RHE) / (T_cold_out_RHE - T_cold_in_RHE)
                C_min = min(cp_cold * massflowrate_RHE, cp_hot * self.m_points[i])
                C_max = max(cp_cold * massflowrate_RHE, cp_hot * self.m_points[i])
                self.cp_min = min(cp_cold, cp_hot)
                C = C_min / C_max
                F1 = (k_hot / self.k_hot_design)**0.7 * (mu_hot / self.mu_hot_design)**(-0.5) * (self.m_points[i] / self.massflowrate_reactor_design)**0.8 * (cp_hot / self.cp_hot_design)**0.3
                F2 = (k_cold / self.k_cold_design)**0.7 * (mu_cold / self.mu_cold_design)**(-0.5) * (massflowrate_RHE / self.massflowrate_reactor_design)**0.8 * (cp_cold / self.cp_cold_design)**0.3
                F = 2 * F1 * F2 / (F1 + F2)
                U = self.U_RHE_design * F
                NTU = U * self.A_RHE / C_min
                self.epsilon_RHE = (1 - np.exp(-NTU * (1 - C))) / (1 - C * np.exp(-NTU * (1 - C)))
                self.Q = self.epsilon_RHE * C_min * (T_hot_in_RHE - T_cold_in_RHE)
                h_cold_out_RHE = h_cold_in_RHE + self.Q / massflowrate_RHE
                T_cold_out_new = T_ph(self.mix_H2N2NH3_aux, self.p_react, T_cold_in_RHE, h_cold_out_RHE)
                massflowrate_bypass = self.m_points[i] - massflowrate_RHE
                h_mixing = (h_cold_out_RHE * massflowrate_RHE + h_cold_in_RHE * massflowrate_bypass) / self.m_points[i]
                T_mixing = T_ph(self.mix_H2N2NH3_aux, self.p_points[i], T_cold_out_RHE, h_mixing)
                h_hot_out_RHE = h_hot_in_RHE - self.Q / self.m_points[i]
                T_hot_out_new = T_ph(self.mix_H2N2NH3, self.p_react, T_hot_in_RHE, h_hot_out_RHE)

                if abs(T_hot_out_new - T_hot_out_guess) < tol and abs(T_cold_out_new - T_cold_out_guess) < tol:
                    T_hot_out_RHE = T_hot_out_new
                    T_cold_out_RHE = T_cold_out_new
                    break

                T_hot_out_guess = T_hot_out_new
                T_cold_out_guess = T_cold_out_new

            return T_mixing - self.T_in_Ibed
        
        try:
            # To reach the right inlet temperature to the first bed a bypass of the RHE of a part of the cold flow can be implemented
            # If passing the entire flow rate of cold fluid through the RHE heats up too much
            massflowrate_RHE = brentq(recuperative_heat_exchanger, 0.01*self.m_points[i], self.m_points[i], xtol=1e-5, maxiter=100)
            self.bypass_RHE = (self.m_points[i]-massflowrate_RHE) / self.m_points[i] *100
            # No heat needed 
            self.electric_resistance_heat = 0
        except ValueError: 
            if recuperative_heat_exchanger(0.01*self.m_points[i]) < 0 and recuperative_heat_exchanger(self.m_points[i]) < 0:
                # If passing the entire flow rate of cold fluid through the RHE it remains too cold, a electric resistance is used to heat the reactants up to the target temperature
                massflowrate_RHE = self.m_points[i]
                self.bypass_RHE = 0
                recuperative_heat_exchanger(self.m_points[i])
                self.reactor_in_specific_heat=(h_reactor_in-h_cold_out_RHE)/1000   # [kJ/kg]
                self.electric_resistance_heat = self.reactor_in_specific_heat * self.m_points[i]
        # T_hot_RHE, T_cold_RHE, Q_RHE = profile_RHE_off_design(self.mix_H2N2NH3, self.mix_H2N2NH3_aux, 50, self.m_points[i], T_hot_in_RHE, self.p_react, massflowrate_RHE, T_cold_in_RHE, T_cold_out_RHE, self.p_react)
        # RHE_plot(T_hot_RHE, T_cold_RHE, Q_RHE)

        recuperative_heat_exchanger(self.m_points[i]*(1-0.14))
        self.epsilon_RHE_fixed_bypass = self.epsilon_RHE
        self.delta_T = T_hot_in_RHE - T_cold_in_RHE
        self.reactor_in_specific_heat_needed=(h_reactor_in-h_cold_in_RHE)/1000
        self.Q_RHE = self.Q / 1000 / self.m_points[i]
        self.cp_RHE = self.cp_min
        recuperative_heat_exchanger(massflowrate_RHE)

            
        i+=1
        self.points.append('RHE')
        self.p_points[i]=self.p_points[i-1]
        self.T_points[i]=T_hot_out_RHE
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.mix_H2N2NH3.hmass()
        self.m_points[i]=self.m_points[i-1] 
        
        

        # Second part - with the ambient (water)
        i+=1
        self.points.append('Condenser 1')
        self.p_points[i]=self.p_points[i-1]
        self.T_points[i]=self.T_IC
        self.m_points[i]=self.m_points[i-1]
        
        # Gas phase
        N2gas_Icondenser_molarfraction, H2gas_Icondenser_molarfraction, NH3gas_Icondenser_molarfraction = x_cond (self.mix_H2N2NH3, self.Ammonia, self.T_points[i], self.p_points[i], self.N2_molarfraction, self.H2_molarfraction)
        self.mix_H2N2NH3.set_mole_fractions([N2gas_Icondenser_molarfraction, H2gas_Icondenser_molarfraction, NH3gas_Icondenser_molarfraction])
        self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        h_gas_Icondenser=self.mix_H2N2NH3.hmass()
        
        # Liquid phase
        self.N2_reactor_out_molarflowrate=self.molarflowrate_N2_out
        gas_reactor_out_molarflowrate=self.N2_reactor_out_molarflowrate/self.N2_reactor_out_molarfraction
        gas_Icondenser_molarflowrate=self.N2_reactor_out_molarflowrate/N2gas_Icondenser_molarfraction
        NH3liq_Icondenser_molarflowrate=self.NH3_reactor_out_molarfraction*gas_reactor_out_molarflowrate-NH3gas_Icondenser_molarfraction*gas_Icondenser_molarflowrate
        NH3liq_Icondenser_massflowrate=NH3liq_Icondenser_molarflowrate*c.NH3MOLMASS
        self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        h_liq_Icondenser=self.Ammonia.hmass()       
        
        gas_Icondenser_massflowrate=self.m_points[i]-NH3liq_Icondenser_massflowrate
        self.h_points[i]=(NH3liq_Icondenser_massflowrate*h_liq_Icondenser+gas_Icondenser_massflowrate*h_gas_Icondenser)/self.m_points[i]
        
        self.Icondenser_specific_heat=(self.h_points[i]-self.h_points[i-1])/1000   # [kJ/kg]
        
        
        
        # Third part - with a refrigeration cycle
        if self.T_cond < self.T_IC:
            i+=1
            self.points.append('Condenser 2')
            self.p_points[i]=self.p_points[i-1]
            self.T_points[i]=self.T_cond
            self.m_points[i]=self.m_points[i-1]
            
            # The gas phase is the one inside the loop 
            self.mix_H2N2NH3.set_mole_fractions([self.N2_loop_molarfraction, self.H2_loop_molarfraction, self.NH3_loop_molarfraction])
            self.mix_H2N2NH3.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
            h_gas_IIcondenser=self.mix_H2N2NH3.hmass()
            
            # Liquid phase
            self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
            h_liq_IIcondenser=self.Ammonia.hmass()
            NH3liq_IIcondenser_massflowrate=self.m_points[i]-self.loop_massflowrate
            
            self.h_points[i]=(NH3liq_IIcondenser_massflowrate*h_liq_IIcondenser+self.loop_massflowrate*h_gas_IIcondenser)/self.m_points[i]
            
            self.IIcondenser_specific_heat=(self.h_points[i]-self.h_points[i-1])/1000   #[kJ/kg])
            PLR = self.IIcondenser_specific_heat * self.m_points[i] / self.IIcondenser_heat_design
            self.COP =  self.COP_design * part_load_COP(PLR)
            self.refrigerationcycle_work = -self.IIcondenser_specific_heat*self.m_points[i]/self.COP/self.eta_motor # [kW]
        else:
            self.refrigerationcycle_work = 0
        
        
        
        # Separator
        i+=1
        self.points.append('Separator-product')
        self.p_points[i]=self.p_points[i-1]
        self.T_points[i]=self.T_points[i-1]
        self.Ammonia.update(CP.PT_INPUTS, self.p_points[i]*100000, self.T_points[i])
        self.h_points[i]=self.Ammonia.hmass()
        self.m_points[i]=self.m_points[i-1]-self.loop_massflowrate       
        

            
        # Calculation of the total electricity consumed
        if self.p_N2==self.p_H2: 
                
            electricity=self.refrigerationcycle_work+np.sum(self.feed_compressor_work)+self.loop_compressor_work+self.electric_resistance_heat
                    
        if self.p_N2!=self.p_H2:
      
            electricity=self.refrigerationcycle_work+np.sum(self.N2_compressor_work)+np.sum(self.H2_compressor_work)+self.loop_compressor_work+self.electric_resistance_heat
        
        return(electricity)
    
    
    
    @lru_cache(maxsize=None)
    def consumption_simple(self):
        """
           Calculates the relationship between hydrogen load and the corresponding electricity consumption,
           while accounting for the fraction of hydrogen coming from the buffer. Both the load and the electricity 
           consumption are normalized by the hydrogen flowrate. The spline is fitted with:
               - the load (hydrogen consumption / nominal hydrogen value) as the first input,
               - the fraction of hydrogen from the buffer (a value between 0 and 1) as the second input.
           The resulting electricity consumption from the spline is normalized and must be multiplied by the flowrate 
           of hydrogen to recover the actual consumption. This normalized spline fit can then be used to model the system 
           behavior under various sizes for the case of 'automatic dimensioning' of the reactor beds.
           This procedure is made for three cases depending on the point of immission of the hydrogen from the buffer (it 
           means the pressure level of the buffer).
        """
        if self.nstages_H2_compressor != 2:
            raise ValueError("The simple calculation mode is made considering 2 stages of the H2 compressor. If another number of stages is used it needs to be modified.")
        
        if self.check == True:
            spline_1 = self.spline_1
            spline_2 = self.spline_2
            spline_3 = self.spline_3
            
        else:
            if self.reactor_beds_volume == 'automatic dimensioning':
                nominal_hyd = 1 * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS)  # for 1 kg/s of ammonia produced
                self.reactor_Ibed_volume = 0.9
                self.reactor_IIbed_volume = 1.25
                self.reactor_IIIbed_volume = 1.55
            else:
                nominal_hyd = self.max_hydrogen
            hydrogen = np.linspace(self.min_hydrogen/self.max_hydrogen * nominal_hyd, nominal_hyd, 50)   # from minimum to maximum hydrogen flowrate
            
            # First case, both compression stages activated
            hyd_buffer_frac_1 = []   # will store the fraction of hydrogen coming from the tank (0 to 1)
            load_1 = []  # will store the normalized hydrogen load (hydrogen flow divided by nominal hydrogen flow)
            consumption_1 = []  # will store the normalized electricity consumption (electricity consumption divided by nominal hydrogen flow)
            p_buffer = self.p_H2
            for hyd in hydrogen:
                hydrogen_buffer = (np.linspace(0, 1, 50))
                for hyd_buffer in hydrogen_buffer:
                    electricity = self.consumption(hyd, hyd*hyd_buffer, p_buffer)
                    load_1.append(hyd/nominal_hyd)
                    hyd_buffer_frac_1.append(hyd_buffer)
                    consumption_1.append(electricity/hyd)
            load_1 = np.array(load_1)
            hyd_buffer_frac_1 = np.array(hyd_buffer_frac_1)
            consumption_1 = np.array(consumption_1)
            spline_1 = SmoothBivariateSpline(load_1, hyd_buffer_frac_1, consumption_1, kx=5, ky=5)
            # Second case, just second compressor stage activated
            hyd_buffer_frac_2 = []  # will store the fraction of hydrogen coming from the tank (0 to 1)
            load_2 = []  # will store the normalized hydrogen load (hydrogen flow divided by nominal hydrogen flow)
            consumption_2 = []  # will store the normalized electricity consumption (electricity consumption divided by nominal hydrogen flow)
            p_buffer = self.p_H2*self.beta_stage_H2
            for hyd in hydrogen:
                hydrogen_buffer = (np.linspace(0, 1, 50))
                for hyd_buffer in hydrogen_buffer:
                    electricity = self.consumption(hyd, hyd*hyd_buffer, p_buffer)
                    load_2.append(hyd/nominal_hyd)
                    hyd_buffer_frac_2.append(hyd_buffer)
                    consumption_2.append(electricity/hyd)
            load_2 = np.array(load_2)
            hyd_buffer_frac_2 = np.array(hyd_buffer_frac_2)
            consumption_2 = np.array(consumption_2)
            spline_2 = SmoothBivariateSpline(load_2, hyd_buffer_frac_2, consumption_2, kx=5, ky=5)
            # Third case, no compressor stages activated
            hyd_buffer_frac_3 = []  # will store the fraction of hydrogen coming from the tank (0 to 1)
            load_3 = []  # will store the normalized hydrogen load (hydrogen flow divided by nominal hydrogen flow)
            consumption_3 = []  # will store the normalized electricity consumption (electricity consumption divided by nominal hydrogen flow)
            p_buffer = self.p_react
            for hyd in hydrogen:
                hydrogen_buffer = (np.linspace(0, 1, 50))
                for hyd_buffer in hydrogen_buffer:
                    electricity = self.consumption(hyd, hyd*hyd_buffer, p_buffer)
                    load_3.append(hyd/nominal_hyd)
                    hyd_buffer_frac_3.append(hyd_buffer)
                    consumption_3.append(electricity/hyd)
            load_3 = np.array(load_3)
            hyd_buffer_frac_3 = np.array(hyd_buffer_frac_3)
            consumption_3 = np.array(consumption_3)
            spline_3 = SmoothBivariateSpline(load_3, hyd_buffer_frac_3, consumption_3, kx=5, ky=5)
            
            # Save new spline in consumption
            splines = {"spline_1": spline_1, "spline_2": spline_2, "spline_3": spline_3}
            with open(self.path + '/consumption/calculations/' + self.name_serie, 'wb') as f:
                pickle.dump(splines, f)
            # Save new parameters in previous_simulation
            data_to_save = {"parameters": self.parameters,
                             "additional_data": {"T_H2": self.T_H2, "p_H2": self.p_H2, "T_N2": self.T_N2, "p_N2": self.p_N2, "buffer_info": self.buffer_info}}
            with open(self.path + f"{self.directory}/ASR_{self.file_structure}_{self.location_name}.pkl", 'wb') as f:
                pickle.dump(data_to_save, f)
                
            if self.reactor_beds_volume == 'automatic dimensioning':
                self.reactor_beds_volume = self.reactor_dimensioning()
                self.reactor_Ibed_volume = self.reactor_beds_volume[0]
                self.reactor_IIbed_volume = self.reactor_beds_volume[1]
                self.reactor_IIIbed_volume = self.reactor_beds_volume[2]
        
        return spline_1, spline_2, spline_3  
    
    
    
    def use(self, step, hyd, buffer_LOC=False, hyd_buffer=False, p_buffer=False):
        """
        The use function calculates the energy consumption in the operating range, with the simplified or extended calculation.
            
        Inputs:
            hyd: (>0) total hydrogen mass flowrate [kg/s]
            buffer_LOC: level of charge of the hydrogen buffer in fraction (0 to 1)
            hyd_buffer: (>0) part of hydrogen mass flowrate from the buffer [kg/s]
            p_buffer: pressure of the hydrogen coming from the buffer [bar]
            
        Outputs:
            ammonia: ammonia produced [kg/s]
            electricity: electricity consumption [kW] 
            nitrogen: nitrogen consumed [kg/s]
            hydrogen: hydrogen consumed [kg/s]
            hyd_buffer: quantity of hydrogen from the buffer [kg/s]
                    
        """
        excess_hyd = 0
        
        # Check if the timestep is inside the operational period, if not the production is limited by the value of self.state
        if not self.initial_hour <= step <= self.final_hour:
            if self.state == 'off':
                hydrogen = 0
                nitrogen = 0
                electricity = 0
                ammonia = 0
                hyd_buffer = 0
                return (ammonia, -electricity, -nitrogen, -hydrogen, hyd_buffer)
            if self.state == 'min load':
                if hyd < self.min_hydrogen:
                     new_hyd = self.min_hydrogen
                     hyd_buffer = hyd_buffer + (new_hyd - hyd)
                     hyd = new_hyd
                else:
                    excess_hyd = hyd - self.min_hydrogen
                    hyd = self.min_hydrogen
            
        
        # First check to see if the hydrogen is inside the available production range
        if round(hyd, 15) < round(self.min_hydrogen, 15):  # the ASR can't be used
            hydrogen = 0
            nitrogen = 0
            electricity = 0
            ammonia = 0
            hyd_buffer = 0
            return (ammonia, -electricity, -nitrogen, -hydrogen, hyd_buffer)
        elif hyd > self.max_hydrogen:
            hyd = self.max_hydrogen    # just the maximum capacity of the reactor can be consumed
            excess_hyd = hyd - self.max_hydrogen
        else:
            hyd = hyd
            
        # Second check to see if the hydrogen buffer level is too low, hydrogen accumulated instead of used
        if self.buffer_control and hyd > self.min_hydrogen and buffer_LOC < self.buffer_control:
            excess_hyd = hyd - self.min_hydrogen
            hyd = self.min_hydrogen
        
        self.load[step] = hyd / self.max_hydrogen
        # Third check to see if the load change is inside the admissible ramp-rate
        if step != 0:    
            if abs(self.load[step] - self.load[step-1]) > self.max_ramp_rate:
                if self.load[step] > self.load[step-1]:
                    self.load[step] = self.load[step-1] + self.max_ramp_rate
                    new_hyd = self.load[step] * self.max_hydrogen
                    hyd = new_hyd
                else:
                    self.load[step] = self.load[step-1] - self.max_ramp_rate
                    new_hyd = self.load[step] * self.max_hydrogen
                    hyd_buffer = hyd_buffer + (new_hyd - hyd)
                    if excess_hyd > 0:
                        hyd_buffer = max(0, (new_hyd - hyd) - excess_hyd)
                    hyd = new_hyd
            else:
                hyd = hyd
            
        if self.calculation_mode == 'simple':
            # Consumption simple function gives the specific electricity consumption with respect to the value of hydrogen flowrate
            if p_buffer < self.p_H2*self.beta_stage_H2: # the H2 coming from the tank is entered before the first stage
                consumption_simple = self.consumption_simple()[0]
                electricity = hyd * consumption_simple(hyd/self.max_hydrogen, hyd_buffer/hyd)[0].item()
            if self.p_H2*self.beta_stage_H2 <= p_buffer < self.p_react: # the H2 coming from the tank is entered before the second stage
                consumption_simple = self.consumption_simple()[1]
                electricity = hyd * consumption_simple(hyd/self.max_hydrogen, hyd_buffer/hyd)[0].item()
            if p_buffer >= self.p_react: 
                consumption_simple = self.consumption_simple()[2]
                electricity = hyd * consumption_simple(hyd/self.max_hydrogen, hyd_buffer/hyd)[0].item()              
        else:
            electricity = self.consumption(hyd, hyd_buffer, p_buffer)


        hydrogen = hyd
        nitrogen = hyd * c.N2MOLMASS / (3 * c.H2MOLMASS)
        ammonia = hydrogen + nitrogen
        if hyd_buffer == False:
            hyd_buffer = 0
            
        return (ammonia, -electricity, -nitrogen, -hydrogen, hyd_buffer)
    
    
    
    def thermodynamic_points(self):
        thermodynamic_points = []
        
        for i in range(len(self.points)):
            
           point = (self.points[i], round(self.T_points[i],2), round(self.p_points[i],2), round(self.m_points[i],2))
           thermodynamic_points.append(point)
           
        df = pd.DataFrame(thermodynamic_points, columns=['Component', 'T [K]', 'P [bar]', 'Mass flowrate [kg/s]'])

        return df
        
    
    
    def reactor_plot(self):
        
            reactor_volume = np.concatenate([np.linspace(0, self.reactor_Ibed_volume, 50), np.linspace(self.reactor_Ibed_volume, self.reactor_Ibed_volume+self.reactor_IIbed_volume, 50), np.linspace(self.reactor_Ibed_volume+self.reactor_IIbed_volume, self.reactor_Ibed_volume+self.reactor_IIbed_volume+self.reactor_IIIbed_volume, 50)])
            XN2 = np.concatenate([self.XN2_Ibed, self.XN2_IIbed, self.XN2_IIIbed])
            T = np.concatenate([self.T_Ibed, self.T_IIbed, self.T_IIIbed])
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.ylim(min(T)-10, max(T)+30)
            plt.xlim(0, max(reactor_volume))
            plt.axvline(x=self.reactor_Ibed_volume, color='dimgrey', linestyle='--')
            plt.text(0+self.reactor_Ibed_volume/3, max(T)+15, '1st bed', fontsize=15, color='black', verticalalignment='center')
            plt.axvline(x=self.reactor_Ibed_volume+self.reactor_IIbed_volume, color='dimgrey', linestyle='--')
            plt.text(self.reactor_Ibed_volume+self.reactor_IIbed_volume/3, max(T)+15, '2nd bed', fontsize=15, color='black', verticalalignment='center')
            plt.text(self.reactor_Ibed_volume+self.reactor_IIbed_volume+self.reactor_IIIbed_volume/3, max(T)+15, '3rd bed', fontsize=15, color='black', verticalalignment='center')
            plt.plot(reactor_volume, T, linestyle='-', color='lightcoral', linewidth = 2.5)
            plt.title("Reactor temperature profile")
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("Temperature [K]")
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(8, 5), dpi=150)
            plt.plot(reactor_volume, XN2*100, linestyle='-', color='lightcoral', linewidth = 2.5)
            plt.ylim(0, max(XN2)*100+5)
            plt.xlim(0, max(reactor_volume))
            plt.axvline(x=self.reactor_Ibed_volume, color='dimgrey', linestyle='--')
            plt.text(0+self.reactor_Ibed_volume/3, max(XN2)*100+2, '1st bed', fontsize=15, color='black', verticalalignment='center')
            plt.axvline(x=self.reactor_Ibed_volume+self.reactor_IIbed_volume, color='dimgrey', linestyle='--')
            plt.text(self.reactor_Ibed_volume+self.reactor_IIbed_volume/3, max(XN2)*100+2, '2nd bed', fontsize=15, color='black', verticalalignment='center')
            plt.text(self.reactor_Ibed_volume+self.reactor_IIbed_volume+self.reactor_IIIbed_volume/3, max(XN2)*100+2, '3rd bed', fontsize=15, color='black', verticalalignment='center')
            plt.title("Reactor nitrogen conversion profile")
            plt.xlabel("Reactor volume [m$^3$]")
            plt.ylabel("N$_2$ conversion [%]")
            plt.grid(True)
            plt.show()
    
    
    
    def pressure_buffer_check(self,original_buffer_LOP, buffer_LOP):
        """
        Since the hydrogen buffer may not provide the correct pressure in the first simulation due to a possible shift in the LOC, 
        this function checks whether the real LOC is within the same range as the previous (not shifted) LOC. 
        If it is not, the ASR consumption calculated is incorrect because, depending on the pressure range, 
        hydrogen can enter a different section of the compression line.

        """
        # For 2 stages hydrogen compressor
        check = True
        mismatch_timesteps = []
        for i in range(len(buffer_LOP)):
            if original_buffer_LOP[i] < self.p_H2 * self.beta_stage_H2:
                if not buffer_LOP[i] < self.p_H2 * self.beta_stage_H2:
                    mismatch_timesteps.append(i)
                    check = False
            
            elif self.p_H2 * self.beta_stage_H2 <= original_buffer_LOP[i] < self.p_react:
                if not (self.p_H2 * self.beta_stage_H2 <= buffer_LOP[i] < self.p_react):
                    mismatch_timesteps.append(i)
                    check = False
            
            elif original_buffer_LOP[i] >= self.p_react:
                if not buffer_LOP[i] >= self.p_react:
                    mismatch_timesteps.append(i)
                    check = False
                    
        return check, mismatch_timesteps

            
    
    def tech_cost(self, tech_cost):
        """
        Inputs:
            tech_cost: dict
                'cost per unit': [€/(tonne/day)]
                'OeM': operation and maintenance costs, percentage on initial investment [%]
                'refund': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of equipment cost [%]
                    'years': after how many years it will be replaced

        Outputs:
            self.cost: dict
                'total cost': [€]
                'OeM': [€]
                'refund': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of equipment cost [%]
                    'years': after how many years it will be replaced
        """
        
        tech_cost = {key: value for key, value in tech_cost.items()} 
        
        size = self.max_ammonia * 3600 * 24 / 1000                                     # theoretical production at max capacity, [tonne/day]
        
        if tech_cost['cost per unit'] == 'default price correlation':
        # This correlations already include all the direct and indirect costs
            if size < 100:
                C = 2 * 10**6 * size**0.6                                              # [€], Ref: Rouwenhorst, Kevin Hendrik Reindert, et al. "Ammonia production technologies." Techno-Economic Challenges of Green Ammonia as an Energy Vector (2021): 41-83.
            else:
                exchange_rate = 0.95                                                   # exchange rate between USD and €
                CEPCI_2010 = 550.8
                CEPCI_2024 = 800
                C = (23850000 * size**(-1.3409) + 173500) * size * exchange_rate       # [€], Ref: Fasihi, Mahdi, et al. "Global potential of green ammonia based on hybrid PV-wind power plants." Applied Energy 294 (2021): 116170.
                C = C * CEPCI_2024 / CEPCI_2010
        else:
            C = size * tech_cost['cost per unit']                            

        tech_cost['total cost'] = tech_cost.pop('cost per unit')
        tech_cost['total cost'] = C
        tech_cost['OeM'] = tech_cost['OeM'] *C /100 # [€]
        self.cost = tech_cost
        

        
        # plant_size = np.linspace(0.1, 500, 100)
        # cost = np.zeros(len(plant_size))
        # specific_cost = np.zeros(len(plant_size))
        # i = 0
        # for ps in plant_size:
        #     if ps < 100:
        #         cost[i] = 2*10**6*ps**0.6
        #         specific_cost[i] = cost[i] / ps
        #     else:
        #         specific_cost[i] = (23850000 * ps**(-1.3409) + 173500) * 0.95 * 800 / 550.8
        #         cost[i] = specific_cost[i] * ps 
        #     i +=1
        # fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
        # ax1.set_xlabel('Plant size [t$_{NH3}$/day]', fontsize=12)
        # ax1.set_ylabel('Cost [M€]', color='blue', fontsize=12)
        # line1, = ax1.plot(plant_size, cost / 1e6, color='blue', linewidth=2, label='Cost')
        # ax1.tick_params(axis='y', labelcolor='blue')
        # ax1.grid()
        # ax1.axvline(x=100, color='black', linestyle='--', linewidth=1)
        # ax1.text(90, ax1.get_ylim()[1] * 0.93, 'eq. (51)', fontsize=12, color='black', ha='right')
        # ax1.text(220, ax1.get_ylim()[1] * 0.93, 'eq. (50)', fontsize=12, color='black', ha='left')
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Cost per unit [M€/(tonne$_{NH3}$/day)]', color='red', fontsize=12)
        # line2, = ax2.plot(plant_size, specific_cost / 1e6, color='red', linewidth=2, label='Cost per unit')
        # ax2.tick_params(axis='y', labelcolor='red')
        # ax2.set_ylim(0, 1.5)
        # plt.title('NH$_{3}$ Plant Cost', fontsize=14)
        # lines = [line1, line2]
        # labels = [line.get_label() for line in lines]
        # ax1.legend(lines, labels, loc="lower right")
        # plt.show()
        

        



'----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
 


         
if __name__ == "__main__":
    
    inp_test = {'Preact'          : 150,
                'comp efficiency'   : 0.85,
                'Tcondenser'          : 263.15,
                'Treact'            : [663.15,693.15, 683.15],
                'max_prod'           : 1,
                'cooling COP'        : 3,
                'reactor beds volume'        : 'automatic dimensioning',
                # 'nstages feed compressor'    : 2,
                'nstages N2 compressor'    : 3,
                'nstages H2 compressor'    : 2,
                "operational_period"   : "01-01,31-12",
   	            "state"                : "min load",
                'only_renewables'   : False}
    
    inp_test_simple = {'Preact'          : 150,
                'comp efficiency'   : 0.85,
                'Tcondenser'          : 263.15,
                'Treact'            : [663.15,693.15, 683.15],
                'max_prod'           : 1,
                'cooling COP'        : 3,
                'reactor beds volume'        : 'automatic dimensioning',
                # 'nstages feed compressor'    : 2,
                'nstages N2 compressor'    : 3,
                'nstages H2 compressor'    : 2,
                'calculation mode' : 'simple',
                "operational_period"   : "01-01,31-12",
   	            "state"                : "min load",
                'only_renewables'   : False}


    T_p_hyd=[343.15,30]
    T_p_nitro=[298.15,7]
    buffer_info=[288.15, 200, 30]
    p_buffer=30
    timestep = 60      # [min] selected timestep for the simulation
    location_name = 'electricity_consumer'
    path = r'I:\Il mio Drive\MESS\input_dev_ammonia' 
    file_structure = 'studycase'
    file_general = 'general'

    hyd = 0.5*inp_test['max_prod'] * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS) # [kg/s]
    hyd_buffer = 0*hyd


    asr=ASR(inp_test, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, buffer_info=buffer_info, timestep=timestep)
    # asr_simple=ASR(inp_test_simple, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, buffer_info=buffer_info, timestep=timestep)
    
    ammonia, electricity, _, _, _ = asr.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
    # ammonia, electricity_simple, _, _, _ = asr_simple.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
    # asr.reactor_plot()
    # thermodynamic_points=asr.thermodynamic_points()
    # print(thermodynamic_points)
    
#%% # Sensitivity on operating pressure
    
    # Pressure VS reactor volumes and condenser temperature (temperatures to have the same fraction of ammonia recirculated, volumes from automatic dimensioning)
    pressure = [150, 200, 250, 300]
    T_condenser = [263.15, 266.15, 268.15, 269.15]
    reactor_volume = [[0.9, 1.25, 1.55], [0.55, 0.8, 1.1], [0.4, 0.6, 0.8], [0.3, 0.45, 0.65]]
    pressure_smooth = np.linspace(min(pressure), max(pressure), 200)  
    spline = make_interp_spline(pressure, T_condenser, k=2) 
    T_condenser_smooth = spline(pressure_smooth)  
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax1.plot(pressure_smooth, T_condenser_smooth, color='lightcoral', label='Condenser temperature', linewidth = 2.2)
    ax1.scatter(pressure, T_condenser, marker='^', color='lightcoral')
    ax1.set_xlabel('Pressure [bar]')
    ax1.set_ylabel('Temperature [K]')
    ax1.grid(True)
    vol_Ibed = [row[0] for row in reactor_volume]
    vol_IIbed = [row[1] for row in reactor_volume]
    vol_IIIbed = [row[2] for row in reactor_volume]
    pressure_smooth = np.linspace(min(pressure), max(pressure), 200)
    spline_I = make_interp_spline(pressure, vol_Ibed, k=2)
    spline_II = make_interp_spline(pressure, vol_IIbed, k=2)
    spline_III = make_interp_spline(pressure, vol_IIIbed, k=2)
    vol_Ibed_smooth = spline_I(pressure_smooth)
    vol_IIbed_smooth = spline_II(pressure_smooth)
    vol_IIIbed_smooth = spline_III(pressure_smooth)
    ax2 = ax1.twinx()
    ax2.plot(pressure_smooth, vol_Ibed_smooth, label='I bed volume', color='darkturquoise', linewidth = 2.2)
    ax2.plot(pressure_smooth, vol_IIbed_smooth, label='II bed volume', color='plum', linewidth = 2.2)
    ax2.plot(pressure_smooth, vol_IIIbed_smooth, label='III bed volume', color='gold', linewidth = 2.2)
    ax2.scatter(pressure, vol_Ibed, color='darkturquoise')
    ax2.scatter(pressure, vol_IIbed, color='plum')
    ax2.scatter(pressure, vol_IIIbed, color='gold')
    ax2.set_ylabel('Reactor bed volume [m³]', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    lines1, labels1 = ax1.get_legend_handles_labels()  
    lines2, labels2 = ax2.get_legend_handles_labels() 
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.6, 1))
    plt.title('Condenser temperature and reactor bed volumes vs pressure')
    plt.show()
    
    
    # Pressure VS temperature and conversion profiles inside the reactor at full load
    XN2_Ibed = np.zeros(len(pressure))
    XN2_IIbed = np.zeros(len(pressure))
    XN2_IIIbed = np.zeros(len(pressure))
    loop_fraction = np.zeros(len(pressure))
    consumption = np.zeros(len(pressure))
    refrigerationcycle_work = np.zeros(len(pressure))
    fresh_feed_compressor_work = np.zeros(len(pressure))
    loop_compressor_work = np.zeros(len(pressure))
    electric_resistance_heat = np.zeros(len(pressure))
    T_max = np.zeros(len(pressure))
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 12), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    styles = ['-', '--', '-.', ':']
    color = ['lightcoral', 'darkturquoise', 'plum', 'goldenrod']
    
    for i, p in enumerate(pressure):
        inp_test['Preact'] = p
        inp_test['Tcondenser'] = T_condenser[i]
        inp_test['reactor beds volume'] = reactor_volume[i]
        asr=ASR(inp_test, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, timestep=timestep)
        
        reactor_volume_tot = np.concatenate([np.linspace(0, asr.reactor_Ibed_volume, 50), np.linspace(asr.reactor_Ibed_volume, asr.reactor_Ibed_volume+asr.reactor_IIbed_volume, 50), np.linspace(asr.reactor_Ibed_volume+asr.reactor_IIbed_volume, asr.reactor_Ibed_volume+asr.reactor_IIbed_volume+asr.reactor_IIIbed_volume, 50)])
        XN2 = np.concatenate([asr.XN2_Ibed, asr.XN2_IIbed, asr.XN2_IIIbed])
        T = np.concatenate([asr.T_Ibed, asr.T_IIbed, asr.T_IIIbed])
        ax[0].plot(reactor_volume_tot, XN2 * 100, linestyle=styles[i], color = color[i], label=f'Pressure = {p} bar', linewidth = 2.2)
        ax[1].plot(reactor_volume_tot, T, linestyle=styles[i], color = color[i], label=f'Pressure = {p} bar', linewidth = 2.2)
        
        XN2_Ibed[i] = asr.XN2_Ibed[-1] * 100
        XN2_IIbed[i] = asr.XN2_IIbed[-1] * 100
        XN2_IIIbed[i] = asr.XN2_IIIbed[-1] * 100
        loop_fraction[i] = round(asr.loop_massflowrate / (asr.H2_massflowrate+asr.N2_massflowrate), 2)
        hyd = 1*inp_test['max_prod'] * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS) # kg/s
        _, electricity, _, _, _ = asr.use(0, hyd)
        consumption[i] = abs(electricity)
        refrigerationcycle_work[i] = asr.refrigerationcycle_work
        fresh_feed_compressor_work[i] = sum(asr.N2_compressor_work) + sum(asr.H2_compressor_work)
        loop_compressor_work[i] = asr.loop_compressor_work
        electric_resistance_heat[i] = asr.electric_resistance_heat
        hyd = 0.3*inp_test['max_prod'] * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS) # kg/s
        _, electricity, _, _, _ = asr.use(0, hyd)
        T_max[i] = max(max(asr.T_Ibed), max(asr.T_IIbed), max(asr.T_IIIbed))
      
    ax[0].set_title("Reactor N$_2$ conversion profile at different pressures")
    ax[0].set_xlabel("Reactor volume [m$^3$]")
    ax[0].set_ylabel("N$_2$ conversion [%]")
    ax[0].legend(loc='lower right')
    ax[0].grid(True)
    ax[1].set_title("Reactor temperature profile at different pressures")
    ax[1].set_xlabel("Reactor volume [m$^3$]")
    ax[1].set_ylabel("Temperature [K]")
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    plt.show()

        
    # Pressure VS final conversion and loop massflowrate fraction
    spline_IIIbed = make_interp_spline(pressure, XN2_IIIbed, k=2)
    spline_loop = make_interp_spline(pressure, loop_fraction, k=2)
    XN2_IIIbed_smooth = spline_IIIbed(pressure_smooth)
    loop_fraction_smooth = spline_loop(pressure_smooth)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax1.plot(pressure_smooth, XN2_IIIbed_smooth, label='N$_2$ conversion', color='lightcoral', linewidth = 2.2)
    ax1.scatter(pressure, XN2_IIIbed, color='lightcoral')
    ax1.set_xlabel('Pressure [bar]')
    ax1.set_ylabel('N$_2$ conversion [%]')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(pressure_smooth, loop_fraction_smooth, label='Loop massflowrate', color='darkturquoise', linewidth = 2.2)
    ax2.scatter(pressure, loop_fraction, color='darkturquoise')  
    ax2.set_ylabel('Loop massflowrate / fresh feed massflowrate [-]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.6, 0.98))
    plt.title('Conversion and loop massflowrate vs pressure')
    plt.show()
    
    
    # Pressure VS electricity consumption
    consumption_smooth = make_interp_spline(pressure, consumption, k=2)(pressure_smooth)
    refrigerationcycle_work_smooth = make_interp_spline(pressure, refrigerationcycle_work, k=2)(pressure_smooth)
    fresh_feed_compressor_work_smooth = make_interp_spline(pressure, fresh_feed_compressor_work, k=2)(pressure_smooth)
    loop_compressor_work_smooth = make_interp_spline(pressure, loop_compressor_work, k=2)(pressure_smooth)
    electric_resistance_heat_smooth = make_interp_spline(pressure, electric_resistance_heat, k=2)(pressure_smooth)    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)  
    axes[0].plot(pressure_smooth, consumption_smooth, label='Total', color='darkturquoise', linewidth = 2.2)
    axes[0].plot(pressure_smooth, fresh_feed_compressor_work_smooth, label='Fresh feed compressors', color='goldenrod', linewidth = 2.2)
    axes[0].plot(pressure_smooth, refrigerationcycle_work_smooth, label='Refrigeration cycle', color='green', linewidth = 2.2)
    axes[0].scatter(pressure, consumption, color='darkturquoise', s=30)
    axes[0].scatter(pressure, fresh_feed_compressor_work, color='goldenrod', s=30)
    axes[0].scatter(pressure, refrigerationcycle_work, color='green', s=30)
    axes[0].set_xlabel('Pressure [bar]')
    axes[0].set_ylabel('Electricity consumption [kW]')
    axes[0].grid(True)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    axes[1].plot(pressure_smooth, loop_compressor_work_smooth, label='Loop compressor', color='lightcoral', linewidth = 2.2)
    axes[1].plot(pressure_smooth, electric_resistance_heat_smooth, label='Electric resistance', color='purple', linewidth = 2.2)
    axes[1].scatter(pressure, loop_compressor_work, color='lightcoral', s=30)
    axes[1].scatter(pressure, electric_resistance_heat, color='purple', s=30)
    axes[1].set_xlabel('Pressure [bar]')
    axes[1].set_ylabel('Electricity consumption [kW]')
    axes[1].grid(True)
    axes[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0.5))
    fig.suptitle('Electricity consumption vs pressure')
    plt.show()
    

    # Pressure vs temperature and conversion profiles inside the reactor at minimum load
    fig, ax = plt.subplots(2, 1, figsize=(8, 12), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    styles = ['-', '--', '-.', ':']
    color = ['lightcoral', 'darkturquoise', 'plum', 'goldenrod']
    
    for i, p in enumerate(pressure):
        inp_test['Preact'] = p
        inp_test['Tcondenser'] = T_condenser[i]
        inp_test['reactor beds volume'] = reactor_volume[i]
        asr=ASR(inp_test, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, timestep=timestep)
        hyd = 0.3*inp_test['max_prod'] * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS) # kg/s
        _, electricity, _, _, _ = asr.use(0, hyd)
        
        reactor_volume_tot = np.concatenate([np.linspace(0, asr.reactor_Ibed_volume, 50), np.linspace(asr.reactor_Ibed_volume, asr.reactor_Ibed_volume+asr.reactor_IIbed_volume, 50), np.linspace(asr.reactor_Ibed_volume+asr.reactor_IIbed_volume, asr.reactor_Ibed_volume+asr.reactor_IIbed_volume+asr.reactor_IIIbed_volume, 50)])
        XN2 = np.concatenate([asr.XN2_Ibed, asr.XN2_IIbed, asr.XN2_IIIbed])
        T = np.concatenate([asr.T_Ibed, asr.T_IIbed, asr.T_IIIbed])
        ax[0].plot(reactor_volume_tot, XN2 * 100, linestyle=styles[i], color = color[i], label=f'Pressure = {p} bar', linewidth=2.2)
        ax[1].plot(reactor_volume_tot, T, linestyle=styles[i], color = color[i], label=f'Pressure = {p} bar', linewidth=2.2)
        
        XN2_Ibed[i] = asr.XN2_Ibed[-1] * 100
        XN2_IIbed[i] = asr.XN2_IIbed[-1] * 100
        XN2_IIIbed[i] = asr.XN2_IIIbed[-1] * 100
    
    ax[0].set_title("Reactor N$_2$ conversion profile at different pressures")
    ax[0].set_xlabel("Reactor volume [m$^3$]")
    ax[0].set_ylabel("N$_2$ conversion [%]")
    ax[0].legend(loc='lower right', fontsize=9)
    ax[0].grid(True)
    ax[1].set_title("Reactor temperature profile at different pressures")
    ax[1].set_xlabel("Reactor volume [m$^3$]")
    ax[1].set_ylabel("Temperature [K]")
    ax[1].axhline(y=800, color='dimgray', linestyle='-', linewidth=1.5, alpha=0.5, label='Limit temperature (800 K)')
    ax[1].legend(loc='upper right', fontsize=9)
    ax[1].grid(True)
    plt.show()


    
#%% # Specific consumption
     
    hydrogen = np.linspace(asr.min_hydrogen, asr.max_hydrogen, 20)
    specific_consumption = np.zeros(len(hydrogen))
    specific_consumption_refrigeration = np.zeros(len(hydrogen))
    specific_consumption_N2_H2_compressors = np.zeros(len(hydrogen))
    specific_consumption_loop_compressor = np.zeros(len(hydrogen))
    specific_consumption_compressors = np.zeros(len(hydrogen))
    specific_consumption_electric_resistance = np.zeros(len(hydrogen))
    loop_fraction = np.zeros(len(hydrogen))
    loop_pressure_drop = np.zeros(len(hydrogen))
    COP = np.zeros(len(hydrogen))
    part_eta_N2_H2 = np.zeros(len(hydrogen))
    part_eta_loop = np.zeros(len(hydrogen))
    bypass_RHE = np.zeros(len(hydrogen))
    reactor_in_specific_heat_needed = np.zeros(len(hydrogen))
    delta_T_RHE = np.zeros(len(hydrogen))
    epsilon_RHE = np.zeros(len(hydrogen))
    Q_RHE = np.zeros(len(hydrogen))
    cp_RHE = np.zeros(len(hydrogen))
    T_Ibed_out = np.zeros(len(hydrogen))
    T_IIbed_out = np.zeros(len(hydrogen))
    T_IIIbed_out = np.zeros(len(hydrogen))
    i=0

    for hyd in hydrogen:
          ammonia, electricity, _, _, _ = asr.use(0, hyd)
          specific_consumption_refrigeration[i] = abs(asr.refrigerationcycle_work) / (ammonia * 3600)
          specific_consumption_N2_H2_compressors[i] = (abs(np.sum(asr.N2_compressor_work)) + abs(np.sum(asr.H2_compressor_work))) / (ammonia * 3600)
          specific_consumption_loop_compressor[i] = abs(asr.loop_compressor_work) / (ammonia * 3600)
          specific_consumption_compressors[i] = (abs(np.sum(asr.N2_compressor_work)) + abs(np.sum(asr.H2_compressor_work)) + abs(asr.loop_compressor_work)) / (ammonia * 3600)
          specific_consumption_electric_resistance[i] = abs(asr.electric_resistance_heat) / (ammonia * 3600)
          specific_consumption[i] = abs(electricity)/(ammonia * 3600)
          loop_fraction[i] = asr.loop_massflowrate / ammonia
          loop_pressure_drop[i] = asr.loop_pressure_drop_scaled
          COP[i] = asr.COP
          part_eta_N2_H2[i] = asr.eta_iso_N2/asr.eta_iso_N2_design
          part_eta_loop[i] = asr.eta_iso_loop/asr.eta_iso_loop_design
          bypass_RHE[i] = asr.bypass_RHE
          Q_RHE[i] = asr.Q_RHE
          cp_RHE[i] = asr.cp_RHE
          reactor_in_specific_heat_needed[i] = asr.reactor_in_specific_heat_needed
          delta_T_RHE[i] = asr.delta_T
          epsilon_RHE[i] = asr.epsilon_RHE_fixed_bypass
          T_Ibed_out[i] = asr.T_Ibed[-1]
          T_IIbed_out[i] = asr.T_IIbed[-1]
          T_IIIbed_out[i] = asr.T_IIIbed[-1]
          i += 1

    # Specific consumption refrigeration cycle
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax1.plot(hydrogen/asr.max_hydrogen*100, specific_consumption_refrigeration, label='Refrigeration cycle consumption', color='green', linewidth=2.2)
    ax1.set_xlabel('Load [%]')
    ax1.set_ylabel('Specific consumption [kWh/kg$_{NH3}$]')
    ax2 = ax1.twinx()
    ax2.plot(hydrogen/asr.max_hydrogen*100, loop_fraction, label='Loop massflowrate', color='darkgrey', linestyle='--', linewidth=2.2)
    ax2.set_ylabel('Loop massflowrate / fresh feed massflowrate [-]')
    ax2.set_ylim(np.min(loop_fraction)-0.5, np.max(loop_fraction)+0.5)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.plot(hydrogen/asr.max_hydrogen*100, COP, label='COP', color='lightgreen', linestyle='--', linewidth=2.2)
    ax3.set_ylabel('COP [-]')
    ax1.set_title('Refrigeration cycle')
    ax1.grid(True)
    fig.legend(loc='upper left', bbox_to_anchor=(0.14, 0.85))
    plt.show()
    
    # Specific consumption compressors
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax1.plot(hydrogen/asr.max_hydrogen*100, specific_consumption_N2_H2_compressors, label='Fresh feed compressors consumption', color='goldenrod', linewidth=2.2)
    ax1.set_xlabel('Load [%]')
    ax1.set_ylabel('Specific consumption [kWh/kg$_{NH3}$]')
    ax1.grid(True)
    ax2 = ax1.twinx() 
    ax2.plot(hydrogen/asr.max_hydrogen*100, part_eta_N2_H2, label=r"$\eta_{part}$ fresh feed compressors", color='orange', linestyle='--', linewidth=2.2)
    ax2.set_ylabel('$\eta_{part}$ [-]')
    lines1, labels1 = ax1.get_legend_handles_labels()  
    lines2, labels2 = ax2.get_legend_handles_labels() 
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.55, 1))
    ax1.set_title('Fresh feed compressors')
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(10, 6), dpi=150)
    ax3.plot(hydrogen/asr.max_hydrogen*100, specific_consumption_loop_compressor, label='Loop compressor consumption', color='lightcoral', linewidth=2.2)
    ax3.set_xlabel('Load [%]')
    ax3.set_ylabel('Specific consumption [kWh/kg$_{NH3}$]')
    ax3.grid(True)
    ax4 = ax3.twinx()
    ax4.plot(hydrogen/asr.max_hydrogen*100, loop_fraction, label='Loop massflowrate', color='darkgrey', linestyle='--', linewidth=2.2)
    ax4.set_ylabel('Loop massflowrate / fresh feed massflowrate [-]')
    ax4.set_ylim(2.2, 3.6)
    ax5 = ax3.twinx()
    ax5.spines['right'].set_position(('outward', 60))  
    ax5.plot(hydrogen/asr.max_hydrogen*100, part_eta_loop, label=r"$\eta_{part}$ loop compressor", color='chocolate', linestyle='--', linewidth=2.2)
    ax5.set_ylabel('$\eta_{part}$ [-]')
    ax6 = ax3.twinx()
    ax6.spines['right'].set_position(('outward', 120))  
    ax6.plot(hydrogen/asr.max_hydrogen*100, loop_pressure_drop, label='Loop pressure drop', color='orangered', linestyle='--', linewidth=2.2)
    ax6.set_ylabel('Loop pressure drop [bar]')
    lines3, labels3 = ax3.get_legend_handles_labels()  
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax3.legend(lines3 + lines4 + lines5 + lines6, labels3 + labels4 + labels5 + labels6, loc='upper right', bbox_to_anchor=(0.5, 1))
    ax3.set_title('Loop compressor')
    plt.show()


    # RHE heat
    Q_RHE_smoothed = savgol_filter(Q_RHE, window_length=11, polyorder=3)
    epsilon_RHE_smoothed = savgol_filter(epsilon_RHE, window_length=11, polyorder=3)
    delta_T_RHE_smoothed = savgol_filter(delta_T_RHE, window_length=11, polyorder=3)
    cp_RHE_smoothed = savgol_filter(cp_RHE, window_length=11, polyorder=3)
    reactor_in_specific_heat_needed_smoothed = savgol_filter(reactor_in_specific_heat_needed, window_length=11, polyorder=3)
    bypass_RHE_smoothed = savgol_filter(bypass_RHE, window_length=11, polyorder=3)
  
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    line1, = ax1.plot(hydrogen/asr.max_hydrogen*100, Q_RHE_smoothed, label='Specific heat available', color='purple', linewidth=2.2)
    ax1.set_xlabel('Load [%]')
    ax1.set_ylabel('Heat [kJ/kg]')
    ax1.grid(True)
    ax1_twin1 = ax1.twinx()
    line2, = ax1_twin1.plot(hydrogen/asr.max_hydrogen*100, epsilon_RHE_smoothed, label='RHE effectiveness', color='slateblue', linestyle='--',linewidth=2.2)
    ax1_twin1.set_ylabel('RHE effectiveness [-]')
    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('outward', 60))
    line3, = ax1_twin2.plot(hydrogen/asr.max_hydrogen*100, delta_T_RHE_smoothed, label='Max temperature difference', color='crimson', linestyle='--', linewidth=2.2)
    ax1_twin2.set_ylabel('Temperature difference [K]')
    ax1_twin3 = ax1.twinx()
    ax1_twin3.spines['right'].set_position(('outward', 120))
    line4, = ax1_twin3.plot(hydrogen/asr.max_hydrogen*100, cp_RHE_smoothed/1000, label='C$_{p, min}$', color='magenta', linestyle='--', linewidth=2.2)
    ax1_twin3.set_ylabel('C$_{p, min}$ [kJ/(kg*K)]')
    ax1.set_title('Fixed bypass 14 % - RHE heat')
    lines1 = [line1, line2, line3, line4]
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='center left', bbox_to_anchor=(0.32, 0.85))
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    line5, = ax2.plot(hydrogen/asr.max_hydrogen*100, Q_RHE_smoothed, label='Specific heat available', color='purple', linewidth=2.2)
    line6, = ax2.plot(hydrogen/asr.max_hydrogen*100, reactor_in_specific_heat_needed_smoothed, label='Specific heat needed', color='magenta', linestyle='--', linewidth=2.2)
    ax2.set_xlabel('Load [%]')
    ax2.set_ylabel('Heat [kJ/kg]')
    ax2.grid(True)
    ax2_twin = ax2.twinx()
    line7, = ax2_twin.plot(hydrogen/asr.max_hydrogen*100, bypass_RHE_smoothed, label='RHE bypass', color='crimson', linewidth=2.2)
    ax2_twin.set_ylabel('RHE bypass [%]')
    ax2_twin.set_ylim(np.min(bypass_RHE_smoothed)-1, np.max(bypass_RHE_smoothed)+1)
    ax2.set_title('Fixed bypass 14 % - Heat needed vs heat RHE')
    lines2 = [line5, line6, line7]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='center left', bbox_to_anchor=(0.32, 0.85))
    plt.show()
    
    # 3D plot - specific consumption VS load and fraction of hydrogen from buffer + section at hydrogen from buffer = 0 with contributions of components to the total consumption
    hyd_range = np.linspace(asr.min_hydrogen, asr.max_hydrogen, 30)
    hyd_buffer_range = np.linspace(0, 1, 30)
    p_buffer = 30
    hyd, hyd_buffer = np.meshgrid(hyd_range, hyd_buffer_range)
    consumption1 = np.zeros_like(hyd)
    consumption2 = np.zeros_like(hyd)
    consumption3 = np.zeros_like(hyd)
    consumption_simple = np.zeros_like(hyd)
    error_relative = np.zeros_like(hyd)

    for i in range(hyd.shape[0]):
        for j in range(hyd.shape[1]):
            hyd1 = hyd[i, j]
            hyd_buffer1 = hyd1 * hyd_buffer[i, j]
            ammonia, electricity, _, _, _ = asr.use(0, hyd1, hyd_buffer=hyd_buffer1, p_buffer=30) 
            consumption1[i, j] = abs(electricity) / (ammonia*3600)
            ammonia, electricity, _, _, _ = asr.use(0, hyd1, hyd_buffer=hyd_buffer1, p_buffer=90) 
            consumption2[i, j] = abs(electricity) / (ammonia*3600)
            ammonia, electricity, _, _, _ = asr.use(0, hyd1, hyd_buffer=hyd_buffer1, p_buffer=200) 
            consumption3[i, j] = abs(electricity) / (ammonia*3600)
            
    import matplotlib.ticker as mticker
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import Patch
    
    fig = plt.figure(figsize=(10, 5.5), dpi=150)
    
    zmin = 0.2
    zmax = 0.45
    
    ax = fig.add_subplot(111, projection='3d')

    surf3 = ax.plot_surface(hyd / asr.max_hydrogen * 100, hyd_buffer * 100, consumption3,
                            cmap='YlGn_r', alpha=0.9)
    surf2 = ax.plot_surface(hyd / asr.max_hydrogen * 100, hyd_buffer * 100, consumption2,
                            cmap='RdPu_r', alpha=0.9)
    surf1 = ax.plot_surface(hyd / asr.max_hydrogen * 100, hyd_buffer * 100, consumption1,
                            cmap='YlOrBr_r', alpha=0.9)
    
    ax.set_xlabel('Load [%]', fontsize=12)
    ax.view_init(elev=25, azim=60)
    ax.set_ylabel('H$_{2}$ from buffer [%]', fontsize=12)
    ax.set_zlabel('Spec. cons. [kWh/kg$_{NH_3}$]', fontsize=12)
    ax.set_zlim(zmin, zmax)
    ax.zaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    idx_zero = np.argmin(np.abs(hyd_buffer - 0))
    
    legend_patches = [
        Patch(color='orange', label='P$\mathrm{_{buffer}}$ < P$\mathrm{_{I stage}}$'),
        Patch(color='#fe02a2', label='P$\mathrm{_{I stage}}$ ≤ P$\mathrm{_{buffer}}$ < P$\mathrm{_{II stage}}$'),
        Patch(color='green', label='P$\mathrm{_{buffer}}$ ≥ P$\mathrm{_{II stage}}$')
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=13, bbox_to_anchor=(-0.55, 0.2))

    ax2d_inset = fig.add_axes([0.23, 0.95, 0.3, 0.4])
    
    x_data = hydrogen / asr.max_hydrogen * 100
    x_data_inv = x_data[::-1]  
    ax2d_inset.plot(x_data_inv, specific_consumption[::-1], label='Total', color='red', linewidth = 2)
    ax2d_inset.plot(x_data_inv, specific_consumption_refrigeration[::-1], label='Refrig.', color='k', linestyle='--')
    ax2d_inset.plot(x_data_inv, specific_consumption_compressors[::-1], label='Compr.', color='k', linestyle='-.')
    ax2d_inset.plot(x_data_inv, specific_consumption_electric_resistance[::-1], label='Elec. res.', color='k', linestyle=':')
    ax2d_inset.set_xlim(ax2d_inset.get_xlim()[::-1])
    ax2d_inset.set_xlim(100, 30)
    ax2d_inset.set_xticks(range(30, 101, 10))
    ax2d_inset.set_xlabel('Load [%]')
    ax2d_inset.set_ylabel('Spec. cons. [kWh/kg$_{NH_3}$]')
    ax2d_inset.set_title('H$_{2}$ from buffer = 0%', fontsize=12)
    ax2d_inset.tick_params(labelsize=12)
    ax2d_inset.grid(True, linestyle='--', alpha=0.4)
    ax2d_inset.legend(fontsize=12, loc='lower center', bbox_to_anchor=(-0.47, 0.3))
    
    x3d = 28.5  
    y3d = -15   
    z3d = 0.455   
    x2d_disp, y2d_disp, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
    point_3d_disp = ax.transData.transform((x2d_disp, y2d_disp))
    
    x2d = 30  
    y2d = -0.02
    point_2d_disp = ax2d_inset.transData.transform((x2d, y2d))
    
    fig_coords_3d = fig.transFigure.inverted().transform(point_3d_disp)
    fig_coords_2d = fig.transFigure.inverted().transform(point_2d_disp)
    
    from matplotlib.lines import Line2D
    line = Line2D([fig_coords_3d[0], fig_coords_2d[0]],
                  [fig_coords_3d[1], fig_coords_2d[1]],
                  transform=fig.transFigure,
                  color='black', linewidth=1)
    fig.add_artist(line)
    
    x3d = 72.5
    y3d = -20  
    z3d = 0.345 
    x2d_disp, y2d_disp, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
    point_3d_disp = ax.transData.transform((x2d_disp, y2d_disp))
    
    x2d = 100  
    y2d = -0.02
    point_2d_disp = ax2d_inset.transData.transform((x2d, y2d))
    
    fig_coords_3d = fig.transFigure.inverted().transform(point_3d_disp)
    fig_coords_2d = fig.transFigure.inverted().transform(point_2d_disp)
    
    from matplotlib.lines import Line2D
    line = Line2D([fig_coords_3d[0], fig_coords_2d[0]],
                  [fig_coords_3d[1], fig_coords_2d[1]],
                  transform=fig.transFigure,
                  color='black', linewidth=1)
    fig.add_artist(line)
    fig.tight_layout()
    fig.savefig("3d plot.svg", bbox_inches = "tight")
    plt.show()

    
#%% # Simple calculation VS extended calculation

    asr=ASR(inp_test, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, buffer_info=buffer_info, timestep=timestep)
    asr_simple=ASR(inp_test_simple, 8760, T_p_hyd, T_p_nitro, location_name, path, file_structure, file_general, buffer_info=buffer_info, timestep=timestep)
    
    # 0 % from the buffer - variable hydrogen flowrate
    hydrogen = np.linspace(asr.min_hydrogen, asr.max_hydrogen, 30)
    electricity = np.zeros(len(hydrogen))
    electricity_simple = np.zeros(len(hydrogen))
    relative_error = np.zeros(len(hydrogen))
    ammonia = np.zeros(len(hydrogen))

    i = 0
    fraction_from_buffer = 0
    for hyd in hydrogen:
        hyd_buffer = fraction_from_buffer * hyd
        ammonia[i], electricity[i], _, _, _=asr.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)  
        _, electricity_simple[i], _, _, _=asr_simple.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
        relative_error[i] = (abs(electricity[i]) - abs(electricity_simple[i])) / np.abs(electricity[i]) * 100
        i+=1

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(hydrogen, abs(electricity), label='Electricity', color='blue', marker='o')
    ax1.plot(hydrogen, abs(electricity_simple), label='Electricity simple', color='red', linestyle='--', marker='x')
    ax1.set_xlabel('Hydrogen [kg/s]')
    ax1.set_ylabel('Electricity [kW]')
    ax1.set_title(f'Comparison of electricity consumption - constant % from the buffer {fraction_from_buffer*100} %')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    ax2.plot(hydrogen, relative_error, color='darkgreen', linestyle=':', label='Relative Error')
    ax2.set_ylabel('Relative Error [%]')
    ax2.legend(loc='lower right')
    ax2.set_ylim(-10, 10)
    plt.show()
    
    
    # Constant hydrogen flowrate from the buffer - variable pressure
    hyd = asr.max_hydrogen
    hyd_buffer = hyd
    pressure_buffer = np.linspace(buffer_info[2], buffer_info[1], 30)
    electricity = np.zeros(len(pressure_buffer))
    electricity_simple = np.zeros(len(pressure_buffer))
    relative_error = np.zeros(len(pressure_buffer))

    i = 0
    for p_buffer in pressure_buffer:
        ammonia, electricity[i], _, _, _=asr.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
        ammonia, electricity_simple[i], _, _, _=asr_simple.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
        relative_error[i] = (abs(electricity[i]) - abs(electricity_simple[i])) / np.abs(electricity[i]) * 100
        i+=1
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(pressure_buffer, abs(electricity), label='Electricity', color='blue', marker='o')
    ax1.plot(pressure_buffer, abs(electricity_simple), label='Electricity simple', color='red', linestyle='--', marker='x')
    ax1.set_xlabel('Pressure from buffer [bar]')
    ax1.set_ylabel('Electricity [kW]')
    ax1.set_title(f'Comparison of electricity consumption - constant hydrogen flowrate from buffer {round(hyd, 4)} kg/s')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()  
    ax2.plot(pressure_buffer, relative_error, color='darkgreen', linestyle=':', label='Relative Error')
    ax2.set_ylabel('Relative Error [%]')
    ax2.legend(loc='lower left')
    ax2.set_ylim(-10, 10)
    plt.show()
    
    
    # Constant pressure - variable hydrogen flowrate from the buffer
    hyd = asr.min_hydrogen
    p_buffer = 30
    hydrogen_buffer = np.linspace(0, hyd, 30)
    electricity = np.zeros(len(hydrogen_buffer))
    electricity_simple = np.zeros(len(hydrogen_buffer))
    relative_error = np.zeros(len(hydrogen_buffer))

    i = 0
    for hyd_buffer in hydrogen_buffer:
        ammonia, electricity[i], _, _, _=asr.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
        ammonia, electricity_simple[i], _, _, _=asr_simple.use(0, hyd, hyd_buffer=hyd_buffer, p_buffer=p_buffer)
        relative_error[i] = (abs(electricity[i]) - abs(electricity_simple[i])) / np.abs(electricity[i]) * 100
        i+=1
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(hydrogen_buffer/hyd*100, abs(electricity), label='Electricity', color='blue', marker='o')
    ax1.plot(hydrogen_buffer/hyd*100, abs(electricity_simple), label='Electricity simple', color='red', linestyle='--', marker='x')
    ax1.set_xlabel('Hydrogen from buffer [%]')
    ax1.set_ylabel('Electricity [kW]')
    ax1.set_title(f'Comparison of electricity consumption - constant pressure from buffer {round(p_buffer, 4)} bar')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()  
    ax2.plot(hydrogen_buffer/hyd*100, relative_error, color='darkgreen', linestyle=':', label='Relative Error')
    ax2.set_ylabel('Relative Error [%]')
    ax2.legend(loc='lower left')
    ax2.set_ylim(-10, 10)
    plt.show()


#%% Reactor validation

    def reactor_bed_validation(mix_H2N2NH3, mix_massflowrate, N2_molarflowrate_i, H2_molarflowrate_i, NH3_molarflowrate_i, CH4_molarflowrate, Ar_molarflowrate, N2_molarflowrate0, T_i, XN2_i, p, n, reactor_bed_volume):
        
        p_atm=p * 0.986923 # conversion in [atm] 
        R = 1.987  # gas constant [cal/(mol*K)]
        # Coefficients for the correction of the conversion rate (eta) - depending on the pressure
        b0 = np.polyval(c.poly_coefficients['b0'], p_atm)
        b1 = np.polyval(c.poly_coefficients['b1'], p_atm)
        b2 = np.polyval(c.poly_coefficients['b2'], p_atm)
        b3 = np.polyval(c.poly_coefficients['b3'], p_atm)
        b4 = np.polyval(c.poly_coefficients['b4'], p_atm)
        b5 = np.polyval(c.poly_coefficients['b5'], p_atm)
        b6 = np.polyval(c.poly_coefficients['b6'], p_atm)
        
        def reactor_ode(V, y, mix_massflowrate, N2_molarflowrate0, CH4_molarflowrate, Ar_molarflowrate):
            N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XN2 = y  
            
            mix_molarflowrate = CH4_molarflowrate + Ar_molarflowrate + N2_molarflowrate + H2_molarflowrate + NH3_molarflowrate
            x_CH4 = CH4_molarflowrate / mix_molarflowrate
            x_Ar = Ar_molarflowrate / mix_molarflowrate
            x_N2 = N2_molarflowrate / mix_molarflowrate
            x_H2 = H2_molarflowrate / mix_molarflowrate
            x_NH3 = NH3_molarflowrate / mix_molarflowrate
            mix_H2N2NH3.set_mole_fractions([x_N2, x_H2, x_NH3, x_CH4, x_Ar])
            mix_H2N2NH3.update(CP.PT_INPUTS, p * 100000, T)

            # Resolution of the reaction kinetics
            "N2 + 3H2 --> 2NH3"
            k = np.exp (2.303 * 14.7102 - 39075 / (R * T))  # reaction rate constant [kmol_NH3/(h*m^3)] (Singh and Saraf, 1979)
            logK_eq = -2.691122 * np.log10(T) - 5.519265e-5 * T + 1.848863e-7 * T**2 + 2001.6 / T + 2.6899  # equilibrium constant [-], Arrhenius-type equation (Gillespie and Beattie, 1930)
            K_eq = 10**logK_eq  # reaction equilibrium constant
            
            # Fugacity coefficients of the mixture components
            fugacity_coefficient_N2 = mix_H2N2NH3.fugacity_coefficient(0)  #0.93431737 + 0.3101804e-3 * T + 0.295895e-3 * p - 0.270729e-6 * T**2 + 0.4775207e-6 * p**2                               
            fugacity_coefficient_H2 = mix_H2N2NH3.fugacity_coefficient(1)  #np.exp(np.exp(-3.8402 * T**0.125 + 0.541) * p - np.exp(-0.1263 * T**0.5 - 15.980) * p**2 + 300 * (np.exp(-0.011901 * T - 5.941)) * (np.exp(-p/300) - 1))              
            fugacity_coefficient_NH3 = mix_H2N2NH3.fugacity_coefficient(2) #0.1438996 + 0.2028538e-2 * T - 0.4487672e-3 * p - 0.1142945e-5 * T**2 + 0.2761216e-6 * p**2                                                                           

            # Activities = fugacity/reference pressure (reference pressure = 1 atm) (Lewis and Randall, 1923)
            # Fugacity = molar fraction * fugacity coefficient * p tot
            a_N2 = x_N2 * fugacity_coefficient_N2 * p_atm
            a_H2 = x_H2 * fugacity_coefficient_H2 * p_atm
            a_NH3 = x_NH3 * fugacity_coefficient_NH3 * p_atm

            # Ammonia production rate - modified form of Temkin equation (Dyson and Simon, 1968)
            alpha = 0.57  # kinetic fitting parameter [−], variable from 0.5 to 0.75
            r = k * (K_eq**2 * a_N2 * (a_H2**3 / a_NH3**2)**alpha - (a_NH3**2 / a_H2**3)**(1 - alpha))  # ammonia production rate [kmol_NH3/(h*m^3)]
            r = r * 1000 / 3600  # conversion in [mol_NH3/(s*m^3)]
            # Correction with the effectiveness factor eta - effect of resistance to transfer of mass or heat (Dyson and Simon, 1968)
            eta = b0 + b1 * T +  b2 * XN2 + b3 * T**2 + b4 * XN2**2 + b5 * T**3 + b6 * XN2**3  
            if eta < 0 or eta > 1:
                 raise ValueError(f'The effectiveness factor can only be between 0 and 1, the calculated value is {eta}')
            r = r * eta

            # Molar flowrate variations for m^3 
            dN2_molarflowrate_dV = -r / 2   # [mol/(s*m^3)], -r/2 because every mole of NH3 produced, half of N2 is consumed
            dH2_molarflowrate_dV = -3/2 * r   # [mol/(s*m^3)], -3/2r because every mole of NH3 produced, 3/2 of H2 is consumed
            dNH3_molarflowrate_dV = r  # [mol/(s*m^3)]
            # Nitrogen based conversion variation for m^3 (XN2 = (N2_in - N2_out) / N2_in)
            dXN2_dV = r / 2 / N2_molarflowrate0  # [1/m^3], r/2 = nitrogen consumed
            # Temperature variation for m^3
            heat_of_reaction = abs(4.184 * ((-0.5426 - 840.609 / T - 4.59734e8 / T**3) * p_atm - 5.34685 * T - 0.2525e-3 * T**2 + 1.69197e-6 * T**3 - 9157.09))  # [J/mol_NH3] (Gillespie and Beattie, 1930)     
            cp_mixture = mix_H2N2NH3.cpmass()    # [J/kg*K]
            dT_dV = heat_of_reaction * r / (mix_massflowrate * cp_mixture) # [K/m^3]

            return [dN2_molarflowrate_dV, dH2_molarflowrate_dV, dNH3_molarflowrate_dV, dT_dV, dXN2_dV]
        
        # Inputs for the solution of the ODE problem
        y0 = [N2_molarflowrate_i, H2_molarflowrate_i, NH3_molarflowrate_i, T_i, XN2_i]  # initial values 
        V_span = [0, reactor_bed_volume]            # interval of integration
        # Solution using method RK45
        sol = solve_ivp(reactor_ode, V_span, y0, method='BDF', args=(mix_massflowrate, N2_molarflowrate0, CH4_molarflowrate, Ar_molarflowrate), t_eval=np.linspace(0, reactor_bed_volume, n))
        N2_molarflowrate_final = sol.y[0]
        H2_molarflowrate_final = sol.y[1]
        NH3_molarflowrate_final = sol.y[2]
        T_final = sol.y[3]
        XN2_final = sol.y[4]
        
        def calculate_cp(N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T):
            mix_molarflowrate = CH4_molarflowrate + Ar_molarflowrate + N2_molarflowrate + H2_molarflowrate + NH3_molarflowrate
            x_CH4 = CH4_molarflowrate / mix_molarflowrate
            x_Ar = Ar_molarflowrate / mix_molarflowrate
            x_N2 = N2_molarflowrate / mix_molarflowrate
            x_H2 = H2_molarflowrate / mix_molarflowrate
            x_NH3 = NH3_molarflowrate / mix_molarflowrate
            mix_H2N2NH3.set_mole_fractions([x_N2, x_H2, x_NH3, x_CH4, x_Ar])
            mix_H2N2NH3.update(CP.PT_INPUTS, p * 100000, T)
            cp_mixture = mix_H2N2NH3.cpmass()
            
            return cp_mixture
        
        def calculate_HoR(T):
            heat_of_reaction = abs(4.184 * ((-0.5426 - 840.609 / T - 4.59734e8 / T**3) * p_atm - 5.34685 * T - 0.2525e-3 * T**2 + 1.69197e-6 * T**3 - 9157.09))
            
            return heat_of_reaction
        
        def calculate_r(N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XN2):
            mix_molarflowrate = CH4_molarflowrate + Ar_molarflowrate + N2_molarflowrate + H2_molarflowrate + NH3_molarflowrate
            x_CH4 = CH4_molarflowrate / mix_molarflowrate
            x_Ar = Ar_molarflowrate / mix_molarflowrate
            x_N2 = N2_molarflowrate / mix_molarflowrate
            x_H2 = H2_molarflowrate / mix_molarflowrate
            x_NH3 = NH3_molarflowrate / mix_molarflowrate
            mix_H2N2NH3.set_mole_fractions([x_N2, x_H2, x_NH3, x_CH4, x_Ar])
            mix_H2N2NH3.update(CP.PT_INPUTS, p * 100000, T)

            # Resolution of the reaction kinetics
            "N2 + 3H2 --> 2NH3"
            k = np.exp (2.303 * 14.7102 - 39075 / (R * T))  # reaction rate constant [kmol_NH3/(h*m^3)] (Singh and Saraf, 1979)
            logK_eq = -2.691122 * np.log10(T) - 5.519265e-5 * T + 1.848863e-7 * T**2 + 2001.6 / T + 2.6899  # equilibrium constant [-], Arrhenius-type equation (Gillespie and Beattie, 1930)
            K_eq = 10**logK_eq  # reaction equilibrium constant
            
            # Fugacity coefficients of the mixture components
            fugacity_coefficient_N2 = mix_H2N2NH3.fugacity_coefficient(0)  #0.93431737 + 0.3101804e-3 * T + 0.295895e-3 * p - 0.270729e-6 * T**2 + 0.4775207e-6 * p**2                                
            fugacity_coefficient_H2 = mix_H2N2NH3.fugacity_coefficient(1)   #np.exp(np.exp(-3.8402 * T**0.125 + 0.541) * p - np.exp(-0.1263 * T**0.5 - 15.980) * p**2 + 300 * (np.exp(-0.011901 * T - 5.941)) * (np.exp(-p/300) - 1))         
            fugacity_coefficient_NH3 = mix_H2N2NH3.fugacity_coefficient(2)   #0.1438996 + 0.2028538e-2 * T - 0.4487672e-3 * p - 0.1142945e-5 * T**2 + 0.2761216e-6 * p**2 #mix_H2N2NH3.fugacity_coefficient(2)                                                                            

            # Activities = fugacity/reference pressure (reference pressure = 1 atm) (Lewis and Randall, 1923)
            # Fugacity = molar fraction * fugacity coefficient * p tot
            a_N2 = x_N2 * fugacity_coefficient_N2 * p_atm
            a_H2 = x_H2 * fugacity_coefficient_H2 * p_atm
            a_NH3 = x_NH3 * fugacity_coefficient_NH3 * p_atm

            # Ammonia production rate - modified form of Temkin equation (Dyson and Simon, 1968)
            alpha = 0.57  # kinetic fitting parameter [−], variable from 0.5 to 0.75
            r = k * (K_eq**2 * a_N2 * (a_H2**3 / a_NH3**2)**alpha - (a_NH3**2 / a_H2**3)**(1 - alpha))  # ammonia production rate [kmol_NH3/(h*m^3)]
            r = r * 1000 / 3600  # conversion in [mol_NH3/(s*m^3)]
            # Correction with the effectiveness factor eta - effect of resistance to transfer of mass or heat (Dyson and Simon, 1968)
            eta = b0 + b1 * T +  b2 * XN2 + b3 * T**2 + b4 * XN2**2 + b5 * T**3 + b6 * XN2**3  
            r = r * eta
            
            return r/eta, eta 
        
        cp_mixture_values = [calculate_cp(N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T) for N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T in zip(N2_molarflowrate_final, H2_molarflowrate_final, NH3_molarflowrate_final, T_final)]
        HoR_values = [calculate_HoR(T) for T in T_final]
        r_eta_values = [calculate_r(N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XN2) for N2_molarflowrate, H2_molarflowrate, NH3_molarflowrate, T, XN2 in zip(N2_molarflowrate_final, H2_molarflowrate_final, NH3_molarflowrate_final, T_final, XN2_final)]
        r_values, eta_values = zip(*r_eta_values)
        
        return T_final, N2_molarflowrate_final, H2_molarflowrate_final, NH3_molarflowrate_final, XN2_final, cp_mixture_values, HoR_values, r_values, eta_values

    # Creation fo the mixture
    mix_H2N2NH3 = CP.AbstractState('PR',"Nitrogen&Hydrogen&Ammonia&Methane&Argon") 
    mix_H2N2NH3.set_binary_interaction_double(0,1,"kij",-0.036)   #from UNISIM
    mix_H2N2NH3.set_binary_interaction_double(0,2,"kij",0.222)    #from UNISIM
    mix_H2N2NH3.set_binary_interaction_double(1,3,"kij",0.202)    #from UNISIM
    mix_H2N2NH3.set_binary_interaction_double(2,4,"kij",-0.18)    #from UNISIM
    mix_H2N2NH3.set_binary_interaction_double(4,3,"kij",0.023)    #from UNISIM
    mix_H2N2NH3.set_binary_interaction_double(0,3,"kij",0.036)    #from UNISIM
    mix_H2N2NH3.specify_phase(CP.iphase_gas)
    
    # Input parameters for the model validation from: Jorqueira, Diogo Silva Sanches, Antonio Marinho Barbosa Neto, and Maria Teresa Moreira Rodrigues. "Modeling and numerical simulation of ammonia synthesis reactors using compositional approach." Advances in Chemical Engineering and Science 8.3 (2018): 124-143.
    p_react=226/0.986923
    T_in_Ibed=658.15
    T_in_IIbed=706.15
    T_in_IIIbed=688.15
    x_NH3=0.0276
    x_N2=0.2219
    x_H2=0.6703
    x_CH4=0.0546
    x_Ar=0.0256
    mix_massflowrate=29.821
    reactor_Ibed_volume=4.75
    reactor_IIbed_volume=7.2
    reactor_IIIbed_volume=7.8
    
    n=50
         
    # First bed
    mix_molarmass=x_N2*c.N2MOLMASS+x_H2*c.H2MOLMASS+x_NH3*c.NH3MOLMASS+x_CH4*CP.PropsSI("M", "CH4")+x_Ar*CP.PropsSI("M", "Argon")
    mix_molarflowrate=mix_massflowrate/mix_molarmass
    N2_molarflowrate_Ibed = mix_molarflowrate*x_N2
    N2_molarflowrate0 = N2_molarflowrate_Ibed
    H2_molarflowrate_Ibed = mix_molarflowrate*x_H2
    NH3_molarflowrate_Ibed = mix_molarflowrate*x_NH3
    CH4_molarflowrate = mix_molarflowrate*x_CH4
    Ar_molarflowrate = mix_molarflowrate*x_Ar
    XN2_Ibed = 0
    T_Ibed , N2_molarflowrate_Ibed, H2_molarflowrate_Ibed, NH3_molarflowrate_Ibed, XN2_Ibed, cp_mixture_Ibed, HoR_Ibed, r_Ibed, eta_Ibed = reactor_bed_validation(mix_H2N2NH3, mix_massflowrate, N2_molarflowrate_Ibed, H2_molarflowrate_Ibed, NH3_molarflowrate_Ibed, CH4_molarflowrate, Ar_molarflowrate, N2_molarflowrate0, T_in_Ibed, XN2_Ibed, p_react, n, reactor_Ibed_volume)
    print(f"The outlet temperature from the first bed is {round(T_Ibed[-1], 2)} K, the nitrogen conversion form the first bed is {round(XN2_Ibed[-1]*100,2)} %")
    x_N2_Ibed = N2_molarflowrate_Ibed[-1] / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_H2_Ibed = H2_molarflowrate_Ibed[-1] / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_NH3_Ibed = NH3_molarflowrate_Ibed[-1] / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_CH4_Ibed = CH4_molarflowrate / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_Ar_Ibed = Ar_molarflowrate / (N2_molarflowrate_Ibed[-1] + H2_molarflowrate_Ibed[-1] + NH3_molarflowrate_Ibed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    # Second bed
    mix_molarflowrate_IIbed = mix_massflowrate/(0.201*c.N2MOLMASS+0.61*c.H2MOLMASS+0.105*c.NH3MOLMASS+0.057*0.01604+0.027*0.039948)
    N2_molarflowrate_IIbed = N2_molarflowrate_Ibed[-1] #mix_molarflowrate_IIbed*0.201 
    H2_molarflowrate_IIbed = H2_molarflowrate_Ibed[-1] #mix_molarflowrate_IIbed*0.61 
    NH3_molarflowrate_IIbed = NH3_molarflowrate_Ibed[-1] #mix_molarflowrate_IIbed*0.105 
    XN2_IIbed = XN2_Ibed[-1] #0.1578 
    T_IIbed , N2_molarflowrate_IIbed, H2_molarflowrate_IIbed, NH3_molarflowrate_IIbed, XN2_IIbed, cp_mixture_IIbed, HoR_IIbed, r_IIbed, eta_IIbed = reactor_bed_validation(mix_H2N2NH3, mix_massflowrate, N2_molarflowrate_IIbed, H2_molarflowrate_IIbed, NH3_molarflowrate_IIbed, CH4_molarflowrate, Ar_molarflowrate, N2_molarflowrate0, T_in_IIbed, XN2_IIbed, p_react, n, reactor_IIbed_volume)
    print(f"The outlet temperature from the second bed is {round(T_IIbed[-1], 2)} K, the nitrogen conversion form the second bed is {round(XN2_IIbed[-1]*100,2)} %")
    x_N2_IIbed = N2_molarflowrate_IIbed[-1] / (N2_molarflowrate_IIbed[-1] + H2_molarflowrate_IIbed[-1] + NH3_molarflowrate_IIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_H2_IIbed = H2_molarflowrate_IIbed[-1] / (N2_molarflowrate_IIbed[-1] + H2_molarflowrate_IIbed[-1] + NH3_molarflowrate_IIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_NH3_IIbed = NH3_molarflowrate_IIbed[-1] / (N2_molarflowrate_IIbed[-1] + H2_molarflowrate_IIbed[-1] + NH3_molarflowrate_IIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_CH4_IIbed = CH4_molarflowrate / (N2_molarflowrate_IIbed[-1] + H2_molarflowrate_IIbed[-1] + NH3_molarflowrate_IIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_Ar_IIbed = Ar_molarflowrate / (N2_molarflowrate_IIbed[-1] + H2_molarflowrate_IIbed[-1] + NH3_molarflowrate_IIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    # Third bed
    mix_molarflowrate_IIIbed = mix_massflowrate/(0.182*c.N2MOLMASS+0.571*c.H2MOLMASS+0.159*c.NH3MOLMASS+0.061*0.01604+0.027*0.039948)
    N2_molarflowrate_IIIbed = N2_molarflowrate_IIbed[-1] #mix_molarflowrate_IIIbed*0.182 
    H2_molarflowrate_IIIbed = H2_molarflowrate_IIbed[-1] #mix_molarflowrate_IIIbed*0.571 
    NH3_molarflowrate_IIIbed = NH3_molarflowrate_IIbed[-1] #mix_molarflowrate_IIIbed*0.159 
    XN2_IIIbed = XN2_IIbed[-1] #0.2555 
    T_IIIbed , N2_molarflowrate_IIIbed, H2_molarflowrate_IIIbed, NH3_molarflowrate_IIIbed, XN2_IIIbed, cp_mixture_IIIbed, HoR_IIIbed, r_IIIbed, eta_IIIbed = reactor_bed_validation(mix_H2N2NH3, mix_massflowrate, N2_molarflowrate_IIIbed, H2_molarflowrate_IIIbed, NH3_molarflowrate_IIIbed, CH4_molarflowrate, Ar_molarflowrate, N2_molarflowrate0, T_in_IIIbed, XN2_IIIbed, p_react, n, reactor_IIIbed_volume) 
    print(f"The outlet temperature from the third bed is {round(T_IIIbed[-1], 2)} K, the nitrogen conversion form the third bed is {round(XN2_IIIbed[-1]*100,2)} %")
    x_N2_IIIbed = N2_molarflowrate_IIIbed[-1] / (N2_molarflowrate_IIIbed[-1] + H2_molarflowrate_IIIbed[-1] + NH3_molarflowrate_IIIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_H2_IIIbed = H2_molarflowrate_IIIbed[-1] / (N2_molarflowrate_IIIbed[-1] + H2_molarflowrate_IIIbed[-1] + NH3_molarflowrate_IIIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_NH3_IIIbed = NH3_molarflowrate_IIIbed[-1] / (N2_molarflowrate_IIIbed[-1] + H2_molarflowrate_IIIbed[-1] + NH3_molarflowrate_IIIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_CH4_IIIbed = CH4_molarflowrate / (N2_molarflowrate_IIIbed[-1] + H2_molarflowrate_IIIbed[-1] + NH3_molarflowrate_IIIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    x_Ar_IIIbed = Ar_molarflowrate / (N2_molarflowrate_IIIbed[-1] + H2_molarflowrate_IIIbed[-1] + NH3_molarflowrate_IIIbed[-1] + CH4_molarflowrate + Ar_molarflowrate)
    
    
    # Plots
    reactor_volume = np.concatenate([np.linspace(0, reactor_Ibed_volume, n), np.linspace(reactor_Ibed_volume, reactor_Ibed_volume+reactor_IIbed_volume, n), np.linspace(reactor_Ibed_volume+reactor_IIbed_volume, reactor_Ibed_volume+reactor_IIbed_volume+reactor_IIIbed_volume, n)])
    XN2 = np.concatenate([XN2_Ibed, XN2_IIbed, XN2_IIIbed])
    T = np.concatenate([T_Ibed, T_IIbed, T_IIIbed])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
    
    axs[0].plot(reactor_volume, T, linestyle='-', color='lightcoral', linewidth=2.5)
    axs[0].set_ylim(min(T) - 10, max(T) + 30)
    axs[0].set_xlim(0, max(reactor_volume))
    axs[0].set_xlabel("Reactor volume [m$^3$]", fontsize=12)
    axs[0].set_ylabel("Temperature [K]", fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].axvline(x=reactor_Ibed_volume, color='dimgrey', linestyle='--')
    axs[0].text(reactor_Ibed_volume -4, max(T) + 10, '1st bed', fontsize=12, color='black', va='center')
    axs[0].axvline(x=reactor_Ibed_volume + reactor_IIbed_volume, color='dimgrey', linestyle='--')
    axs[0].text(reactor_Ibed_volume + reactor_IIbed_volume - 5, max(T) + 10, '2nd bed', fontsize=12, color='black', va='center')
    axs[0].text(reactor_Ibed_volume + reactor_IIbed_volume + reactor_IIIbed_volume - 5, max(T) + 10, '3rd bed', fontsize=12, color='black', va='center')
    
    axs[1].plot(reactor_volume, XN2 * 100, linestyle='-', color='lightcoral', linewidth=2.5)
    axs[1].set_ylim(0, max(XN2) * 100 + 5)
    axs[1].set_xlim(0, max(reactor_volume))
    axs[1].set_xlabel("Reactor volume [m$^3$]", fontsize=12)
    axs[1].set_ylabel("N$_2$ conversion [%]", fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].axvline(x=reactor_Ibed_volume, color='dimgrey', linestyle='--')
    axs[1].text(reactor_Ibed_volume - 4, max(XN2) * 100 + 0.6, '1st bed', fontsize=12, color='black', va='center')
    axs[1].axvline(x=reactor_Ibed_volume + reactor_IIbed_volume, color='dimgrey', linestyle='--')
    axs[1].text(reactor_Ibed_volume + reactor_IIbed_volume - 5, max(XN2) * 100 + 0.6, '2nd bed', fontsize=12, color='black', va='center')
    axs[1].text(reactor_Ibed_volume + reactor_IIbed_volume + reactor_IIIbed_volume - 5, max(XN2) * 100 + 0.6, '3rd bed', fontsize=12, color='black', va='center')
    
    plt.tight_layout()
    plt.show()       

    





    












        