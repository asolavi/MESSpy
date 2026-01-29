import numpy as np
import os
import sys 
import math
from functools import lru_cache
from scipy.optimize import minimize
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir)))   # temporarily adding constants module path 
from CoolProp.CoolProp import PropsSI
from core import constants as c
 
@lru_cache(maxsize=None)
def part_load_eta(massflowrate_fraction):
    """
    Calculation of the part_load_eta (scaling factor for the isentropic efficiency) for a reciprocating 
    compressor based on the given massflowrate fraction, regulation method: suction valve unloading or clearance pocket.
    Uses a second-order polynomial fit based on literature curves.
    """
    # [Wang, L., et al. "Performance comparison of capacity control methods for reciprocating compressors." IOP Conference Series: Materials Science and Engineering. Vol. 90. No. 1. IOP Publishing, 2015.]
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
    # plt.figure(figsize=(8, 5))
    # plt.plot(m_fractions*100, eta, label=r"$\eta_{part} = \eta_{iso, off-design} / \eta_{iso, design}$", color='blue', linewidth=2)
    # plt.xlabel('Massflowrate fraction [%]', fontsize=12)
    # plt.ylabel('$\eta_{part}$ [-]', fontsize=12)
    # plt.title('Compressor partial load efficiency', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.ylim(0, 1.1)
    # plt.show()
    
    return part_load_eta
 

class PSA:
    
    def __init__(self, parameters, timestep_number, timestep=False):
        """
        Create a general PSA object.
        
        Air compression plus PSA operation for nitrogen separation. The key components of a PSA nitrogen system include air compressor, 
        pretreatment filters, adsorbent beds.
        For the ammonia reactor, a nitrogen purity of 99.999% is usually needed, as even small impurities of oxygen could interfere with the 
        catalyst and affect the reaction, not even impurities of argon must be present since no purge is considered inside the H-B loop.
        
        Inputs:
            parameters: dictionary
                'P_psa': operating PSA pressure [bar]
                'T_psa': operating PSA temperature [K]
                'comp efficiency': compressor politropic efficiency [-]
                'comp nstages': compressor number of stages
                'PSA efficiency': psa efficiency, nitrogen recovery rate (extracted nitrogen/available nitrogen in the air) [-]
            Nflowrate : maximum mass flow rate of nitrogen required by the downstream ammonia reactor [kg/s]
                    
        Outputs: PSA object able to:
            consume air and electricity and produce nitrogen .use(demand,timestep)
        """
        self.cost           = False  # to be updated via tech_cost() function   
        self.only_renewables = parameters['only_renewables']
            
        # Cycle parameters
        self.P_in = 1.01325                                    # [bar] inlet pressure for the cycle - ambient pressure
        self.P_psa = parameters['P_psa']
        self.T_psa = parameters['T_psa']  
        self.AF = parameters['air factor']                     # [-] air factor (volume of air needed / volume of nitrogen extracted)
        self.Nflowrate = parameters['Nflowrate']               # [Nm^3/h] nominal flowrate of nitrogen of the unit
        self.air_flow = self.Nflowrate * self.AF                    # [Nm^3/h] flowrate of air needed to produce the nominal flowrate of nitrogen for the given purity (99.999%)
        self.Nflowrate = self.Nflowrate / 3600 * PropsSI('D', 'T', 273.15, 'P', 101325, 'N2')  # [kg/s]
        self.air_flow = self.air_flow / 3600 * PropsSI('D', 'T', 273.15, 'P', 101325, 'air')  # [kg/s]
        self.AF = self.air_flow / self.Nflowrate                     # [-] air factor (mass of air needed / mass of nitrogen extracted) 
        self.n_units = parameters['number of units']            # number of units
        self.eta_pol = parameters['comp efficiency']    
        
        self.min_load = 0.3                                    # minimum load = 30 %
        self.nstages   = 2            
        self.T_in = c.AMBTEMP                                  # [K] inlet temperature for the cycle
        self.T_IC = c.AMBTEMP+10                               # [K] cooling with water at ambient temperature, DT = 10 K
        self.eta_motor = 0.95                                  # [-] assumed efficiency of electric motor driving the compressor https://transitionaccelerator.ca/wp-content/uploads/2023/04/TA-Technical-Brief-1.1_TEEA-Hydrogen-Compression_PUBLISHED.pdf

        self.n_units_used=np.zeros(timestep_number)

        if timestep == False: 
            self.timestep   = c.timestep              # [min]       simulation timestep if launched from main
        else:
            self.timestep   = timestep                # [min]       simulation timestep if launched from psa.py
               

        # Initialisation of variables - thermodynamic points of compression process
        self.P_points   = np.zeros(self.nstages*2+1) # [bar] Pressure
        self.T_points   = np.zeros(self.nstages*2+1) # [K] Temperature
        self.h_points   = np.zeros(self.nstages*2+1) # [kJ/kg] Enthalpy
        self.s_points   = np.zeros(self.nstages*2+1) # [kJ/kgK] Entropy
        self.rho_points = np.zeros(self.nstages*2+1) # [kg/m^3] Density
        
        self.comp_lav_spec=np.zeros(self.nstages) 
                    
        # Thermodynamic and chemical properties
        self.CP_air         = c.CP_AIR                      # [kJ/kgK]
        self.CV_air         = c.CV_AIR                      # [kJ/kgK]
        self.R_univ         = c.R_UNIVERSAL                 # [J/mol*K] Molar ideal gas constant                 
        self.M_air          = c.AIRMOLMASS*1000             # [kg/kmol]
        self.R_air          = self.R_univ/(self.M_air/1000) # [J/kgK]
        
        # Calculation of polytropic exponents for air 
        self.beta           = self.P_psa/self.P_in                      # [-] compression ratio 
        self.gamma_air      = self.CP_air/self.CV_air                   # [-]
        self.epsilon_air    = (self.gamma_air-1)/self.gamma_air         # [-]
        self.exp_pol_air    = (self.eta_pol/self.epsilon_air)/(self.eta_pol/self.epsilon_air-1) # [-]
        self.omega_air      = (self.exp_pol_air-1)/self.exp_pol_air     # [-]
        
        ####################################################################################
        # Inlet conditions
        self.P_points[0]=self.P_in
        self.T_points[0]=self.T_in
        self.h_points[0]=PropsSI('H', 'P', self.P_points[0]*100000, 'T', self.T_points[0], 'air')/1000
        self.s_points[0]=PropsSI('S', 'P', self.P_points[0]*100000, 'T', self.T_points[0], 'air')/1000
        self.rho_points[0]=PropsSI('D', 'P', self.P_points[0]*100000, 'T', self.T_points[0], 'air')
           
        # Compressor operation with interrefrigeration
        self.comp_beta_targ = self.beta**(1/self.nstages)
        self.eta_is = (self.comp_beta_targ**self.epsilon_air - 1)/(self.comp_beta_targ**(self.epsilon_air/self.eta_pol)-1)
        i=0      
        
        for n in range(self.nstages):
            i+=1 
            self.P_points[i]  = self.P_points[i-1]*self.comp_beta_targ
            s_iso = self.s_points[i-1]
            h_iso = PropsSI('H', 'P', self.P_points[i]*100000, 'S', s_iso*1000, 'air')/1000     # [kJ/kg]
            self.h_points[i]= (h_iso - self.h_points[i-1]) / self.eta_is + self.h_points[i-1]   # [kJ/kg]    
            self.T_points[i] = PropsSI('T', 'P', self.P_points[i]*100000, 'H', self.h_points[i]*1000, 'air')
            self.s_points[i]  = PropsSI('S', 'P', self.P_points[i]*100000, 'H', self.h_points[i]*1000, 'air')/1000
            self.rho_points[i] = PropsSI('D', 'P', self.P_points[i]*100000, 'H', self.h_points[i]*1000, 'air')
            
            self.comp_lav_spec[n] = (self.h_points[i]-self.h_points[i-1])    # [kJ/kg] specific work of compression given in and out conditions
            
            # Interrefrigeration - refrigeration also after the last compressor at the PSA working temperature
            i+=1
            if i == self.nstages*2:
                self.T_points[i]=self.T_psa
            else:
                self.T_points[i]=self.T_IC
            self.P_points[i]=self.P_points[i-1]
            self.h_points[i]=PropsSI('H', 'P', self.P_points[i]*100000, 'T', self.T_points[i], 'air')/1000
            self.s_points[i]=PropsSI('S', 'P', self.P_points[i]*100000, 'T', self.T_points[i], 'air')/1000
            self.rho_points[i]=PropsSI('D', 'P', self.P_points[i]*100000, 'T', self.T_points[i], 'air')
         
        self.Npower = self.air_flow * self.n_units * np.sum(self.comp_lav_spec) / self.eta_motor 
            
        # print(f"\nPSA air compressor with a nominal power of {int(self.Npower)} kW to produce a max flow rate of {round(self.Nflowrate * self.n_units,3)} kg/s of nitrogen")
                    
                        
            
    def use(self, nitro, step):
        '''
        The use function computes the electricity consumption and the nitrogen produced
        
        Inputs: 
            nitro: (>0) nitrogen required [kg/s]
            
        Outputs: 
            nitro: nitrogen produced [kg/s] 
            electricity: energy consumed [kW]
        '''
        
        # Minimum flowrate for the unit
        min_flowrate = self.Nflowrate * self.min_load
        max_flowrate = self.Nflowrate * self.n_units
        if nitro < min_flowrate:
            nitro = 0
            electricity = 0
            return (nitro,-electricity)
        elif nitro > max_flowrate:
            nitro = max_flowrate    # just the maximum capacity of the PSA units can be produced
        else:
            nitro = nitro    
                                                 
        air_flow = nitro * self.AF            # [kg/s] quantity of air required to produce necessary nitrogen - air factor considered constant at part load
        electricity = air_flow * sum(self.comp_lav_spec) / self.eta_motor / part_load_eta(air_flow/(self.air_flow*self.n_units))   # [kW]               
        
        return (nitro,-electricity)
       

           
    def tech_cost(self,tech_cost): 
        """
        Inputs:
            tech_cost: dict
                'cost per unit': [€/(kg_N2/s)]
                'OeM': percentage on initial investment [%]
                'refud': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of the initial investment [%]
                    'years': after how many years it will be replaced

        Outputs:
            self.cost: dict
                'total cost': [€]
                'OeM': percentage on initial investment [%]
                'refud': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of the initial investment [%]
                    'years': after how many years it will be replaced
        """
        tech_cost = {key: value for key, value in tech_cost.items()}
        
        if tech_cost['cost per unit'] == 'default price correlation':
            # https://www.compressorworld.com/products/140791/pneumatech-ppng-100-800-he-psa-nitrogen-generator
            capacity = [45.9, 57.7, 67.6, 87.4, 107, 126.6, 155.1, 185.5, 232.3, 294.7, 374.7]    # [m^3/h]     
            capacity = np.array(capacity) * PropsSI('D', 'P', 101325, 'T', 293.15, 'nitrogen')    # [kg/h]
            price =  [182372, 207550, 244496, 277832, 311168, 335579, 372839, 414457, 483544, 566330, 663088]     # [$]
            exchange_rate = 0.95
            price = np.array(price) * exchange_rate  # [€]
            coefficients = np.polyfit(capacity, price, 3)
            polynomial = np.poly1d(coefficients)
            PSA_cost = polynomial(self.Nflowrate*3600) * self.n_units
            
            # capacity_fit = np.linspace(min(capacity), max(capacity), 100)
            # price_fit = polynomial(capacity_fit)
            # plt.figure(figsize=(8, 5), dpi=150)
            # plt.scatter(capacity, price / 1e3, color='lightcoral', label='Original data')
            # plt.plot(capacity_fit, price_fit / 1e3, color='darkturquoise', label='Fitting')
            # plt.xlabel('PSA size [kg$_{N2}$/h]')
            # plt.ylabel('Cost [k€]')
            # plt.title('PSA unit cost (Pneumatech PPNG 150-800 HE)')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            # Compressor cost, Ref: Shamoushaki, M., et al. "Development of Cost Correlations for the Economic Assessment of Power Plant Equipment. Energies 2021, 14, 2665."
            a = 0.04147
            b = 454.8
            c = 1.81 * 10**5
            exchange_rate = 0.95
            CEPCI_2021 = 708.8
            CEPCI_2024 = 800
            I_stage_Npower = self.comp_lav_spec[0] * self.air_flow * self.n_units
            II_stage_Npower = self.comp_lav_spec[1] * self.air_flow * self.n_units
            compressor_cost = (np.log(I_stage_Npower) + a * I_stage_Npower**2 + b * I_stage_Npower + c) + (np.log(II_stage_Npower) + a * II_stage_Npower**2 + b * II_stage_Npower + c)
            intercoolers_cost = 2.5   # % of compressor cost, Ref: Luyben, William L. "Capital cost of compressors for conceptual design." Chemical Engineering and Processing-Process Intensification 126 (2018): 206-209.
            compressor_cost = compressor_cost * (1 + intercoolers_cost / 100) * exchange_rate * CEPCI_2024 / CEPCI_2021   
                  
            BoP_cost = 55   # % of PSA and compressor cost, Ref: Gomez, Jamie R., John Baca, and Fernando Garzon. "Techno-economic analysis and life cycle assessment for electrochemical ammonia production using proton conducting membrane." International Journal of Hydrogen Energy 45.1 (2020): 721-737.
            C = (PSA_cost + compressor_cost) * (1 + BoP_cost / 100)
        else:
            C = tech_cost['cost per unit'] * self.Nflowrate * self.n_units
        
        tech_cost['total cost'] = tech_cost.pop('cost per unit')
        tech_cost['total cost'] = C
        tech_cost['OeM'] = tech_cost['OeM'] *C /100 # [€]
        
        self.cost = tech_cost
        

        
        

#%%%############

if __name__ == "__main__":
    
    'PSA test'

    inp_test = {'P_psa'             : 7,
                'T_psa'             : 298.15,
                'only_renewables'   : False,
                 'comp efficiency'   : 0.85, 
                 'air factor'        : 5.2,
                  'Nflowrate'         : 380,
                  'number of units'   : 1,                                        		       
                 'priority'          : 6}
    
    tech_cost = { "cost per unit": "default price correlation",
            "OeM": 2, 
            "refund": { "rate": 0, "years": 0},
            "replacement": {"rate": 50, "years": 15}}

    sim_steps   = 5      # [-] number of steps to be considered for the simulation - usually a time horizon of 1 year minimum is considered
    timestep    = 60      # [min] selected timestep for the simulation
        


    psa  = PSA(inp_test,sim_steps, timestep=timestep)  # creating compressor object
    a,b = psa.use(0.132,1)
    # print(b/(0.132*3600))
    psa.tech_cost(tech_cost)
    
    


        

        
        
        
    
   
        
            
            
            
            