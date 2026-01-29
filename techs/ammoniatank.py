import numpy as np
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir)))    
from core import constants as c
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt

class NH3_tank:    
    
    def __init__(self,parameters,timestep_number):
        
        """
        Create a NH3_tank object. Ammonia storage system in tanks. 
        
        Inputs:
            parameters: dictionary
                'max capacity': tank capacity [kg]
                'tank model': storage technology
                    - pressurized: up to 270/1500 ton @ 16-18bar 298.15K
                    - semirefrigerated: 450 - 2700 ton @ 3-5bar 273.15K
                    - refrigerated: 4550 - 50000 ton @ 1.1-1.2bar 240.15K
            timestep_number: number of the simulation timesteps 
                      
        Outputs: NH3 tank object able to:
            supply or absorb ammonia .use(step,nh3)
            keep track of the level of charge .LOC
        """
       
        tank_parameters ={"pressurized":{"pressure": 18,
                                    "temperature": c.AMBTEMP},
                     "semiref":{"pressure": 5,
                                    "temperature": 273.15},
                     "refrigerated":{"pressure": 1.1,
                                     "temperature": 240.15}}        # dict of parameters for the different storage technologies
        
        self.model = parameters['tank model']
        self.pressure = tank_parameters[self.model]['pressure']
        self.temperature = tank_parameters[self.model]['temperature']
        self.cost = False # will be updated with tec_cost()
        self.timestep       = 60#c.timestep                         # [min] selected timestep for simulation        
        self.LOC            = np.zeros(timestep_number+1)           # [kg] array keeping trak ammonia tank level of charge 
        self.max_capacity   = parameters['max capacity']            # [kg] NH3 tank max capacity 
        self.used_capacity  = 0                                     # [kg] NH3 tank used capacity <= max_capacity        
        self.density        = PropsSI('D', 'P', self.pressure*100000 , 'T', self.temperature, 'ammonia')                                   # [kg/m^3] liquid ammonia density 
        if self.max_capacity:
            self.tank_volume = round(self.max_capacity/self.density,2)   # [m^3] tank volume
            
    def use(self,step,nh3,constant_demand=False):
         """
         Ammonia tank can supply or absorb hydrogen
         
         Inputs:
             step: step to be simulated
             nh3: ammonia requested (nh3<0) or provided (nh3>0) [kg/s]
       
         Outputs: 
             charge_flow_rate or discharge_flow_rate: ammonia absorbed or supplied in the timestep [kg/s]
         """
         
         nh3 = nh3*self.timestep*60      # [kg] conversion from kg/s to kg for the considered timestep
         
         if self.max_capacity:           # if tank size has been defined when setting up the case study
             if nh3 >= 0:                # [kg] if nh3 has positive value it means excess ammonia is being produced by the system. Ammonia tank is charged
                 charge = min(nh3,self.max_capacity-self.LOC[step])  # [kg] how much ammonia can NH3 tank absorb? Minimum between ammonia produced and available capacity in tank -> maximum capacity - Level of Charge (always a positive value, promptly shifted above 0 every time it goes beyond)
                 self.LOC[step+1] = self.LOC[step]+charge            # [kg] charge NH3 tank
                 
                 """
                 EXPLANATION self.used_capacity parameter 
                 self.used_capacity -> parameter in which is stored the memory of the tank charging story.
                                       It is not representative of the used capacity at the considered timestep.
                                       It represents the maximum level reached inside the tank up to time step of the simulation.
                                       Once self.used_capacity reaches the value of self.max.capacity it is no longer possible to allow 
                                       the tank's LOC to move towards negative values depending on the ammonia demand.
                                       From this point onwards, the value of self.used_capacity remains the same until the end of the simulation.
                 """
                                        
                 if self.LOC[step+1] > self.used_capacity:           # update used capacity
                     self.used_capacity = self.LOC[step+1]           # [kg]
                     
                 charge_flow_rate = charge/(self.timestep*60)        # [kg/s] converting the amount of ammonia stored into a flow rate
                 return(-charge_flow_rate)                           # [kg/s] return ammonia absorbed 
                 
             else: # if nh3 value is negative. Discharge NH3 tank (this logic allows to back-calculate LOC[0], it's useful for long term storage systems)
                 if (self.used_capacity==self.max_capacity):         # the tank max_capacity has been reached in this step, so LOC[step+1] can't become negative anymore in case requested ammonia is greater than available value
                     discharge           = min(-nh3,self.LOC[step])  # how much ammonia can NH3 tank supply?
                     self.LOC[step+1]    = self.LOC[step]-discharge  # discharge NH3 tank
             
                 else:   # the max_capacity has not yet been reached, so LOC[step+1] may become negative and then the past LOC can be shifted upwards to positive values. 
                         # The history of LOC during simulation is created by taking the self.used_capacity parameter into account. 
                         # Once the maximum capacity is reached for the first time, no more shifts to negative values are permitted and LOC
                         # remains the only parameter to represent the actualammonia amount in side the tank.
                                                       
                     discharge = min(-nh3,self.LOC[step]+self.max_capacity-self.used_capacity)   # [kg] ammonia that can be supplied by NH3 tank at the considered timestep
                     self.LOC[step+1] = self.LOC[step]-discharge                                 # [kg] discharge NH3_tank
                     if self.LOC[step+1] < 0:                                                    # if the level of charge has become negative
                         self.used_capacity  += - self.LOC[step+1]                               # increase the used capacity
                         self.LOC[:step+2]   += - self.LOC[step+1]                               # shift the past LOC array
                 
                 discharge_flow_rate = discharge/(self.timestep*60)                              # [kg/s] converting the amount of ammonia to be dischrged into a flow rate
                 return(discharge_flow_rate)                                                     # [kg/s] return ammonia supplied 
             
         else:           # This option is activated when ammonia tank is sized at the end of simulation as a result of 'supply-led' operation strategy.
                         # A costant mass-flow rate demand is created based on the cumulative production of ASR througout the year. 
                         # Tank size in this case smoothes surplus or deficit of production during operation, allowing for a constant rate deliver. 
             
             constant_demand     = constant_demand*self.timestep*60              # [kg] transforming kg/s into kg for ammonia demand
             charge              = nh3 - constant_demand                         # [kg] for the considered strategy nh3[step] is either positive or 0, representing the ammonia produced by the ASR at each timestep. 
             self.LOC[step+1]    = self.LOC[step]+charge                         # [kg] charge NH3 tank
             if step == (len(self.LOC)-2):                                       # at the end of simulation. LOC array has self.sim_timesteps+1 values
                 self.max_capacity   = max(self.LOC)+abs(min(self.LOC))          # [kg] max tank capacity
                 self.shift          = abs(min(self.LOC))                        # [kg] ammonia amount in storage at time 0
                 self.LOC            = self.LOC + self.shift                     # shifting the Level Of Charge curve to avoid negative minimum value (minimum is now at 0kg)
                                                                                 # It is now possible to define how much ammonia must be present in storage at the beginning of simulation. 
                 self.tank_volume    = round(self.max_capacity/self.density,2)   # [m^3] tank volume   
              
             charge_flow_rate = charge/(self.timestep*60)                        # [kg/s] converting the amount of ammonia stored into a flow rate                                     
             return(charge_flow_rate)                                            # [kg/s] return ammonia absorbed    
   
    def tech_cost(self,tech_cost):
        """
        Inputs:
            tech_cost: dict
                'cost per unit': [€/kg]
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

        size = self.max_capacity
        
        if self.model == 'pressurized':
            if tech_cost['cost per unit'] == 'default price correlation':
                exchange_rate = 0.95                                # exchange rate between USD and €
                tech_cost['cost per unit'] = 5 * exchange_rate      # [€/kg_NH3], Ref: Bose, Abhishek, et al. "Spatial variation in cost of electricity-driven continuous ammonia production in the United States." ACS Sustainable Chemistry & Engineering 10.24 (2022): 7862-7872.
            C = tech_cost['cost per unit'] * size 
   
        elif self.model == 'refrigerated': 
             if tech_cost['cost per unit'] == 'default price correlation':
             # This correlation already includes all the direct and indirect costs
                 exchange_rate = 0.95                               # exchange rate between USD and €
                 CEPCI_2010 = 550.8
                 CEPCI_2024 = 800
                 C = (46660 * (size/1000)**(-0.8636) + 536.9) * exchange_rate * size / 1000      # [€], Ref: Fasihi, Mahdi, et al. "Global potential of green ammonia based on hybrid PV-wind power plants." Applied Energy 294 (2021): 116170.
                 C = C * CEPCI_2024 / CEPCI_2010  
             else:
                 C = size * tech_cost['cost per unit']                            

        tech_cost['total cost'] = tech_cost.pop('cost per unit')
        tech_cost['total cost'] = C
        tech_cost['OeM'] = tech_cost['OeM'] *C /100 # [€]
        self.cost = tech_cost    
             
             
             

'----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
 


         
if __name__ == "__main__":             
             

    inp_test ={     "tank model"       : "pressurized",
           		     "max capacity"      : 140000,                                           		      
           		     "priority"          : 6}
    tech_cost = {"cost per unit": "default price correlation",
               "equipment OeM": 2, 
               "facility OeM": 2,
               "OeM": 2,
               "other costs" : 23,           
               "refund": { "rate": 0, "years": 0},
               "replacement": {"rate": 100, "years": 30}}
    
    
    timestep    = 60
    
    ammoniatank = NH3_tank(inp_test, timestep)
    ammoniatank.tech_cost(tech_cost)
             
             