import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys 
from scipy.interpolate import interp1d
from CoolProp.CoolProp import PropsSI
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.path.pardir)))  
from core import constants as c

class CCGT:    
    
    def __init__(self, parameters, path, timestep_number):
        
        """
        Create a combined cycle GT object
    
        Inputs: 
            Parameters: dictionary  
              "Fuel": type of fuel used
              'Npower' : nominal power [kW]
              'efficiency': efficiency of the CCGT [-]
        
                      
        Outputs: CCGT object able to:
              consume fuel and produce electricity .use(step, demand)

        """
        self.cost = False # will be updated with tech_cost()

        self.fuel = parameters["Fuel"]
        if self.fuel not in ["ammonia", "hydrogen"]:
            raise ValueError("Only H2 or NH3 fuelled GT available as combined cycle system.")
        if self.fuel == "ammonia":
            self.LHV = 18.6 * 1000 # [kJ/kg]
        if self.fuel == "hydrogen":
            self.LHV = 120 * 1000 # [kJ/kg]
        self.efficiency = parameters["efficiency"] 
        self.Npower = parameters["Npower"]

        # Operation constraints
        self.Wel_max= self.Npower*1      # [kW] Max Electric Power Output - 100% of Nominal Power
        self.Wel_min= self.Npower*parameters.get("min_load",0)  # [kW] Min Electric Power Output     
     
    
    def use(self, step, electricity_demand, fuel_available):
        """
        The CCGT system consumes fuel and produces electricity.
        
        Inputs:
            step: step to be simulated
            electricity_demand: (<0) electricity demand [kW]
            fuel_available: fuel available [kg/s]
      
        Outputs: 
            fuel_consumed: fuel consumption [kg/s]
            electricity: electricity production [kW] 
        """  

        electricity_demand = abs(electricity_demand)
            
        if electricity_demand < self.Wel_min:
            electricity = 0
            fuel_consumed = 0
            return(-fuel_consumed, electricity)
        if electricity_demand > self.Wel_max:
            electricity_demand = self.Wel_max
        
        fuel_required = electricity_demand  / (self.efficiency * self.LHV)

        if fuel_required <= fuel_available:
            fuel_consumed = fuel_required
            electricity = electricity_demand
        else:
            fuel_consumed = fuel_available
            electricity = fuel_available * self.LHV * self.efficiency 
            if electricity < self.Wel_min:
                electricity = 0
                fuel_consumed = 0

        return(-fuel_consumed, electricity)
    
    
        
    def tech_cost(self,tech_cost):
        """
        Inputs:
            tech_cost: dict
                'cost per unit': [€/kW]
                'OeM': operation and maintenance costs, percentage on initial investment [%]
                'refund': dict
                    'rate': percentage of initial investment which will be rimbursed [%]
                    'years': years for reimbursment
                'replacement': dict
                    'rate': replacement cost as a percentage of the initial investment [%]
                    'years': after how many years it will be replaced

        Outputs:
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
        
        size = self.Npower

        if tech_cost['cost per unit'] == 'default price correlation':
           exchange_rate = 0.95                                # exchange rate between USD and €
           if self.fuel == "ammonia": 
              tech_cost['cost per unit'] = 1250 * exchange_rate   # [€/kW], Ref: Shepherd, Jack, et al. "Open-source project feasibility tools for supporting development of the green ammonia value chain." Energy Conversion and Management 274 (2022): 116413.
           if self.fuel == "hydrogen":
               tech_cost['cost per unit'] = 1250 * exchange_rate 
        C = tech_cost['cost per unit'] * size 

        tech_cost['total cost'] = tech_cost.pop('cost per unit')
        tech_cost['total cost'] = C
        tech_cost['OeM'] = tech_cost['OeM'] *C /100 # [€]
        self.cost = tech_cost
        
#%%##########################################################################################

if __name__ == "__main__":
    
    """
    Functional test
    """
    
    inp_test = {"Npower": 11000,
                "Fuel"      : "ammonia",
                "efficiency": 0.5}       
    
    simulation_hours = 8760                   # 1 year-long simulation
    CCGT = CCGT(inp_test, simulation_hours)   # creating CCGT object

    available_fuel = 99999999   # [kg]
    el_demand= 400
    fuel_consumption = CCGT.use(0, el_demand, available_fuel)[0]
    print(fuel_consumption)


