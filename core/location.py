
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from techs import (heatpump, boiler_el, boiler_ng, boiler_h2, PV, wind, battery, H_tank, HPH_tank, O2_tank, fuel_cell, electrolyzer, inverter, chp_gt, Chp, Absorber, mhhc_compressor, Compressor, SMR, PSA, ASR, cracker, NH3_tank, CCGT)
from core import constants as c
import math

class location:
    
    def __init__(self,system,location_name,path,check,file_structure,file_general):
        """
        Create a location object (producer, consumer or prosumer) 
    
        system: dictionary (all the inputs are optional)
            'demand'': dictionary
                'electricity':              str 'file_name.csv' time series of electricity demand                               [kW]
                'heating water':            str 'file_name.csv' time series of heating and dhw demand                           [kW]
                'cooling water':            str 'file_name.csv' time series of cooling demand                                   [kW]
                'process heat':             str 'file_name.csv' time series of process heat demand                              [kW]
                'process hot water':        str 'file_name.csv' time series of process hot water demand                         [kW]
                'process cold water':       str 'file_name.csv' time series of process cold water demand (absorber, 7-12 °C)    [kW]
                'process chilled water':    str 'file_name.csv' time series of process chilled water demand (absorber, 1-5 °C)  [kW]
                'hydrogen':                 str 'file_name.csv' time series of hydrogen demand                                  [kg/s]
                'ammonia':                  str 'file_name.csv' time series of ammonia demand                                   [kg/s]
                'HP hydrogen':              str 'file_name.csv' time series of High-Pressure hydrogen demand                    [kg/s]
                'process steam':            str 'file_name.csv' time series of process steam demand                             [kg/s]
                'gas':                      str 'file_name.csv' time series of gas demand                                       [Sm3/s]
            'PV':                       dictionary parameters needed to create PV object (see PV.py)
            'inverter':                 dictionary parameters needed to create inverter object (see inverter.py)
            'wind':                     dictionary parameters needed to create wind object (see wind.py)
            'battery':                  dictionary parameters needed to create battery object (see battery.py)
            'electrolyzer':             dictionary parameters needed to create electrolyzer object (see electrolyzer.py)
            'H tank':                   dictionary parameters needed to create H_tank object (see hydrogentank.py)
            'O2 tank':                  dictionary parameters needed to create O2_tank object (see oxygentank.py)
            'HPH tank':                 dictionary parameters needed to create High Pressure H_tank object (see hydrogentank.py)
            'heatpump':                 dictionary parameters needed to create heat pump object (see heatpump.py)
            'boiler_ng':                dictionary parameters needed to create fuel cell object (see boiler.py)
            'boiler_el':                dictionary parameters needed to create fuel cell object (see boiler.py)
            'boiler_h2':                dictionary parameters needed to create fuel cell object (see boiler.py)
            'chp_gt':                   dicitonary parameters needed to create a chp object based on gas turbine technoology (see chp_gt.py)
            'mhhc compressor':          dicitonary parameters needed to create a mhhc object (see mhhc compressor.py)
            'mechanical compressor':    dicitonary parameters needed to create a mechanical object (see compressor.py)
            'SMR':                      dicitonary parameters needed to create a mechanical object (see steam_methane_reformer.py)
            'PSA':                      dicitonary parameters needed to create a pressure swing adsorption object (see psa.py)
            'ASR':                      dicitonary parameters needed to create an ammonia synthesis reactor object (see ASR.py)
            'NH3 tank':                 dicitonary parameters needed to create an ammonia tank object (see ammoniatank.py)   
            'CCGT':                     dicitonary parameters needed to create a combined cycle GT object (see CCGT.py)
            'cracker':                  dicitonary parameters needed to create a cracker object (see cracker.py)
            'fuel cell':                dicitonary parameters needed to create an ammonia tank object (see fuelcell.py)
            
        output : location object able to:
            simulate the energy flows of present technologies .loc_simulation
            record power balances (electricity, heating water, cooling water, gas, hydrogen, ammonia, nitrogen)
        """
        
        self.system = dict(sorted(system.items(), key=lambda item: item[1]['priority'])) # ordered by priority
        self.name = location_name
        self.technologies = {}                                  # initialise technologies dictionary
        self.power_balance = {                                  # initialise power balances dictionaries
                                'electricity'            : {},  # [kW]
                                'heating water'          : {},  # [kW]
                                'cooling water'          : {},  # [kW]
                                'process heat'           : {},  # [kW]
                                'process hot water'      : {},  # [kW]
                                'process cold water'     : {},  # [kW]
                                'process chilled water'  : {},  # [kW]
                                'hydrogen'               : {},  # [kg/s]
                                'LP hydrogen'            : {},  # [kg/s]
                                'HP hydrogen'            : {},  # [kg/s]
                                'oxygen'                 : {},  # [kg/s]
                                'process steam'          : {},  # [kg/s]
                                'gas'                    : {},  # [Sm^3/s]
                                'water'                  : {},  # [m^3/s]
                                'nitrogen'               : {},  # [kg/s]
                                'ammonia'                : {}}  # [kg/s]
                
                         
        self.consumption = {                                  # initialise consumption dictionaries
                                'electricity'            : {},  # [kW]
                                'heating water'          : {},  # [kW]
                                'cooling water'          : {},  # [kW]
                                'process heat'           : {},  # [kW]
                                'process hot water'      : {},  # [kW]
                                'process cold water'     : {},  # [kW]
                                'process chilled water'  : {},  # [kW]
                                'hydrogen'               : {},  # [kg/s]
                                'LP hydrogen'            : {},  # [kg/s]
                                'HP hydrogen'            : {},  # [kg/s]
                                'oxygen'                 : {},  # [kg/s]
                                'process steam'          : {},  # [kg/s]
                                'gas'                    : {},  # [Sm^3/s]
                                'water'                  : {},  # [m^3/s]
                                'nitrogen'               : {},  # [kg/s]
                                'ammonia'                : {}}  # [kg/s]
        
        self.production = {                                  # initialise production dictionaries
                                'electricity'            : {},  # [kW]
                                'heating water'          : {},  # [kW]
                                'cooling water'          : {},  # [kW]
                                'process heat'           : {},  # [kW]
                                'process hot water'      : {},  # [kW]
                                'process cold water'     : {},  # [kW]
                                'process chilled water'  : {},  # [kW]
                                'hydrogen'               : {},  # [kg/s]
                                'LP hydrogen'            : {},  # [kg/s]
                                'HP hydrogen'            : {},  # [kg/s]
                                'oxygen'                 : {},  # [kg/s]
                                'process steam'          : {},  # [kg/s]
                                'gas'                    : {},  # [Sm^3/s]
                                'water'                  : {},  # [m^3/s]
                                'nitrogen'               : {},  # [kg/s]
                                'ammonia'                : {}}  # [kg/s]
        
        for carrier in self.consumption:
                self.consumption[carrier] = {tech: {tech: np.zeros(c.timestep_number) for tech in self.system} for tech in self.system}
                for tech in self.consumption[carrier]:
                    self.consumption[carrier][tech]['Aux'] = np.zeros(c.timestep_number)
                
        for carrier in self.production:
                self.production[carrier] = {tech: {tech: np.zeros(c.timestep_number) for tech in self.system} for tech in self.system}
                for tech in self.production[carrier]:
                    self.production[carrier][tech]['Aux'] = np.zeros(c.timestep_number)
        # create the objects of present technologies and add them to the technologies dictionary
        # initialise power balance and add them to the power_balance dictionary
        
        for carrier in self.power_balance: # creating grid series and importing energy carrier demand series if defined
            
            if f"{carrier} grid" in self.system:    # grid connection 
                self.power_balance[carrier][carrier+' grid'] = np.zeros(c.timestep_number) # creating the energy carrier array for grid exchanges. Bought from the grid (-) or feed into the grid (+)
            if f"{carrier} demand" in self.system:  # read and check energy/material stream demand series
                if carrier in ['hydrogen','HP hydrogen']:   # hydrogen energy carrier specific requirements. Depending oon the selected simulation strategy. 
                    self.hydrogen_demand = carrier                      # demand can be defined as 'hydrogen demand' or 'HP hydrogen demand' depending on the required delivery pressure
                    if self.system[carrier+' demand']['strategy'] == 'supply-led' and self.system[carrier+' demand']['series'] != False:
                        raise ValueError(f"Warning in {self.name} location: supply-led strategy is not consistent with providing a demand series.\n\
                        Options to fix the problem: \n\
                            (a) - Insert 'false' at {carrier} demand 'series' in studycase.json\n\
                            (b) - Change 'strategy' to 'demand-led' in studycase.json")
                    elif self.system[carrier+' demand']['strategy'] == 'supply-led':            # if selected strategy is supply-led and a demand series is not provided (as it should be the case) 
                        self.power_balance[carrier][carrier+' demand'] =  np.zeros(c.timestep_number)    # no demand is considered in the simulation - the system is investigated in order to assess how much hydrogen it can produce    
                    elif self.system[carrier+' demand']['strategy'] == 'demand-led':
                        self.power_balance[carrier][carrier+' demand'] = - pd.read_csv(path+'/loads/'+system[f"{carrier} demand"]['series'])['kg/s'].to_numpy() 
                
                # checking input files, different units for different energy carriers
                elif carrier == 'process steam' or carrier == 'ammonia':    # [kg/s]
                    self.power_balance[carrier][carrier+' demand']   = - pd.read_csv(path+'/loads/'+system[f"{carrier} demand"]['series'])['kg/s'].to_numpy() 
                elif carrier == 'gas':              # [Sm3/s]
                    self.power_balance[carrier][carrier+' demand']   = - pd.read_csv(path+'/loads/'+system[f"{carrier} demand"]['series'])['Sm3/s'].to_numpy() 
                else:                               # [kW]
                    self.power_balance[carrier][carrier+' demand']   = - pd.read_csv(path+'/loads/'+system[f"{carrier} demand"]['series'])['kW'].to_numpy() 
                     
                ### check demand series length
                if len(self.power_balance[carrier][carrier+' demand']) == c.timestep_number:             # if demand series has the length of the entire simulation (for all years considered)
                    pass 
                elif len(self.power_balance[carrier][carrier+' demand']) < c.timestep_number:            # if the length of the demand array is less than the total number of timesteps in the simulation
                    if c.timestep_number % len(self.power_balance[carrier][carrier+' demand']) == 0:     # if the number of timesteps is evenly divisible by the length of the current demand array
                        self.power_balance[carrier][carrier+' demand'] = np.tile(self.power_balance[carrier][carrier+' demand'],int(c.timestep_number/len(self.power_balance[carrier][carrier+' demand']))) # replicate the demand array for the considered number of years to cover all timesteps in the simulation 
                else:
                    raise ValueError(f"Warning! Check the length of the {carrier} input demand series in {self.name}. Allign it with selected timestep and simulation length in general.json")

        if 'chp_gt' in self.system:
            self.technologies['chp_gt'] = chp_gt(system['chp_gt'],c.timestep_number) # chp_gt object created and added to 'technologies' dictionary
            self.power_balance['process steam']['chp_gt']   = np.zeros(c.timestep_number) # array chp_gt process steam balance 
            self.power_balance['electricity']['chp_gt']     = np.zeros(c.timestep_number) # array chp_gt electricity balance
            self.power_balance['hydrogen']['chp_gt']        = np.zeros(c.timestep_number) # array chp_gt process hydrogen balance 
        
        if 'chp' in self.system:
            self.technologies['chp'] = Chp(system['chp'],c.timestep_number) # chp object created and added to 'technologies' dictionary
            self.power_balance[self.technologies['chp'].th_out]['chp']  = np.zeros(c.timestep_number) # array chp thermal output balance (process steam/hot water)
            self.power_balance['electricity']['chp']                    = np.zeros(c.timestep_number) # array chp electricity balance
            self.power_balance[self.technologies['chp'].fuel]['chp']    = np.zeros(c.timestep_number) # array chp fuel consumption balance
            self.power_balance['process heat']['chp']                   = np.zeros(c.timestep_number) # array chp process heat balance
            self.power_balance['process hot water']['chp']              = np.zeros(c.timestep_number) # array chp process hot water balance
            self.power_balance['process cold water']['chp']             = np.zeros(c.timestep_number) # array chp process cold water balance
       
        if 'absorber' in self.system:
            self.technologies['absorber'] = Absorber(system['absorber'],c.timestep_number) # absorber object created and added to 'technologies' dictionary
            self.power_balance['process heat']['absorber']          = np.zeros(c.timestep_number) # array absorber process steam balance 
            self.power_balance['process hot water']['absorber']     = np.zeros(c.timestep_number) # array absorber process steam balance 
            self.power_balance['process cold water']['absorber']    = np.zeros(c.timestep_number) # array absorber process steam balance 
            
        if 'heatpump' in self.system:
            self.technologies['heatpump'] = heatpump(system['heatpump']) # heatpump object created and add to 'technologies' dictionary
            self.power_balance['electricity']['heatpump']           = np.zeros(c.timestep_number) # array heatpump electricity balance
            self.power_balance['heating water']['heatpump']         = np.zeros(c.timestep_number) # array heatpump heat balance
            self.power_balance['heating water']['inertial TES']     = np.zeros(c.timestep_number) # array inertial tank heat balance
                
        if 'boiler_el' in self.system:
            self.technologies['boiler_el'] = boiler_el(self.system['boiler_el'])    # boiler_el object created and add to 'technologies' dictionary
            self.power_balance['electricity']['boiler_el']      = np.zeros(c.timestep_number)   # array boiler_el electricity balance
            self.power_balance['heating water']['boiler_el']    = np.zeros(c.timestep_number)   # array boiler_el heat balance
               
        if 'boiler_ng' in self.system:
            self.technologies['boiler_ng'] = boiler_ng(self.system['boiler_ng'])    # boiler_ng object created and add to 'technologies' dictionary
            self.power_balance['gas']['boiler_ng']              = np.zeros(c.timestep_number)   # array boiler_ng gas balance
            self.power_balance['heating water']['boiler_ng']    = np.zeros(c.timestep_number)   # array boiler_ng heat balance 
            
        if 'boiler_h2' in self.system:
            self.technologies['boiler_h2'] = boiler_h2(self.system['boiler_h2'])    # boiler_h2 object created and added to 'technologies' dictionary
            self.power_balance['hydrogen']['boiler_h2']         = np.zeros(c.timestep_number)   # array boiler_h2 gas balance
            self.power_balance['heating water']['boiler_h2']    = np.zeros(c.timestep_number)   # array boiler_h2 heat balance 
        
        if 'PV' in self.system:
            self.technologies['PV'] = PV(self.system['PV'],self.name,path,check,file_structure,file_general) # PV object created and add to 'technologies' dictionary
            self.power_balance['electricity']['PV'] = np.zeros(c.timestep_number) # array PV electricity balance
           
        if 'inverter' in self.system:
            self.technologies['inverter'] = inverter(self.system['inverter'],c.timestep_number) # inverter object created and add to 'technologies' dictionary
            self.power_balance['electricity']['inverter'] = np.zeros(c.timestep_number)        # array inverter electricity balance
            
        if 'wind' in self.system:
            self.technologies['wind'] = wind(self.system['wind'],self.name,path,check,file_structure,file_general)    # wind object created and add to 'technologies' dictionary
            self.power_balance['electricity']['wind'] = np.zeros(c.timestep_number)        # array wind electricity balance 
           
        if 'battery' in self.system:
            self.technologies['battery'] = battery(self.system['battery'])    # battery object created and to 'technologies' dictionary
            self.power_balance['electricity']['battery'] = np.zeros(c.timestep_number)         # array battery electricity balance
                           
        if 'electrolyzer' in self.system:
            self.technologies['electrolyzer'] = electrolyzer(self.system['electrolyzer'],c.timestep_number) # electrolyzer object created and to 'technologies' dictionary
            self.power_balance['electricity']['electrolyzer']              = np.zeros(c.timestep_number) # array electrolyzer electricity balance
            self.power_balance['oxygen']['electrolyzer']                   = np.zeros(c.timestep_number) # array electrolyzer oxygen balance
            self.power_balance['water']['electrolyzer']                    = np.zeros(c.timestep_number) # array electrolyzer water balance
            self.power_balance['hydrogen']['electrolyzer']                 = np.zeros(c.timestep_number) # array electrolyzer hydrogen balance
            if self.technologies['electrolyzer'].strategy == "full-time" and (not self.system["electricity grid"]["draw"] or self.system['electrolyzer']['only_renewables']):
                raise ValueError(f"Full-time electrolyzers operation considered without electricity grid connection in {self.name} location.\n\
                Options to fix the problem: \n\
                    (a) - Insert electricity grid withdrawal in studycase.json\n\
                    (b) - Change electrolyzers strategy in studycase.json\n\
                    (c) - Check electrolyzers for 'only_renewables' value in studycase.json to be 'false'")
            if self.technologies['electrolyzer'].strategy == "full-time" and self.system[self.hydrogen_demand+' demand']['strategy'] == 'supply-led':  
                if 'mechanical compressor' in self.system and 'tank' in self.system: # when electrolyzers working continuously in supply-mode there is no need for storage
                    raise ValueError(f"Full-time electrolyzers operation considered in supply-led mode in {self.name} location. Compression and storage must not be considered\n\
                    Options to fix the problem: \n\
                        (a) - Remove 'mechanical compressor' and 'H tank' from studycase.json")
            if self.technologies['electrolyzer'].strategy == "full-time" and self.system[self.hydrogen_demand+' demand']['strategy'] == 'demand-led':
                raise ValueError(f"Warning in {self.name} location: full-time electrolyzers operation not consistent in demand-led mode.\n\
                Feasible combinations: \n\
                    (a) - hydrogen demand:'supply-led' & electrolyzer strategy:'full-time' \n\
                    (b) - hydrogen demand:'demand-led' & electrolyzer strategy:'hydrogen-first'")
            if self.system['electrolyzer']['only_renewables'] == False and self.system["electricity grid"]["draw"] == False:
                raise ValueError(f"If 'only_renewables' strategy for electrolyzers operation is set 'false', grid connection in {self.name} location must be allowed.\n\
                Options to fix the problem: \n\
                    (a) - Insert electricity grid withdrawal in studycase.json\n\
                    (b) - Change electrolyzers 'only_renewables' strategy in studycase.json")
                    
        if 'H tank' and 'HPH tank' in self.system: 
            # H_tank
            self.technologies['H tank'] = H_tank(self.system['H tank'], c.timestep_number) # H tank object created and to 'technologies' dictionary
            self.power_balance['hydrogen']['H tank']        = np.zeros(c.timestep_number)   # array H tank hydrogen balance
            # HPH tank
            self.technologies['HPH tank'] = HPH_tank(self.system['HPH tank'],c.timestep_number) # HPH tank object created and to 'technologies' dictionary
            self.power_balance['HP hydrogen']['HPH tank'] = np.zeros(c.timestep_number)     # array HPH tank hydrogen balance
            self.tank_stream = {'H tank'    :'hydrogen',        # dictionary assigning different hydrogen streams to different storage technologies - necessary for loc_energy_simulation
                                'HPH tank'  :'HP hydrogen'}   
 
        if 'H tank' in self.system and not 'HPH tank' in self.system:
            if 'hydrogen demand' in self.system or 'HP hydrogen demand' in self.system:
                if self.system[self.hydrogen_demand+' demand']['strategy'] == 'supply-led' and self.system['H tank']['max capacity'] != False:
                    raise ValueError(f"Adjust {self.name} location system in studycase.json. When the system is operated in supply-led mode, H tank size cannot be defined in advance.\n\
            Options to fix the problem: \n\
            (a) - Insert false for 'max capacity' among H tank parameters in studycase.json\n\
            (b) - Switch to 'demand-led' in 'hydrogen-demand'('strategy')")
            self.technologies['H tank'] = H_tank(self.system['H tank'], c.timestep_number)   # H tank object created and to 'technologies' dictionary
            self.power_balance['hydrogen']['H tank'] = np.zeros(c.timestep_number)         # array H tank hydrogen balance
            self.tank_stream = {'H tank':'hydrogen'}     # dictionary assigning hydrogen stream to H tank storage technologies - necessary for loc_energy_simulation
            if self.technologies['electrolyzer'].strategy == "ammonia production" and self.technologies['H tank'].Operating_T_p[1] < self.system['ASR']['Preact']:
                raise ValueError ('Warning: the pressure of the hydrogen buffer cannot be less than the operating pressure of the ASR')
            if self.technologies['electrolyzer'].strategy == "ammonia production" and self.system['H tank']['priority'] < self.system['ASR']['priority']:
                raise ValueError('Warning: for the strategy ammonia production of the electrolyzer the priority of the H tank needs to be after the one of the ASR')
            
        if 'ASR' in self.system:
            PSA_operating_T_p = [self.system['PSA']['T_psa'], self.system['PSA']['P_psa']] 
            electrolyzer_operating_T_p = [self.technologies['electrolyzer'].OperatingTemp, self.technologies['electrolyzer'].OperatingPress/100000]
            buffer_T_p = self.technologies['H tank'].Operating_T_p
            self.technologies['ASR'] = ASR(self.system['ASR'], c.timestep_number, electrolyzer_operating_T_p, PSA_operating_T_p, self.name, path, file_structure, file_general, buffer_info = buffer_T_p)  # ASR object created and added to 'technologies' dictionary
            self.power_balance['ammonia']['ASR'] = np.zeros(c.timestep_number) # array ASR nitrogen balance
            self.power_balance['electricity']['ASR'] = np.zeros(c.timestep_number) # array ASR electricity balance
            self.power_balance['nitrogen']['ASR'] = np.zeros(c.timestep_number) # array ASR nitrogen balance
            self.power_balance['hydrogen']['ASR'] = np.zeros(c.timestep_number) # array ASR electricity balance
            if self.technologies['electrolyzer'].strategy != "ammonia production":
                raise ValueError('Warning: if the ASR is present, the electrolyzer needs to be in strategy ammonia production.')
            if self.technologies['ASR'].strategy != 'full-time':
                raise ValueError("Warning: for the ASR the 'full-time' strategy is only possible.")
            if 'H tank' not in self.system:
                raise ValueError("Warning: since for the ASR the 'full-time' strategy is only possible, there is a need of a hydrogen buffer.\n\
                Options to fix the problem: \n\
                    (a) - Insert a hydrogen tank.")
            if self.system['ASR']['only_renewables'] == True:
                raise ValueError("Warning: the 'full-time' strategy of the ASR is not possible with 'only_renewables' strategy 'true'.\n\
                Options to fix the problem: \n\
                    (a) - Change ASR 'only_renewables' strategy in studycase.json\n\
                    (b) - If no source of electricity are included except to renewable energy insert a battery and/or a CCGT or electricity from the grid.")
            if self.system['ASR']['priority'] < self.system['electrolyzer']['priority']:
                raise ValueError('Warning: the priority of the ASR needs to be after the one of the electrolyzer.')
            if self.system['ASR']['priority'] > self.system['PSA']['priority']:
                raise ValueError('Warning: the priority of the ASR needs to be before the one of the PSA.')
                    
        if 'PSA' in self.system:
            if "ASR" not in self.system:
                raise ValueError(f"ASR not present in the {self.name} location. The model as it is considers nitrogen production\n\
                                    only when ammonia is produced in situ. PSA is directly connected either to \n\
                                    the ASR, check priorities in studycase.json.\n\
                                    Options to fix the problem: \n\
                    (a) - Insert ASR technology in studycase.json")
            self.technologies['PSA'] = PSA(self.system['PSA'],c.timestep_number)  # PSA object created and added to 'technologies' dictionary
            self.power_balance['nitrogen']['PSA'] = np.zeros(c.timestep_number) # array PSA nitrogen balance
            self.power_balance['electricity']['PSA'] = np.zeros(c.timestep_number) # array PSA electricity balance
            if self.system['PSA']['priority'] < self.system['electrolyzer']['priority']:
                raise ValueError('Warning: for the strategy ammonia production of the electrolyzer the priority of the PSA needs to be after the one of the electrolyzer.')
            if self.system['PSA']['only_renewables'] != self.system['ASR']['only_renewables']:
                raise ValueError("Warning: 'only_renewables' strategy needs to be the same for ASR and PSA as the PSA works just with the ASR.")    
        
        if 'SMR' in self.system:
            self.technologies['SMR'] = SMR(self.system['SMR'],c.timestep)             # Steam methane reformer object created and to 'technologies' dictionary
            self.power_balance['hydrogen']['SMR']   = np.zeros(c.timestep_number)     # array Steam methane reformer hydrogen balance
            self.power_balance['gas']['SMR']        = np.zeros(c.timestep_number)     # array Steam methane reformer gas balance
            
        if 'mhhc compressor' in self.system:
            self.technologies['mhhc compressor'] = mhhc_compressor(self.system['mhhc compressor'],c.timestep_number) # MHHC compressor object created and to 'technologies' dictionary
            self.power_balance['hydrogen']['mhhc compressor']  = np.zeros(c.timestep_number)     # array hydrogen compressor hydrogen compressed
            self.power_balance['gas']['mhhc compressor']       = np.zeros(c.timestep_number)     # array hydrogen compressor heating water balanced use
            
        if 'NH3 tank' in self.system:
            self.technologies['NH3 tank'] = NH3_tank(self.system['NH3 tank'],c.timestep_number) # NH3 tank object created and to 'technologies' dictionary
            self.power_balance['ammonia']['NH3 tank'] = np.zeros(c.timestep_number)         # array NH3 tank ammonia balance                      
        
        if 'O2 tank' in self.system: 
            self.technologies['O2 tank'] = O2_tank(self.system['O2 tank'],c.timestep_number) # LPH tank object created and to 'technologies' dictionary
            self.power_balance['oxygen']['O2 tank'] = np.zeros(c.timestep_number)         # array LPH tank hydrogen balance
            
        if 'mechanical compressor' in self.system:
            if "electrolyzer" not in self.system:
                raise ValueError(f"Electrolyzer not present in the {self.name} location. The model as it is considers hydrogen compression\n\
                                    only when hydrogen is produced in situ. Compressor is directly connected either to \n\
                                    electrolyzer or a buffer tank, check priorities in studycase.json.\n\
                                    Options to fix the problem: \n\
                    (a) - Insert electrolyzer technology in studycase.json")
            if self.technologies['electrolyzer'].strategy == "ammonia production":
                maxflowrate_ele = self.technologies['electrolyzer'].maxh2prod_stack - self.technologies['ASR'].min_hydrogen
            else:
                maxflowrate_ele = self.technologies['electrolyzer'].maxh2prod_stack            
            self.technologies['mechanical compressor'] = Compressor(self.system['mechanical compressor'],c.timestep_number,maxflowrate_ele=maxflowrate_ele) # compressor object created and to 'technologies' dictionary
            self.power_balance['electricity']['mechanical compressor']    = np.zeros(c.timestep_number) # array compressor hydrogen balance
            self.power_balance['hydrogen']['mechanical compressor']       = np.zeros(c.timestep_number) # array of hydrogen flow entering the mechanical compressor from LPH tank
            self.power_balance['HP hydrogen']['mechanical compressor']    = np.zeros(c.timestep_number) # array of compressed hydrogen flow sent toward HPH tank
            self.power_balance['cooling water']['mechanical compressor']  = np.zeros(c.timestep_number) # array of water flow to be fed to the refrigeration system 
                
        if 'CCGT' in self.system:
            self.technologies['CCGT'] = CCGT(self.system['CCGT'], path, c.timestep_number) # CCGT object created and to 'technologies' dictionary
            self.power_balance['electricity']['CCGT'] = np.zeros(c.timestep_number) # array CCGT electricity balance
            self.power_balance[self.technologies['CCGT'].fuel]['CCGT'] = np.zeros(c.timestep_number) # array CCGT fuel consumption balance
            
        if 'fuel cell' in self.system:
            n_modules = self.system['fuel cell']['number of modules']
            if isinstance(n_modules, (list, str)) and 'automatic dimensioning' in n_modules:
                FC_power = 0
                if 'electricity demand' in self.system['fuel cell']['number of modules']:
                    FC_power += np.max(abs(self.power_balance['electricity']['electricity demand']))
                if 'ASR' in self.system['fuel cell']['number of modules']:
                    FC_power += self.technologies['ASR'].electricity_design
                if 'PSA' in self.system['fuel cell']['number of modules']:
                    FC_power += self.technologies['PSA'].Npower
                if 'cracker' in self.system['fuel cell']['number of modules']:
                    cracker_n_modules_temp = self.system['cracker']['number of modules']
                    self.system['cracker']['number of modules'] = 1
                    cracker_module = cracker(self.system['cracker'], self.technologies['NH3 tank'].temperature, self.technologies['NH3 tank'].pressure, self.name, path, file_structure, file_general,)  # initialization just to compute the specific consumption, then is initialized again with the right size
                    cracker_SEC = cracker_module.electricity_design / (cracker_module.H2_massflowrate * 3600)
                    self.system['cracker']['number of modules'] = cracker_n_modules_temp
                    self.system['fuel cell']['number of modules'] = 1
                    self.technologies['fuel cell'] = fuel_cell(self.system['fuel cell'],c.timestep_number)   # Fuel cell object created and to 'technologies' dictionary
                    if cracker_SEC > 0:
                        FC_power = FC_power / (1 - (cracker_SEC / c.HHV_H2 / self.technologies['fuel cell'].eta_module[-1]))
                self.system['fuel cell']['number of modules'] = math.ceil(FC_power / self.system['fuel cell']['Npower'])
            self.technologies['fuel cell'] = fuel_cell(self.system['fuel cell'],c.timestep_number) # Fuel cell object created and to 'technologies' dictionary
            self.power_balance['electricity']['fuel cell']     = np.zeros(c.timestep_number)     # array fuel cell electricity balance
            self.power_balance['hydrogen']['fuel cell']        = np.zeros(c.timestep_number)     # array fuel cell hydrogen balance
            self.power_balance['heating water']['fuel cell']   = np.zeros(c.timestep_number)     # array fuel cell heat balance used
        
        if 'cracker' in self.system:
            H2_FC = False
            Npower_CCGT = False
            eff_CCGT = False
            n_modules = self.system['cracker']['number of modules']
            if isinstance(n_modules, (list, str)) and 'automatic dimensioning' in n_modules:
                if 'fuel cell' in self.system['cracker']['number of modules']:
                    H2_FC = self.technologies['fuel cell'].max_h2_stack
                    if self.system['cracker']['priority'] > self.system['fuel cell']['priority']:
                        raise ValueError('Cracker priority has to be before than fuel cell priority')
                if 'CCGT' in self.system['cracker']['number of modules']:
                    Npower_CCGT = self.technologies['CCGT'].Npower
                    eff_CCGT = self.technologies['CCGT'].efficiency
                    if self.system['cracker']['priority'] > self.system['CCGT']['priority']:
                        raise ValueError('Cracker priority has to be before than CCGT priority')
            self.technologies['cracker'] = cracker(self.system['cracker'], self.technologies['NH3 tank'].temperature, self.technologies['NH3 tank'].pressure, self.name, path, file_structure, file_general, H2_FC=H2_FC, Npower_CCGT=Npower_CCGT, eff_CCGT=eff_CCGT)
            self.power_balance['ammonia']['cracker'] = np.zeros(c.timestep_number)
            self.power_balance['electricity']['cracker'] = np.zeros(c.timestep_number)
            self.power_balance['hydrogen']['cracker'] = np.zeros(c.timestep_number)
        
        self.power_balance['electricity']['collective self consumption']   = np.zeros(c.timestep_number) # array contribution to collective-self-consumption as producer (-) or as consumer (+)
        #self.power_balance['heating water']['collective self consumption'] = np.zeros(c.timestep_number) # array contribution to collective-self-consumption as producer (-) or as consumer (+)---heat----mio!!!
        #self.power_balance['process steam']['collective self consumption'] = np.zeros(c.timestep_number) # array contribution to collective-self-consumption as producer (-) or as consumer (+)---heat----mio!!!
   
    ### Function to address where the energy produced is used, and vice versa
            
    def consumption_logic(self,carrier,tech_name,step):
        
        if self.power_balance[carrier][tech_name][step] < 0:   # energy consumption
            self.consumption[carrier][tech_name][tech_name][step] = - self.power_balance[carrier][tech_name][step]
            self.consumption[carrier][tech_name]['Aux'][step] = - self.power_balance[carrier][tech_name][step]       # save an auxiliar variable to be updated 
            
            required_energy = - self.power_balance[carrier][tech_name][step] 
            for tech in self.production[carrier]:                                      
                if self.production[carrier][tech]['Aux'][step] > 0:                     # if some tech with higher priority has available energy
                    self.production[carrier][tech][tech_name][step] = min(required_energy,self.production[carrier][tech]['Aux'][step])     # calculate how much energy this tech can take from the higher priority one
                    required_energy -= self.production[carrier][tech][tech_name][step]                                                     # see if there is still energy to take
                    self.consumption[carrier][tech_name]['Aux'][step] -= self.production[carrier][tech][tech_name][step]                   # update auxiliar variable
                    self.consumption[carrier][tech_name][tech][step] = self.production[carrier][tech][tech_name][step]                     # update consumption from the higher priority tech
                    self.production[carrier][tech]['Aux'][step] -= self.production[carrier][tech][tech_name][step]                         # update the auxiliar variable of higher priority tech (less energy to give) 
        else:
            pass
        
    def production_logic(self,carrier,tech_name,step):
        
        if self.power_balance[carrier][tech_name][step] > 0:   # energy production
            self.production[carrier][tech_name][tech_name][step] = self.power_balance[carrier][tech_name][step]
            self.production[carrier][tech_name]['Aux'][step] = self.power_balance[carrier][tech_name][step]         # save an auxiliar variable to be updated
            available_energy = self.power_balance[carrier][tech_name][step]
            
            
            for tech in self.consumption[carrier]:
                if self.consumption[carrier][tech]['Aux'][step] > 0:                   # if some tech with higher priority has required energy     
                    self.consumption[carrier][tech][tech_name][step] = min(available_energy,self.consumption[carrier][tech]['Aux'][step])  # calculate how much energy this tech can give to the higher priority one
                    available_energy -= self.consumption[carrier][tech][tech_name][step]                                                   # see if there is still energy to give    
                    self.consumption[carrier][tech]['Aux'][step] -= self.consumption[carrier][tech][tech_name][step]                       # update the auxiliar variable of higher priority tech (less energy to take)
                    self.production[carrier][tech_name]['Aux'][step] -= self.consumption[carrier][tech][tech_name][step]                   # update production to the higher priority tech
                    self.production[carrier][tech_name][tech][step] = self.consumption[carrier][tech][tech_name][step]                     # update auxiliar variable
        
        else:
            pass
     
    def hydrogen_available_producible(self,step,pb):    # Tank storage system handling
        
        ## Available Hydrogen ##
        
        hydrogen_produced = max(0,pb['hydrogen'])*c.timestep*60
        
        if "hydrogen grid" in self.system and self.system["hydrogen grid"]["draw"] and "H tank" not in self.system:  #hydrogen can be drawn from a hydrogen grid
            available_hyd = float('inf')
            tank_availability = 0
        elif "hydrogen grid" in self.system and self.system["hydrogen grid"]["draw"] and "H tank" in self.system:
            tank_availability = self.technologies['H tank'].LOC[step] + self.technologies['H tank'].max_capacity - self.technologies['H tank'].used_capacity - self.technologies['H tank'].max_capacity * self.technologies['H tank'].min_level
            available_hyd = float('inf') 
        elif "H tank" in self.system and ("hydrogen grid" not in self.system or not self.system["hydrogen grid"].get("draw")):
            tank_availability = self.technologies['H tank'].LOC[step] + self.technologies['H tank'].max_capacity - self.technologies['H tank'].used_capacity - self.technologies['H tank'].max_capacity * self.technologies['H tank'].min_level
            available_hyd = tank_availability + hydrogen_produced
        else:
            available_hyd = hydrogen_produced
            tank_availability = 0

        ## Producible Hydrogen ##
        
        if self.technologies['electrolyzer'].strategy == 'ammonia production':
            producible_hyd = self.ammonia_available_producible(step,pb)[1] * 3 * c.H2MOLMASS / (2 * c.NH3MOLMASS)   # each molecule of ammonia (NH3) contains 3 atoms of hydrogen, dividing by 2 reflects that two hydrogen atoms combine to form one H2 molecule
            if 'H tank' in self.system:
                producible_hyd +=  self.technologies['H tank'].max_capacity-self.technologies['H tank'].LOC[step]
        else:
            if "hydrogen grid" in self.system and self.system["hydrogen grid"]["feed"]: # hydrogen can be fed into an hydrogen grid
                producible_hyd = float('inf')
            elif 'hydrogen demand' not in self.system and 'H tank' in self.system: # hydrogen-energy-storage configuration, only renewable energy is stored in the form of hydrogen to be converted back into electricity via fuel cell 
                producible_hyd  = self.technologies['H tank'].max_capacity-self.technologies['H tank'].LOC[step] # the tank can't be full
                if producible_hyd < self.technologies['H tank'].max_capacity*0.00001: # to avoid unnecessary iteration
                    producible_hyd = 0
            elif 'H tank' in self.system and 'HPH tank' not in self.system and self.system[self.hydrogen_demand+' demand']['strategy'] == 'demand-led':   # hydrogen can only be stored into an H tank 
                producible_hyd  = self.technologies['H tank'].max_capacity-self.technologies['H tank'].LOC[step] + (-pb['hydrogen'])*c.timestep*60 # the tank can't be full
                if producible_hyd < self.technologies['H tank'].max_capacity*0.00001:                           # to avoid unnecessary iteration
                    producible_hyd = 0
            elif 'H tank' in self.system and 'HPH tank' not in self.system and self.system[self.hydrogen_demand+' demand']['strategy'] == 'supply-led':   # hydrogen can only be stored into an H tank 
                producible_hyd = float('inf')   # electrolyzer can produce continuously as the storage capacity is infinite. Tank is dimensioned at the end of simulation
            elif 'H tank' in self.system and 'HPH tank' in self.system:
                producible_hyd   = self.technologies['H tank'].max_capacity-self.technologies['H tank'].LOC[step] # the tank can't be full                        
            else:
                producible_hyd = max(0,-pb['hydrogen']*c.timestep*60) # hydrogen is consumed by a technology with a higher priority than tank
            
        return(available_hyd, tank_availability, producible_hyd)
    
    def ammonia_available_producible(self,step,pb):
        
        ## Available Ammonia ##
        
        produced_nh3 = max(0,pb['ammonia'])*c.timestep*60
        
        if "ammonia grid" in self.system and self.system["ammonia grid"]["draw"]:  # ammonia can be drawn from an ammonia grid
            available_nh3 = float('inf')
        elif "NH3 tank" in self.system:
            tank_availability = self.technologies['NH3 tank'].LOC[step] + self.technologies['NH3 tank'].max_capacity - self.technologies['NH3 tank'].used_capacity
            available_nh3 = tank_availability + produced_nh3
        else:
            available_nh3 = produced_nh3
        
        ## Producible Ammonia ##
        
        if "ammonia grid" in self.system and self.system["ammonia grid"]["feed"]:
            producible_nh3 = self.technologies['ASR'].max_ammonia * (c.timestep*60)  # maximum hydrogen production imposed only by the reactor capacity of producing ammonia
        if ("ammonia grid" in self.system and not self.system["ammonia grid"]["feed"]) or ("ammonia grid" not in self.system):
            if 'NH3 tank' in self.system and 'ammonia demand' not in self.system: # just a tank and no possibility to send ammonia inside the grid or directly to a user - just storage
                producible_nh3 = min(self.technologies['ASR'].max_ammonia * (c.timestep*60), self.technologies['NH3 tank'].max_capacity-self.technologies['NH3 tank'].LOC[step]) # the ammonia tank can't be full                     
            if 'ammonia demand' in self.system and 'NH3 tank' not in self.system: # just an ammonia demand and no possibility of storage or to send the excess ammonia to the grid
                producible_nh3 = min(self.technologies['ASR'].max_ammonia * (c.timestep*60), abs(self.power_balance['ammonia']['ammonia demand'][step]*(c.timestep*60)))     
            if 'NH3 tank' in self.system and 'ammonia demand' in self.system: # ammonia demand but also a tank to store the excess ammonia produced
                producible_nh3 = min(self.technologies['ASR'].max_ammonia * (c.timestep*60), self.technologies['NH3 tank'].max_capacity-self.technologies['NH3 tank'].LOC[step]+abs(self.power_balance['ammonia']['ammonia demand'][step]*(c.timestep*60)))
        if producible_nh3 < self.technologies['ASR'].min_ammonia * (c.timestep*60):   # the ASR can't work under the minimum load
            producible_nh3 = 0
        # Check on limitations for buffer level
        if self.technologies['ASR'].buffer_control:
            if self.technologies['H tank'].LOC[step]/self.technologies['H tank'].max_capacity < self.technologies['ASR'].buffer_control:
               producible_nh3 = self.technologies['ASR'].min_ammonia * (c.timestep*60)
        # Check on operational period
        if not self.technologies['ASR'].initial_hour <= step <= self.technologies['ASR'].final_hour:
            if self.technologies['ASR'].state == 'off':
                producible_nh3 = 0
            if self.technologies['ASR'].state == 'min load':
                producible_nh3 = self.technologies['ASR'].min_ammonia * (c.timestep*60)
        if step != 0: 
            # Check on ramp-rate constrains
            load = producible_nh3 / (c.timestep*60) / self.technologies['ASR'].max_ammonia
            if abs(load - self.technologies['ASR'].load[step-1]) > self.technologies['ASR'].max_ramp_rate:
                if load > self.technologies['ASR'].load[step-1]:
                    load = self.technologies['ASR'].load[step-1] + self.technologies['ASR'].max_ramp_rate
                else:
                    load = self.technologies['ASR'].load[step-1] - self.technologies['ASR'].max_ramp_rate
                producible_nh3 = load * self.technologies['ASR'].max_ammonia * (c.timestep*60)
                
        if self.technologies['ASR'].strategy == "full-time":
            if self.technologies['ASR'].initial_hour <= step <= self.technologies['ASR'].final_hour and producible_nh3 == 0:  # if it's operational period
               raise ValueError ('Warning: the selected strategy for the ASR is full-time but the NH3 tank / the demand is too small to guarantee it') 
            elif not self.technologies['ASR'].initial_hour <= step <= self.technologies['ASR'].final_hour and self.technologies['ASR'].state != 'off' and producible_nh3 == 0:  # if not operational period but the ASR has to in any case work at minimum load
                raise ValueError ('Warning: the selected strategy for the ASR is full-time but the NH3 tank / the demand is too small to guarantee it')
                
        return (available_nh3, producible_nh3)
    
    def power_allocation(self, step, producible_hyd, H2tank_availability, producible_ammonia, el_input, weather):
        # Computation of the energy quota to be given to the electrolyzer in order for the other components to work
        new_producible_ammonia = None
        ASR_hydrogen_from_buffer = None
        delta = None
        tolerance = 1e-5
        maxiter = 100
        low, high = 0, 1  # bounds for the quota
        # If there is the minimum hydrogen available, priority is given to the operation of the ASR
        def energy_quota(quota):
            nonlocal new_producible_ammonia, ASR_hydrogen_from_buffer, delta
            electrolyzer = self.technologies['electrolyzer'].use(step, storable_hydrogen=producible_hyd, p=el_input*quota, Text=weather['temp_air'][step])
            hyd_el = electrolyzer[0]
            en_el = electrolyzer[1]
            if hyd_el < self.technologies['ASR'].min_hydrogen and H2tank_availability/(c.timestep*60) > (self.technologies['ASR'].min_hydrogen-hyd_el) and producible_ammonia >= self.technologies['ASR'].min_ammonia: 
            # If the electrolyzer doesn't produce enough H2, but there is enough H2 inside the tank (H2tank_availability) to compensate and activate the ASR at minimum load
                hyd = self.technologies['ASR'].min_hydrogen  # the ASR works at minimum load
                hyd_buffer = self.technologies['ASR'].min_hydrogen-hyd_el  # the difference to reach the minimum load is taken from the H tank
            else:
            # If the hydrogen from the electrolyzer is enough: hydrogen intake limited if the hydrogen derived from producible ammonia (stoichiometry) is less than the produced hydrogen (producible hydrogen calculated also for the buffer)
            # If there is no hydrogen available inside the tank to reach the minimum load: if the hydrogen furnished is under the minimum load the ASR functions automtically returns 0 
            # If the producible ammonia is less than the minimum producible: no hydrogen furnished to the ASR as hyd is calculated as the minimum between hydrogen from electrolyzer and hydrogen derived from producible ammonia (stoichiometry)
                hyd = min(hyd_el, producible_ammonia * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS))
                hyd_buffer = 0
            # The hydrogen consumed can vary as compared to the quantity just calculated, if ramp-rates limitations are imposed inside ASR.use
            # If more hydrogen is needed it is taken from the buffer if is available, if not the ASR turns off
            asr = self.technologies['ASR'].use(step, hyd, self.technologies['H tank'].LOC[step]/self.technologies['H tank'].max_capacity, hyd_buffer, self.technologies['H tank'].LOP[step]) 
            new_producible_ammonia = asr[0]
            en_asr = asr[1]
            nitro_asr = abs(asr[2])
            hyd_asr = abs(asr[3])
            ASR_hydrogen_from_buffer = asr[4]  
            if ASR_hydrogen_from_buffer != hyd_buffer and H2tank_availability/(c.timestep*60) < ASR_hydrogen_from_buffer:
                new_producible_ammonia = 0
                en_asr = 0
                nitro_asr = 0
                hyd_asr = 0
                ASR_hydrogen_from_buffer = 0
            psa = self.technologies['PSA'].use(nitro_asr, step) # PSA activated depending on the nitrogen required from the ASR
            en_psa = psa[1]
            if (hyd_el - hyd_asr) > 0:   # if there is hydrogen in excess that needs to be compressed and stored
                compr = self.technologies['mechanical compressor'].use(step, massflowrate = (hyd_el - hyd_asr), p_H2_tank = self.technologies['H tank'].LOP[step])
                en_compr = compr[1]
            else:
                en_compr = 0
            el_tot = abs(en_el) + abs(en_psa) + abs(en_asr) + abs(en_compr)
            delta = el_input - el_tot
            return delta 
        try:
            quota = brentq(energy_quota, low, high, xtol=tolerance, maxiter=maxiter)
            energy_quota(quota)  # to return the right value of delta and producible ammonia
        except ValueError:
            # If f(a) e f(b) are both negative, there is not enough electricity to activate anything at the minimum load
            # Even with quota 0 (electrolyzer off) there is not enough electricity to activate the PSA and the ASR at the minimum load (available hydrogen inside the tank)
            if energy_quota(low) < 0 and energy_quota(high) < 0:  
                quota = 0
                new_producible_ammonia = 0  # even with the electrolyzer off there is not enough electricity to activate the ASR and PSA at the minimum
                ASR_hydrogen_from_buffer = 0
                return quota, new_producible_ammonia, ASR_hydrogen_from_buffer
            elif energy_quota(low) > 0 and energy_quota(high) > 0:   
            # If f(a) e f(b) are both positive, there is enough electricity even giving all the energy to the electrolyzer (quota = 1)
            # It means that or the electrolyzer is off (not enough electricity to activate it), and just the ASR and PSA are on (available hydrogen inside the tank) at minimum load
            # or even the ASR and the PSA are off - > everything is off. Another possible case is if with quota = 1 the electrolyzer is on, it doesnt' take all the energy assigned
            # to it (limitation on the maximum power or producible hydrogen), the ASR and PSA can be on or off, but the group of machines doesn't absorb all the power available
                quota = 1  # all the energy is given to the electrolyzer and even in this way not all the energy is consumed
                energy_quota(quota)
                return quota, new_producible_ammonia, ASR_hydrogen_from_buffer
            else:
                raise  
        if delta < 0:  # the energy consumed needs to be less than the energy available, the convergence with brentq method can reach the wrong value 
                        # based on the tolerance that is verified on the quota and not on the delta
            # Fall back to bisection method
            i = 0
            while i < maxiter:
                quota = (low + high) / 2
                energy_quota(quota) 
                if (high - low) < tolerance and delta >= 0 :
                    break
                if delta > 0:
                    low = quota
                else:
                    high = quota
                i += 1
            if i == maxiter:
                print("Maximum number of iterations reached inside the calculation of the energy quota for the electrolyzer in ammonia production mode (location.py)") 
        return quota, new_producible_ammonia, ASR_hydrogen_from_buffer
            
    
    def loc_power_simulation(self,step,weather):
        """
        Simulate the location
        
        input : int step to simulate
    
        output : updating of location power balances
        """
        pb = {} # power balance [kW] [kg/s] [Sm^3/s]
        
        for carrier in self.power_balance:
           pb[carrier] = 0 # initialise power balances 
            
        for tech_name in self.system: # (which is ordered by priority)
            
            if "ASR" in self.system and self.technologies['electrolyzer'].strategy == 'ammonia production':
                available_ammonia, producible_ammonia = self.ammonia_available_producible(step, pb)
                available_ammonia /= (c.timestep * 60)  # [kg/s]
                producible_ammonia /= (c.timestep * 60)  # [kg/s]
                
            if "electrolyzer" in self.system or "hydrogen grid" in self.system:
                available_hyd, H2tank_availability, producible_hyd = self.hydrogen_available_producible(step,pb) # [kg]
        
            if tech_name == 'PV': 
                self.power_balance['electricity']['PV'][step] = self.technologies['PV'].use(step) # electricity produced from PV
                pb['electricity'] += self.power_balance['electricity']['PV'][step] # elecricity balance update: + electricity produced from PV
                self.production_logic('electricity', 'PV', step)                  
                
            elif tech_name == 'wind': 
                self.power_balance['electricity']['wind'][step] = self.technologies['wind'].use(step) # electricity produced from wind
                pb['electricity'] += self.power_balance['electricity']['wind'][step] # elecricity balance update: + electricity produced from wind
                self.production_logic('electricity', 'wind', step)   
                
            elif tech_name == 'boiler_el': 
                self.power_balance['electricity']['boiler_el'][step],\
                self.power_balance['heating water']['boiler_el'][step] = self.technologies['boiler_el'].use(step,pb['heating water']) # elctricity consumed and heat produced from boiler_el
                pb['electricity']   += self.power_balance['electricity']['boiler_el'][step]     # [kW] elecricity balance update: - electricity consumed by boiler_el
                pb['heating water'] += self.power_balance['heating water']['boiler_el'][step]   # [kW] heat balance update: + heat produced by boiler_el
                self.consumption_logic('electricity', 'boiler_el', step)
                self.production_logic('heating water', 'boiler_el', step)              
                
            elif tech_name == 'boiler_ng': 
                self.power_balance['gas']['boiler_ng'][step],\
                self.power_balance['heating water']['boiler_ng'][step] = self.technologies['boiler_ng'].use(step,pb['heating water']) # ng consumed and heat produced from boiler_ng
                pb['gas']           += self.power_balance['gas']['boiler_ng'][step]             # [Sm3/s] gas balance update: - gas consumed by boiler_ng
                pb['heating water'] += self.power_balance['heating water']['boiler_ng'][step]   # [kW] heat balance update: + heat produced by boiler_ng
                self.consumption_logic('gas', 'boiler_ng', step)
                self.production_logic('heating water', 'boiler_ng', step)
                
            elif tech_name == 'boiler_h2':                                                                                                                                   
                self.power_balance['hydrogen']['boiler_h2'][step],\
                self.power_balance['heating water']['boiler_h2'][step]        = self.technologies['boiler_h2'].use(step,pb['heating water'],available_hyd) # h2 consumed from boiler_h2 and heat produced by boiler_h2
                pb['hydrogen']      += self.power_balance['hydrogen']['boiler_h2'][step]            # [kg/s] hydrogen balance update: - hydrogen consumed by boiler_h2
                pb['heating water'] += self.power_balance['heating water']['boiler_h2'][step]       # [kW] heat balance update: + heat produced by boiler_h2                        
                self.consumption_logic('hydrogen', 'boiler_h2', step)
                self.production_logic('heating water', 'boiler_h2', step)                
                
            elif tech_name == 'heatpump':     
                self.power_balance['electricity']['heatpump'][step], self.power_balance['heating water']['heatpump'][step], self.power_balance['heating water']['inertial TES'][step] = self.technologies['heatpump'].use(weather['temp_air'][step],pb['heating water'],pb['electricity'],step) 
                pb['electricity'] += self.power_balance['electricity']['heatpump'][step] # electricity absorbed by heatpump
                self.consumption_logic('electricity', 'heatpump', step)                                                       
                self.power_balance['electricity']['electricity demand'][step] += self.power_balance['electricity']['heatpump'][step] # add heatpump demand to 'electricity demand'
                pb['heating water'] += self.power_balance['heating water']['inertial TES'][step] + self.power_balance['electricity']['heatpump'][step] # heat or cool supplied by HP or inertial TES
                self.production_logic('heating water', 'heatpump', step)               
                
            elif tech_name == 'battery':
                if self.technologies['battery'].collective == 0: 
                    self.power_balance['electricity']['battery'][step] = self.technologies['battery'].use(step,pb['electricity']) # electricity absorbed(-) or supplied(+) by battery
                    pb['electricity'] += self.power_balance['electricity']['battery'][step]  # electricity balance update: +- electricity absorbed or supplied by battery
                    self.battery_available_electricity = max(0,(self.technologies['battery'].LOC[step+1]+self.technologies['battery'].max_capacity*(1-self.technologies['battery'].DoD)-self.technologies['battery'].used_capacity) / c.P2E)  # electricity available in the battery
                    if self.power_balance['electricity']['battery'][step] >= 0:
                        self.production_logic('electricity', 'battery', step)
                    elif self.power_balance['electricity']['battery'][step] <= 0:
                        self.consumption_logic('electricity', 'battery', step)                   
                        
            elif tech_name == 'chp_gt':
                if available_hyd > 0:
                    use = self.technologies['chp_gt'].use(step,weather['temp_air'][step],pb['process steam'],available_hyd)     # saving chp_gt working parameters for the current timeframe
                    self.power_balance['process steam']['chp_gt'][step] = use[0]   # produced steam (+)
                    self.power_balance['electricity']['chp_gt'][step] =   use[1]   # produced electricity (+)
                    self.power_balance['hydrogen']['chp_gt'][step] =      use[2]   # hydrogen required by chp system to run (-)  
                pb['hydrogen'] += self.power_balance['hydrogen']['chp_gt'][step]            
                pb['process steam'] += self.power_balance['process steam']['chp_gt'][step]    
                pb['electricity'] += self.power_balance['electricity']['chp_gt'][step] 
                self.consumption_logic('hydrogen', 'chp_gt', step)
                self.production_logic('electricity', 'chp_gt', step)
                self.production_logic('process steam', 'chp_gt',step)               
                
            elif tech_name == 'chp':
                strategy    = self.technologies['chp'].strategy     # thermal-load follow or electric-load follow 
                coproduct   = self.technologies['chp'].coproduct    # process co-product depending on the  approache chosen above 
                if self.system['chp']['Fuel'] == 'hydrogen':
                    use = self.technologies['chp'].use(step,weather['temp_air'][step],pb[strategy],pb[coproduct], available_hyd)     # saving chp working parameters for the current timeframe
                else:
                    use = self.technologies['chp'].use(step,weather['temp_air'][step],pb[strategy],pb[coproduct])                    # saving chp working parameters for the current timeframe
                self.power_balance[self.technologies['chp'].th_out]['chp'][step]  = use[0]   # produced thermal output (+) (steam/hot water)
                self.power_balance['electricity']['chp'][step]                    = use[1]   # produced electricity (+)
                self.power_balance[self.technologies['chp'].fuel]['chp'][step]    = use[2]   # fuel required by chp system to run (-)  
                self.power_balance['process heat']['chp'][step]                   = use[3]   # process heat produced by chp system (+)  
                self.power_balance['process hot water']['chp'][step]              = use[4]   # process heat produced by chp system (+)  
                pb[self.technologies['chp'].fuel] += self.power_balance[self.technologies['chp'].fuel]['chp'][step]            
                pb[self.technologies['chp'].th_out] += self.power_balance[self.technologies['chp'].th_out]['chp'][step]    
                pb['electricity'] += self.power_balance['electricity']['chp'][step] 
                pb['process heat'] += self.power_balance['process heat']['chp'][step]             
                self.consumption_logic(self.technologies['chp'].fuel, 'chp', step)
                self.production_logic(self.technologies['chp'].th_out, 'chp', step)
                self.production_logic('electricity', 'chp', step)
                self.production_logic('electricity', 'process heat', step)                                        
                                          
            elif tech_name == 'absorber':  
                self.power_balance['process cold water']['absorber'][step],self.power_balance['process heat']['absorber'][step] = self.technologies['absorber'].use(step,pb['process heat'])  # cold energy produced via the absorption cycle (+)
                pb['process heat'] += self.power_balance['process heat']['absorber'][step]             
                pb['process cold water'] += self.power_balance['process cold water']['absorber'][step]
                self.consumption_logic('process heat', 'absorber', step)
                self.production_logic('process cold water', 'absorber', step)                                                                                        
                
            elif tech_name == 'electrolyzer':
                
                if step == 0:
                    self.battery_available_electricity = 0  
                if 'battery' in self.system and 'hydrogen demand' in self.system:
                    hyd_dem_to_be_satisfied = pb['hydrogen']
                    n_modules_necessary = int(-hyd_dem_to_be_satisfied/self.technologies['electrolyzer'].maxh2prod)+1
                    El_Power_necessary = self.technologies['electrolyzer'].Npower*n_modules_necessary
                    El_available_bat = self.battery_available_electricity*self.technologies['battery'].etaD
                    if pb['electricity'] >= El_Power_necessary:
                        el_input = pb['electricity']
                    else:
                        if (pb['electricity'] + El_available_bat) < El_Power_necessary:
                            el_input = pb['electricity'] + El_available_bat
                        else:
                            El_from_battery = El_Power_necessary - pb['electricity']
                            el_input = pb['electricity'] + El_from_battery 
                else:
                    el_input = pb['electricity'] 
                    
                if self.technologies['electrolyzer'].strategy == 'ammonia production':  
                    if pb['electricity'] > 0:  
                        # Calculation of the part of energy for every component with power_allocation, the results new_producible_ammonia and ASR_hydrogen_from_buffer are needed for the ASR calculation
                        quota, new_producible_ammonia, ASR_hydrogen_from_buffer = self.power_allocation(step, producible_hyd, H2tank_availability, producible_ammonia, el_input, weather)
                        self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = self.technologies['electrolyzer'].use(step, storable_hydrogen=producible_hyd, p=el_input * quota, Text=weather['temp_air'][step])
                        if new_producible_ammonia == 0 and self.technologies['ASR'].strategy == 'full-time' and self.technologies['ASR'].only_renewables == False: 
                            min_producible_ammonia = self.technologies['ASR'].min_ammonia # if not enough renewable energy available the ASR works at its minimum load
                            if step != 0:    # correction due to limitations on ramp-rate
                                load = min_producible_ammonia / self.technologies['ASR'].max_ammonia
                                if abs(load - self.technologies['ASR'].load[step-1]) > self.technologies['ASR'].max_ramp_rate:
                                    load = self.technologies['ASR'].load[step-1] - self.technologies['ASR'].max_ramp_rate
                                    min_producible_ammonia = load * self.technologies['ASR'].max_ammonia 
                            ASR_hydrogen_from_buffer = min_producible_ammonia * 3 * c.H2MOLMASS / (2 * c.NH3MOLMASS)
                            if H2tank_availability/(c.timestep*60) >= ASR_hydrogen_from_buffer:
                            # It means that there is not enough renewable energy to activate the ASR and PSA at minimum admissible load even if there is enough hydrogen inside the tank
                                new_producible_ammonia = min_producible_ammonia  
                            elif H2tank_availability/(c.timestep*60) < ASR_hydrogen_from_buffer and self.technologies['electrolyzer'].only_renewables == False:
                                new_producible_ammonia = min_producible_ammonia
                                ASR_hydrogen_from_electrolyzer = ASR_hydrogen_from_buffer - H2tank_availability/(c.timestep*60)
                                ASR_hydrogen_from_buffer = H2tank_availability/(c.timestep*60)
                                self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = self.technologies['electrolyzer'].use(step, storable_hydrogen=producible_hyd, hydrog=ASR_hydrogen_from_electrolyzer, Text=weather['temp_air'][step])
                    elif pb['electricity'] <= 0 and self.technologies['ASR'].strategy == 'full-time' and self.technologies['ASR'].only_renewables == False:
                        min_producible_ammonia = self.technologies['ASR'].min_ammonia # if no renewable energy available the ASR works at it's minimum load
                        if step != 0:    # correction due to limitations on ramp-rate
                            load = min_producible_ammonia / self.technologies['ASR'].max_ammonia
                            if abs(load - self.technologies['ASR'].load[step-1]) > self.technologies['ASR'].max_ramp_rate:
                                load = self.technologies['ASR'].load[step-1] - self.technologies['ASR'].max_ramp_rate
                                min_producible_ammonia = load * self.technologies['ASR'].max_ammonia 
                        ASR_hydrogen_from_buffer = min_producible_ammonia * 3 * c.H2MOLMASS / (2 * c.NH3MOLMASS)
                        if H2tank_availability/(c.timestep*60) >= ASR_hydrogen_from_buffer:
                            new_producible_ammonia = min_producible_ammonia     
                            self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = 0, 0, 0, 0  # the electrolyzer doesn't turn-on if renewable energy is not available
                        else:
                            if self.technologies['electrolyzer'].only_renewables == True:  # electrolyzer activated only when renewable energy is available
                                new_producible_ammonia = 0
                                self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = 0, 0, 0, 0  # the electrolyzer doesn't turn-on if renewable energy is not available
                            elif self.technologies['electrolyzer'].only_renewables == False:
                                new_producible_ammonia = min_producible_ammonia
                                ASR_hydrogen_from_electrolyzer = ASR_hydrogen_from_buffer - H2tank_availability/(c.timestep*60)
                                ASR_hydrogen_from_buffer = H2tank_availability/(c.timestep*60)
                                self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = self.technologies['electrolyzer'].use(step, storable_hydrogen=producible_hyd, hydrog=ASR_hydrogen_from_electrolyzer, Text=weather['temp_air'][step])
                    else:
                        new_producible_ammonia = 0
                        self.power_balance['hydrogen']['electrolyzer'][step], self.power_balance['electricity']['electrolyzer'][step], self.power_balance['oxygen']['electrolyzer'][step], self.power_balance['water']['electrolyzer'][step] = 0, 0, 0, 0  # the electrolyzer doesn't turn-on if renewable energy is not available
                    pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]
                    pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step]
                    pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]
                    pb['water']         += self.power_balance['water']['electrolyzer'][step]
                    
                elif self.technologies['electrolyzer'].strategy == 'hydrogen-first' and self.technologies['electrolyzer'].only_renewables == True: # electrolyzer activated when renewable energy is available
                    if pb['electricity'] > 0: # electrolyzer activated only when renewable energy is available
                        if producible_hyd > 0:
                            self.power_balance['hydrogen']['electrolyzer'][step],   \
                            self.power_balance['electricity']['electrolyzer'][step],\
                            self.power_balance['oxygen']['electrolyzer'][step],     \
                            self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=el_input,Text=weather['temp_air'][step])      # [:2] # hydrogen supplied by electrolyzer(+) # electricity absorbed by the electorlyzer(-) 
                            pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]
                            pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step]
                            pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]
                            pb['water']         += self.power_balance['water']['electrolyzer'][step]
                            
                elif self.technologies['electrolyzer'].strategy == 'hydrogen-first' and self.technologies['electrolyzer'].only_renewables == False: # electrolyzer working both with energy from renewables and from grid, but giving precedence to electricity from renewables
                    if producible_hyd > 0:
                        self.power_balance['hydrogen']['electrolyzer'][step],   \
                        self.power_balance['electricity']['electrolyzer'][step],\
                        self.power_balance['oxygen']['electrolyzer'][step],     \
                        self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=el_input,Text=weather['temp_air'][step])      # hydrogen [kg/s] and oxygen [kg/s] produced by the electrolyzer (+) electricity [kW] and water absorbed [m^3/s] (-) 
                    # Evaluating need for grid interaction based on available hydrogen
                    if (available_hyd/(c.timestep*60) + self.power_balance['hydrogen']['electrolyzer'][step]) < -pb['hydrogen']:    # if hydrogen produced from electrolyzer with only renewables and the tank is not sufficient to cover the hydrogen demand in this timestep --> need for grid interaction
                        hyd_from_ele = (-pb['hydrogen']) - available_hyd/(c.timestep*60)                                            # [kg/s] the electrolyzer is required to produce the amount of hydrogen the tank can't cover (thus using also grid electricity) 
                        self.power_balance['hydrogen']['electrolyzer'][step],   \
                        self.power_balance['electricity']['electrolyzer'][step],\
                        self.power_balance['oxygen']['electrolyzer'][step],     \
                        self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,hydrog=hyd_from_ele,Text=weather['temp_air'][step])      # hydrogen [kg/s] and oxygen [kg/s] produced by the electrolyzer (+) electricity [kW] and water absorbed [m^3/s] (-) 
                        # Evaluating the need for grid interaction based on the minimum load of each module: if the electrolyzer operates below its minimum load, it won’t produce hydrogen, and the tank won’t meet the demand. Therefore, the electrolyzer must reach its minimum load, thus the tank won't be completely emptied.
                        if self.power_balance['hydrogen']['electrolyzer'][step] == 0: # no hydrogen has been produced
                            self.power_balance['hydrogen']['electrolyzer'][step],   \
                            self.power_balance['electricity']['electrolyzer'][step],\
                            self.power_balance['oxygen']['electrolyzer'][step],     \
                            self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=self.technologies['electrolyzer'].MinInputPower,Text=weather['temp_air'][step])      # hydrogen [kg/s] and oxygen [kg/s] produced by the electrolyzer (+) electricity [kW] and water absorbed [m^3/s] (-)                                                                                                                                                                                                                                                                                                                                                       
                    # Evaluating need for grid interaction based on imposed constant minimum operational load. Lower functoning boundary.
                    if self.system['electrolyzer']['minimum_load'] and abs(self.power_balance['electricity']['electrolyzer'][step]) < self.technologies['electrolyzer'].min_partial_load:
                        self.power_balance['hydrogen']['electrolyzer'][step],   \
                        self.power_balance['electricity']['electrolyzer'][step],\
                        self.power_balance['oxygen']['electrolyzer'][step],     \
                        self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=self.technologies['electrolyzer'].min_partial_load,Text=weather['temp_air'][step])      # [:2] # hydrogen supplied by electrolyzer(+) # electricity absorbed by the electorlyzer(-) 
                    pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]
                    pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step]
                    pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]
                    pb['water']         += self.power_balance['water']['electrolyzer'][step]
                
                elif self.technologies['electrolyzer'].strategy == 'full-time': # electrolyzer working continuously at each time step of the simulation
                    self.power_balance['hydrogen']['electrolyzer'][step],     \
                    self.power_balance['electricity']['electrolyzer'][step],  \
                    self.power_balance['oxygen']['electrolyzer'][step],       \
                    self.power_balance['water']['electrolyzer'][step]         = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,Text=weather['temp_air'][step])      # [:2] # hydrogen supplied by electrolyzer(+) # electricity absorbed by the electorlyzer(-) 
                    pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]
                    pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step]
                    pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]
                    pb['water']         += self.power_balance['water']['electrolyzer'][step]
                    
                if 'mechanical compressor' in self.system and not self.technologies['electrolyzer'].strategy == 'ammonia production':
                    pass                      #if there is mechanical compressor the electrolyzer is updated there, but for the ammonia production the power are already allocated
                else:                        
                    self.consumption_logic('electricity', 'electrolyzer', step)
                    self.consumption_logic('water', 'electrolyzer', step)
                    self.production_logic('hydrogen', 'electrolyzer', step) 
                    self.production_logic('oxygen', 'electrolyzer', step)                                                                      
   
                if step == (c.timestep_number - 1) and ('hydrogen demand' in self.system or 'HP hydrogen demand' in self.system):
                    if self.system[self.hydrogen_demand+' demand']['strategy'] == 'supply-led':  # activates only at the final step of simulation
                        self.constant_flow = sum(self.power_balance['hydrogen']['electrolyzer'])/c.timestep_number # [kg/s] constant hydrogen output based on the total production                   
                    
            elif tech_name == 'PSA': 
                if self.technologies['electrolyzer'].strategy == 'ammonia production':
                    producible_nitro = self.power_balance['ammonia']['ASR'][step] * c.N2MOLMASS / (2*c.NH3MOLMASS)  # nitrogen in stoichiometric ratio
                    if producible_nitro > 0:           
                        self.power_balance['nitrogen']['PSA'][step], self.power_balance['electricity']['PSA'][step] = self.technologies['PSA'].use(producible_nitro, step)                                                           
                        pb['nitrogen']      += self.power_balance['nitrogen']['PSA'][step]
                        pb['electricity']   += self.power_balance['electricity']['PSA'][step]
                        self.consumption_logic('electricity', 'PSA', step)
                        self.production_logic('nitrogen', 'PSA', step)                         
                        
            elif tech_name == 'ASR': 
                 if self.technologies['electrolyzer'].strategy == 'ammonia production':
                   if self.technologies['ASR'].strategy == 'full-time' and new_producible_ammonia == 0:
                           raise ValueError ('Warning: the selected strategy for the ASR is full-time, but the H tank and/or the electrolyzer and/or the energy generation is/are too small to guarantee it')
                   if new_producible_ammonia > 0:  
                       # Ammonia production computed inside the electrolyzer, all the conditions has been already verified, new_producible_ammonia and ASR_hydrogen_from_buffer are already the results
                       ASR_hydrogen = new_producible_ammonia * 3 * c.H2MOLMASS / (2*c.NH3MOLMASS)  # stoichiometry
                       self.power_balance['ammonia']['ASR'][step], self.power_balance['electricity']['ASR'][step], self.power_balance['nitrogen']['ASR'][step], self.power_balance['hydrogen']['ASR'][step], _ = self.technologies['ASR'].use(step, ASR_hydrogen, self.technologies['H tank'].LOC[step]/self.technologies['H tank'].max_capacity, ASR_hydrogen_from_buffer, self.technologies['H tank'].LOP[step])
                       pb['hydrogen']      += self.power_balance['hydrogen']['ASR'][step]
                       pb['electricity']   += self.power_balance['electricity']['ASR'][step]
                       pb['ammonia']        += self.power_balance['ammonia']['ASR'][step]
                       pb['nitrogen']         += self.power_balance['nitrogen']['ASR'][step]
                       if abs(pb['electricity']) < 0.0000001: #remove eventual approximation errors that could cause bugs in the following components
                            pb['electricity'] = 0
                       self.consumption_logic('electricity', 'ASR', step)
                       self.consumption_logic('nitrogen', 'ASR', step)
                       self.consumption_logic('hydrogen', 'ASR', step)
                       self.production_logic('ammonia', 'ASR', step)                   
                    
            elif tech_name == 'mhhc compressor':  
                if self.power_balance['hydrogen']['electrolyzer'][step] > 0:
                    storable_hydrogen = producible_hyd                
                    if storable_hydrogen>self.technologies['H tank'].max_capacity*0.00001:
                        self.power_balance['hydrogen']['mhhc compressor'][step], self.power_balance['gas']['mhhc compressor'][step] = self.technologies['mhhc compressor'].use(step,self.power_balance['hydrogen']['electrolyzer'][step],storable_hydrogen) # hydrogen compressed by the compressor (+) and heat requested to make it work expressed as heating water need (-) 
                        pb['gas'] += self.power_balance['gas']['mhhc compressor'][step]
                        #pb['hydrogen']=...self.power_balance['hydrogen']['mhhc compressor'][step]?? come ne tengo conto di quanto comprimo? in linea teorica ne dovrei sempre comprimere esattamente quanto me ne entra perchè il controllo sullo sotrable hydrogen lho gia fatto nell'elettrolizzatore'
                        self.consumption_logic('gas', 'mhhc compressor', step)                                   
                                          
            elif tech_name == 'mechanical compressor':   
                
                if self.technologies['electrolyzer'].strategy == 'ammonia production':
                    if pb['electricity'] > 0 and pb['hydrogen'] > 0:
                        self.power_balance['hydrogen']['mechanical compressor'][step], \
                        self.power_balance['electricity']['mechanical compressor'][step] = self.technologies['mechanical compressor'].use(step,massflowrate = pb['hydrogen'], p_H2_tank = self.technologies['H tank'].LOP[step])[:2]
                        pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                        self.consumption_logic('electricity', 'mechanical compressor', step)
                
                else:
                    if 'HPH tank' not in self.system:
                        
                        if 'O2 tank' not in self.system:
                            
                            if "electricity grid" in self.system and self.system["electricity grid"]["draw"] and self.technologies['mechanical compressor'].only_renewables == False:
                                if 'hydrogen demand' in self.system:
                                    massflow = max(0,self.power_balance['hydrogen']['electrolyzer'][step] + self.power_balance['hydrogen']['hydrogen demand'][step])
                                else:
                                    massflow = max(0,self.power_balance['hydrogen']['electrolyzer'][step])
                                self.power_balance['hydrogen']['mechanical compressor'][step], \
                                self.power_balance['electricity']['mechanical compressor'][step] = self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step])[:2] # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                self.consumption_logic('electricity', 'electrolyzer', step)
                                self.consumption_logic('water', 'electrolyzer', step)
                                self.production_logic('hydrogen', 'electrolyzer', step) 
                                self.production_logic('oxygen', 'electrolyzer', step)
                                self.consumption_logic('electricity', 'mechanical compressor', step) 
                                
                            elif "electricity grid" not in self.system or self.technologies['mechanical compressor'].only_renewables == True:  #self.system["electricity grid"]["draw"] == False:   # if the system is configurated as fully off-grid, relying only on RES production
                                if self.power_balance['hydrogen']['electrolyzer'][step] > 0:  # if hydrogen has been produced by the electrolyzer and electricity is available in the system
                                    if 'hydrogen demand' in self.system:
                                        demand = self.power_balance['hydrogen']['hydrogen demand'][step] # hydrogen demand at timestep h
                                        massflow = max([0, self.power_balance['hydrogen']['electrolyzer'][step] + demand] )  # hydrogen mass flow rate to be compressed and stored in hydrogen tank
                                    elif 'CCGT' in self.system and self.power_balance['hydrogen']['CCGT'][step] < 0:
                                        demand = self.power_balance['hydrogen']['CCGT'][step]
                                        massflow = max([0, self.power_balance['hydrogen']['electrolyzer'][step] + self.power_balance['hydrogen']['CCGT'][step]])
                                    else:           # no hydrogen demand
                                        demand = 0  # hydrogen demand at timestep h
                                        massflow = self.power_balance['hydrogen']['electrolyzer'][step]  # in case no hydrogen demand is present all produced hydrogen is compressed and flows through the hydrogne tank
                                    a = self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step])[1] # [kW] compressor energy consumption for a certain h2 mass flow rate
                                    if abs(a) <=pb['electricity'] or a == 0:     # there is enough renewable electricity to power the compressor 
                                        self.power_balance['hydrogen']['mechanical compressor'][step],    \
                                        self.power_balance['electricity']['mechanical compressor'][step], \
                                        self.power_balance['cooling water']['mechanical compressor'][step]= self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step]) # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                        pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                        self.consumption_logic('electricity', 'electrolyzer', step)
                                        self.consumption_logic('water', 'electrolyzer', step)
                                        self.production_logic('hydrogen', 'electrolyzer', step) 
                                        self.production_logic('oxygen', 'electrolyzer', step)
                                        self.consumption_logic('electricity', 'mechanical compressor', step)                                                                                                     
                                    elif abs(a) >pb['electricity']:    # if available electricity in the system is not enough to power the compression system - enter the loop to reallocate the energy among the components
                                        a1  = 1     # % of available electricity fed to the electrolyzer
                                        a11 = 0     # % of available electricity fed to the compressor
                                        en  = pb['electricity'] + abs(self.power_balance['electricity']['electrolyzer'][step]) # [kW] electric energy available at time h before entering the electorlyzer
                                        el  = self.power_balance['electricity']['electrolyzer'][step]
                                        hy  = self.power_balance['hydrogen']['electrolyzer'][step]
                                        ox  = self.power_balance['oxygen']['electrolyzer'][step]
                                        wa  = self.power_balance['water']['electrolyzer'][step]
                                        # Iteration parameters
                                        i   = 0             # initializing iteration count
                                        maxiter = 10000     # max number of iterations allowed
                                        abs_err = 0.00001   # absolute error allowed
                                        while a1 >= 0:       # while loop necessary to iterate in the redistribution of renewable electricity to satisfy both electrolyzer and compressor demand
                                            hydrogen_ele,  \
                                            electricity_ele = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=a1*en)[:2]  # [kg] of produced H2 and [kW] of consumed electricity for the given energy input  
                                            massflow = np.max([0, hydrogen_ele + demand])
                                            a = -self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step])[1] # [kW] compressor energy consumption for a certain h2 mass flow rate
                                            b1 = a/en
                                            a11 = 1-b1
                                            i += 1      # updating iteration count
                                            if abs(a1-a11) < abs_err or i > maxiter:    # strict tolerance for convergence 
                                                break
                                            else: 
                                                a1=a11          
                                        # Electorlyzer balances update and overwriting
                                        self.power_balance['hydrogen']['electrolyzer'][step],   \
                                        self.power_balance['electricity']['electrolyzer'][step],\
                                        self.power_balance['oxygen']['electrolyzer'][step],     \
                                        self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,storable_hydrogen=producible_hyd,p=a1*en)      # [:2] # hydrogen supplied by electrolyzer(+) # electricity absorbed by the electorlyzer(-) 
                                        pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]    - hy
                                        pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step] - el
                                        pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]      - ox
                                        pb['water']         += self.power_balance['water']['electrolyzer'][step]       - wa
                                        self.consumption_logic('electricity', 'electrolyzer', step)
                                        self.consumption_logic('water', 'electrolyzer', step)
                                        self.production_logic('hydrogen', 'electrolyzer', step) 
                                        self.production_logic('oxygen', 'electrolyzer', step)                                                                                     
                                        # Compressor balances update and overwriting
                                        self.power_balance['hydrogen']['mechanical compressor'][step],    \
                                        self.power_balance['electricity']['mechanical compressor'][step], \
                                        self.power_balance['cooling water']['mechanical compressor'][step]   = self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step]) # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                        pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                        self.consumption_logic('electricity', 'mechanical compressor', step)
                                    else:  # if no hydrogen has been produced at time h
                                        self.power_balance['hydrogen']['mechanical compressor'][step]     = 0
                                        self.power_balance['electricity']['mechanical compressor'][step]  = 0
                                        
                        elif 'O2 tank' in self.system:   # simplified approach for oxygen compression. To be updated
                            massflow_tot = (self.power_balance['hydrogen']['electrolyzer'][step])+(self.power_balance['oxygen']['electrolyzer'][step])
                            
                            if "electricity grid" in self.system and self.system["electricity grid"]["draw"]:
                                self.power_balance['hydrogen']['mechanical compressor'][step], \
                                self.power_balance['electricity']['mechanical compressor'][step] = self.technologies['mechanical compressor'].use(step,massflowrate = massflow_tot, p_H2_tank = self.technologies['H tank'].LOP[step])[:2] # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                self.consumption_logic('electricity', 'electrolyzer', step)
                                self.consumption_logic('water', 'electrolyzer', step)
                                self.production_logic('hydrogen', 'electrolyzer', step) 
                                self.production_logic('oxygen', 'electrolyzer', step)
                                self.consumption_logic('electricity', 'mechanical compressor', step)  
                                                                             
                            elif "electricity grid" not in self.system or self.system["electricity grid"]["draw"] == False:   # if the system is configurated as fully off-grid, relying only on RES production
                                if self.power_balance['hydrogen']['electrolyzer'][step] > 0 :  # if hydrogen has been produced by the electrolyzer and electricity is available in the system
                                    a = self.technologies['mechanical compressor'].use(step,massflowrate = massflow_tot, p_H2_tank = self.technologies['H tank'].LOP[step])[1] # [kW] compressor energy consumption for a certain h2 mass flow rate
                                    if abs(a) <=pb['electricity'] or a == 0:    # there is enough renewable electricity to power the compressor 
                                        self.power_balance['hydrogen']['mechanical compressor'][step],    \
                                        self.power_balance['electricity']['mechanical compressor'][step], \
                                        self.power_balance['cooling water']['mechanical compressor'][step]= self.technologies['mechanical compressor'].use(step,massflowrate= massflow_tot, p_H2_tank = self.technologies['H tank'].LOP[step]) # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                        pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                        self.consumption_logic('electricity', 'electrolyzer', step)
                                        self.consumption_logic('water', 'electrolyzer', step)
                                        self.production_logic('hydrogen', 'electrolyzer', step) 
                                        self.production_logic('oxygen', 'electrolyzer', step)
                                        self.consumption_logic('electricity', 'mechanical compressor', step)                                                                 
                                    elif abs(a) > pb['electricity']:    # if available electricity in the system is not enough to power the compression system - enter the loop to reallocate the energy among the components
                                        a1  = 1     # % of available electricity fed to the electrolyzer
                                        a11 = 0     # % of available electricity fed to the compressor
                                        en  =pb['electricity'] + abs(self.power_balance['electricity']['electrolyzer'][step]) # [kW] electric energy available at time h before entering the electorlyzer
                                        el  = self.power_balance['electricity']['electrolyzer'][step]
                                        hy  = self.power_balance['hydrogen']['electrolyzer'][step]
                                        ox  = self.power_balance['oxygen']['electrolyzer'][step]
                                        wa  = self.power_balance['water']['electrolyzer'][step]
                                        # Iteration parameters
                                        i   = 0             # initializing iteration count
                                        maxiter = 10000     # max number of iterations allowed
                                        abs_err = 0.00001   # absolute error allowed
                                        while a1 >= 0:       # while loop necessary to iterate in the redistribution of renewable electricity to satisfy both electrolyzer and compressor demand
                                            hydrogen_ele,  \
                                            electricity_ele = self.technologies['electrolyzer'].use(step,a1*en,producible_hyd)[:2]  # [kg] of produced H2 and [kW] of consumed electricity for the given energy input  
                                            a = -self.technologies['mechanical compressor'].use(step,massflowrate= hydrogen_ele + hydrogen_ele*7.93, p_H2_tank = self.technologies['H tank'].LOP[step])[1] # [kW] compressor energy consumption for a certain h2 mass flow rate
                                            b1 = a/en
                                            a11 = 1-b1
                                            i += 1      # updating iteration count
                                        
                                            if abs(a1-a11) < abs_err or i > maxiter:    # strict tolerance for convergence 
                                                break
                                            else: 
                                                a1=a11         
                                        # Electrolyzer balances update and overwriting
                                        self.power_balance['hydrogen']['electrolyzer'][step],   \
                                        self.power_balance['electricity']['electrolyzer'][step],\
                                        self.power_balance['oxygen']['electrolyzer'][step],     \
                                        self.power_balance['water']['electrolyzer'][step]        = self.technologies['electrolyzer'].use(step,a1*en,producible_hyd)      # [:2] # hydrogen supplied by electrolyzer(+) # electricity absorbed by the electorlyzer(-)                                  
                                        pb['hydrogen']      += self.power_balance['hydrogen']['electrolyzer'][step]    - hy
                                        pb['electricity']   += self.power_balance['electricity']['electrolyzer'][step] - el
                                        pb['oxygen']        += self.power_balance['oxygen']['electrolyzer'][step]      - ox
                                        pb['water']         += self.power_balance['water']['electrolyzer'][step]       + wa
                                        self.consumption_logic('electricity', 'electrolyzer', step)
                                        self.consumption_logic('water', 'electrolyzer', step)
                                        self.production_logic('hydrogen', 'electrolyzer', step) 
                                        self.production_logic('oxygen', 'electrolyzer', step)                                                   
                                        # Compressor balances update and overwriting
                                        self.power_balance['hydrogen']['mechanical compressor'][step],    \
                                        self.power_balance['electricity']['mechanical compressor'][step], \
                                        self.power_balance['cooling water']['mechanical compressor'][step]   = self.technologies['mechanical compressor'].use(step,massflowrate= massflow, p_H2_tank = self.technologies['H tank'].LOP[step]) # hydrogen compressed by the compressor (+) and electricity consumption (-) 
                                        pb['electricity']   += self.power_balance['electricity']['mechanical compressor'][step]
                                        self.consumption_logic('electricity', 'mechanical compressor', step)
                                    else:  # if no hydrogen has been produced at time h
                                        self.power_balance['hydrogen']['mechanical compressor'][step]     = 0
                                        self.power_balance['electricity']['mechanical compressor'][step]  = 0 
                                        
                    if 'H tank' in self.system and 'HPH tank' in self.system:
                        # self.power_balance['hydrogen']['H tank'][step] = self.technologies['H tank'].use(h,pb['hydrogen'])
                        #pb['hydrogen'] += self.power_balance['hydrogen']['H tank'][step]
                        available_hyd_lp = self.technologies['H tank'].LOC[step] + self.technologies['H tank'].max_capacity - self.technologies['H tank'].used_capacity
                        storable_hydrogen_hp = self.technologies['HPH tank'].max_capacity-self.technologies['HPH tank'].LOC[step]
                        if self.technologies['HPH tank'].LOC[step] == self.technologies['HPH tank'].max_capacity:  # if High-Pressure-Tank is full 
                            self.power_balance['HP hydrogen']['mechanical compressor'][step]      = 0     
                            self.power_balance['electricity']['mechanical compressor'][step]      = 0    
                            self.power_balance['cooling water']['mechanical compressor'][step]    = 0
                            self.power_balance['hydrogen']['mechanical compressor'][step]         = 0
                            pb['HP hydrogen']    += 0
                            pb['electricity']    += 0   # compressor not working
                            pb['hydrogen']       += 0   # compressor not working
                            self.consumption_logic('electricity', 'electrolyzer', step)
                            self.consumption_logic('water', 'electrolyzer', step)
                            self.production_logic('hydrogen', 'electrolyzer', step) 
                            self.production_logic('oxygen', 'electrolyzer', step)                                                               
                        else:  # if there is enough room available in the High Pressure Tank, the compressor is activated
                            self.power_balance['HP hydrogen']['mechanical compressor'][step],     \
                            self.power_balance['electricity']['mechanical compressor'][step],     \
                            self.power_balance['cooling water']['mechanical compressor'][step]    = self.technologies['mechanical compressor'].use(step, available_hyd_lp=available_hyd_lp ,storable_hydrogen_hp=storable_hydrogen_hp, p_H2_tank = self.technologies['H tank'].LOP[step]) # hydrogen supplied by H tank (+) and electricity absorbed(-) 
                            self.power_balance['hydrogen']['mechanical compressor'][step] = - self.power_balance['HP hydrogen']['mechanical compressor'][step]
                            pb['HP hydrogen'] += self.power_balance['HP hydrogen']['mechanical compressor'][step]
                            pb['hydrogen']    += self.power_balance['hydrogen']['mechanical compressor'][step]
                            pb['electricity'] += self.power_balance['electricity']['mechanical compressor'][step]
                            self.consumption_logic('electricity', 'electrolyzer', step)
                            self.consumption_logic('water', 'electrolyzer', step)
                            self.production_logic('hydrogen', 'electrolyzer', step) 
                            self.production_logic('oxygen', 'electrolyzer', step)
                            self.consumption_logic('hydrogen', 'mechanical compressor', step)
                            self.consumption_logic('electricity', 'mechanical compressor', step)
                            self.production_logic('HP hydrogen', 'mechanical compressor', step)                                                                       

            elif tech_name == 'fuel cell':
                if pb['electricity'] < 0: 
                    available_hyd = available_hyd - (-pb['hydrogen'])*c.timestep*60
                    if available_hyd > 0:
                        use = self.technologies['fuel cell'].use(step,pb['electricity'],available_hyd)     # saving fuel cell working parameters for the current timeframe
                        self.power_balance['hydrogen']['fuel cell'][step] =    use[0] # hydrogen absorbed by fuel cell(-)
                        self.power_balance['electricity']['fuel cell'][step] = use[1] # electricity supplied(+) 
                        if use[2] < -pb['heating water']: #all of the heat producted by FC is used      
                            self.power_balance['heating water']['fuel cell'][step]=use[2] # heat loss of fuel cell
                        else:
                            self.power_balance['heating water']['fuel cell'][step]=-pb['heating water'] # heat loss of fuel cell- demand
                        pb['hydrogen'] += self.power_balance['hydrogen']['fuel cell'][step]
                        pb['electricity'] += self.power_balance['electricity']['fuel cell'][step]
                        pb['heating water'] += self.power_balance['heating water']['fuel cell'][step] 
                        self.production_logic('electricity', 'fuel cell', step)
                        self.consumption_logic('hydrogen','fuel cell', step)
                        self.production_logic('heating water', 'fuel cell', step)                                                                                   
            
            elif tech_name == 'SMR':
                if pb['hydrogen'] < 0:      # currently activated only in presence of hydrogen demand
                    self.power_balance['gas']['SMR'][step], self.power_balance['hydrogen']['SMR'][step] = self.technologies['SMR'].use(pb['hydrogen']) # NG consumed and hydrogen produced from SMR
                    pb['gas']       += self.power_balance['gas']['SMR'][step]       # gas balance update: - gas consumed by SMR
                    pb['hydrogen']  += self.power_balance['hydrogen']['SMR'][step]  # hydrogen balance update: + hydrogen produced by SMR                   
                    self.production_logic('hydrogen', 'SMR', step)
                    self.consumption_logic('gas', 'SMR', step)                                              
        
            elif tech_name == 'cracker':
                if pb['electricity'] < 0:
                    if self.technologies['cracker'].heat_source == 'electric' and 'fuel cell' in self.technologies:
                        def cracker_FC_balance(electricity):
                            hyd_FC, electricity_FC, *_ = self.technologies['fuel cell'].use(step, -electricity, 1e100)     
                            ammonia_cracker, hyd_cracker, electricity_cracker = self.technologies['cracker'].use(abs(hyd_FC), available_ammonia)
                            return electricity_FC - abs(pb['electricity'] + electricity_cracker)
                        electricity = brentq(cracker_FC_balance, abs(pb['electricity']), self.technologies['fuel cell'].Npower * self.technologies['fuel cell'].n_modules, xtol=10e-5, maxiter=100)
                    else:
                        electricity = abs(pb['electricity'])
                    hyd, *_ = self.technologies['fuel cell'].use(step, -electricity, 1e100)
                    self.power_balance['ammonia']['cracker'][step], self.power_balance['hydrogen']['cracker'][step], self.power_balance['electricity']['cracker'][step] = self.technologies['cracker'].use(abs(hyd), available_ammonia)
                    pb['hydrogen']      += self.power_balance['hydrogen']['cracker'][step]
                    pb['electricity']   += self.power_balance['electricity']['cracker'][step]
                    pb['ammonia']        += self.power_balance['ammonia']['cracker'][step]
                    self.consumption_logic('ammonia', 'cracker', step)
                    self.consumption_logic('electricity', 'cracker', step)
                    self.production_logic('hydrogen', 'cracker', step)       
        
            elif tech_name == 'H tank':
                if 'HPH tank' not in self.system and ('hydrogen demand' in self.system or 'HP hydrogen demand' in self.system):
                    if self.system[self.hydrogen_demand+' demand']['strategy'] == 'demand-led':
                        self.power_balance['hydrogen']['H tank'][step] = self.technologies['H tank'].use(step,pb['hydrogen'])
                        pb['hydrogen'] += self.power_balance['hydrogen']['H tank'][step]
                    elif self.system[self.hydrogen_demand+' demand']['strategy'] == 'supply-led' and step == (c.timestep_number - 1):
                        prod = self.power_balance['hydrogen']['electrolyzer']
                        for step in range(c.timestep_number):
                            self.power_balance['hydrogen']['H tank'][step] = self.technologies['H tank'].use(step,prod[step],constant_demand=self.constant_flow )                        
                    # else:
                        # pass
                else:
                    self.power_balance['hydrogen']['H tank'][step] = self.technologies['H tank'].use(step,pb['hydrogen'])
                    pb['hydrogen'] += self.power_balance['hydrogen']['H tank'][step]
                if self.power_balance['hydrogen']['H tank'][step] >= 0:
                    self.production_logic('hydrogen', 'H tank', step)
                elif self.power_balance['hydrogen']['H tank'][step] <= 0:
                    self.consumption_logic('hydrogen', 'H tank', step)                                                              
                        
            elif tech_name == 'HPH tank':
                self.power_balance['HP hydrogen']['HPH tank'][step] = self.technologies['HPH tank'].use(step,pb['HP hydrogen'])
                pb['HP hydrogen'] += self.power_balance['HP hydrogen']['HPH tank'][step]
                if self.power_balance['hydrogen']['HPH tank'][step] >= 0:
                    self.production_logic('hydrogen', 'HPH tank', step)
                elif self.power_balance['hydrogen']['HPH tank'][step] <= 0:
                    self.consumption_logic('hydrogen', 'HPH tank', step)  
                                       
            elif tech_name == 'NH3 tank':
                self.power_balance['ammonia']['NH3 tank'][step] = self.technologies['NH3 tank'].use(step,pb['ammonia'])
                pb['ammonia'] += self.power_balance['ammonia']['NH3 tank'][step]    
                if self.power_balance['ammonia']['NH3 tank'][step] >= 0:
                    self.production_logic('ammonia', 'NH3 tank', step)
                elif self.power_balance['ammonia']['NH3 tank'][step] <= 0:
                    self.consumption_logic('ammonia', 'NH3 tank', step)
                                                                           
            elif tech_name == 'O2 tank':
                if 'oxygen demand' in self.system and self.system['oxygen demand']['strategy'] != 'supply-led':
                    self.power_balance['oxygen']['O2 tank'][step] = self.technologies['O2 tank'].use(step,pb['oxygen'])
                    pb['oxygen'] += self.power_balance['oxygen']['O2 tank'][step]
                elif self.system['hydrogen demand']['strategy'] == 'supply-led' and step == (c.timestep_number - 1):
                    self.technologies['O2 tank'].sizing(self.technologies['H tank'].max_capacity)
                else:
                    pass
                if self.power_balance['oxygen']['O2 tank'][step] >= 0:
                    self.production_logic('oxygen', 'O2 tank', step)
                elif self.power_balance['oxygen']['O2 tank'][step] <= 0:
                    self.consumption_logic('oxygen', 'O2 tank', step)                                                       
                      
            elif tech_name == 'inverter':
                self.power_balance['electricity']['inverter'][step] = self.technologies['inverter'].use(step,pb['electricity']) # electricity lost in conversion by the inverter
                pb['electricity'] += self.power_balance['electricity']['inverter'][step] # electricity balance update: - electricity lost in conversion by the invertert
                self.consumption_logic('electricity', 'inverter', step)
                        
            elif tech_name == 'CCGT':
                if pb['electricity'] < 0:
                    electricity_demand = pb['electricity']
                    if self.technologies['CCGT'].fuel == 'ammonia':
                        available_fuel = available_ammonia
                    elif self.technologies['CCGT'].fuel == 'hydrogen':
                        available_fuel = available_hyd / (c.timestep * 60)
                    self.power_balance[self.technologies['CCGT'].fuel]['CCGT'][step], self.power_balance['electricity']['CCGT'][step] = self.technologies['CCGT'].use(step, electricity_demand, available_fuel)     
                    pb[self.technologies['CCGT'].fuel] += self.power_balance[self.technologies['CCGT'].fuel]['CCGT'][step]             
                    pb['electricity'] += self.power_balance['electricity']['CCGT'][step] 
                    self.consumption_logic(self.technologies['CCGT'].fuel, 'CCGT', step)
                    self.production_logic('electricity', 'CCGT',step)              
                
            ### demand and grid   
            for carrier in pb: # for each energy carrier
                if tech_name == f"{carrier} demand":                
                    pb[carrier] += self.power_balance[carrier][tech_name][step]    # power balance update: energy demand(-)
                    if self.power_balance[carrier][tech_name][step] >= 0:
                        self.production_logic(carrier, tech_name, step)
                    elif self.power_balance[carrier][tech_name][step] <= 0:
                        self.consumption_logic(carrier, tech_name, step)                                          
                    break
                
                if tech_name == f"{carrier} grid":
                    if pb[carrier] > 0 and self.system[f"{carrier} grid"]['feed'] or pb[carrier] < 0 and self.system[f"{carrier} grid"]['draw']:
                        self.power_balance[carrier][tech_name][step] = -pb[carrier] # energy from grid(+) or into grid(-) 
                        pb[carrier] += self.power_balance[carrier][tech_name][step]  # electricity balance update
                        if self.power_balance[carrier][tech_name][step] >= 0:
                            self.production_logic(carrier, tech_name, step)
                        elif self.power_balance[carrier][tech_name][step] <= 0:
                            self.consumption_logic(carrier, tech_name, step)                                            
                        break
                    
#%%            
        ### Global check on power balances at the end of every timestep
        for carrier in pb:
            if carrier == 'heating water':
                continue
            tol = 0.0001  # [-] tolerance on error
            if pb[carrier] != 0:
                maxvalues = []
                for arr in self.power_balance[carrier]:
                    maxvalues.append(np.max(self.power_balance[carrier][arr]))
                m = np.max(np.array(maxvalues))
                if abs(pb[carrier]) > abs(m*tol):
                    if pb[carrier] >0:  sign = 'positive'
                    else:               sign = 'negative'
                    raise ValueError(f'Warning: {carrier} balance at the end of timestep {step} shows {sign} value of {round(pb[carrier],2)} \n\
                    It means there is an overproduction not fed to grid or demand is not satisfied.\n\
                    Options to fix the problem: \n\
                        (a) - Include {carrier} grid[\'draw\']: true if negative or {carrier} grid[\'feed\']: true if positive in studycase.json \n\
                        (b) - Vary components size or demand series in studycase.json')

#%%
        #### Cleaning of production and consumption dictionaries at the last timestep
        if step == (c.timestep_number - 1):
            cleaned_consumption = {}
            
            for carrier in self.consumption:
                cleaned_consumption[carrier] = {}
                
                for tech_name in self.consumption[carrier]:
                    cleaned_tech = {
                        tech: value for tech, value in self.consumption[carrier][tech_name].items()
                        if not all(x == 0 for x in value)
                    }
                    if cleaned_tech:
                        # Remove tech name and aux variable
                        cleaned_tech.pop(tech_name, None)
                        cleaned_tech.pop('Aux', None)
                        
                        if cleaned_tech:
                            cleaned_consumption[carrier][tech_name] = cleaned_tech
                    
                
                # Remove carrier if not present in the analysis
                if not cleaned_consumption[carrier]:
                    del cleaned_consumption[carrier]
            
            self.consumption = cleaned_consumption
            
            # Adding total consumption for each technology
            
            for carrier in self.consumption:
                for tech_name in self.consumption[carrier]:
                    self.consumption[carrier][tech_name]['Tot'] = np.zeros(c.timestep_number)
                    for tech in self.consumption[carrier][tech_name]:
                        if tech != 'Tot':  
                            self.consumption[carrier][tech_name]['Tot'] += self.consumption[carrier][tech_name][tech]
                
        if step == (c.timestep_number - 1):
            cleaned_production = {}
            
            for carrier in self.production:
                cleaned_production[carrier] = {}
                
                for tech_name in self.production[carrier]:
                    cleaned_tech = {
                        tech: value for tech, value in self.production[carrier][tech_name].items()
                        if not all(x == 0 for x in value)
                    }
                    if cleaned_tech:
                        # Remove tech name and aux variable
                        cleaned_tech.pop(tech_name, None)
                        cleaned_tech.pop('Aux', None)
                        
                        if cleaned_tech:
                            cleaned_production[carrier][tech_name] = cleaned_tech
                
               # Remove carrier if not present in the analysis
                if not cleaned_production[carrier]:
                    del cleaned_production[carrier]
            
            self.production = cleaned_production
            
            # Adding total production for each technology
            
            for carrier in self.production:
                for tech_name in self.production[carrier]:
                    self.production[carrier][tech_name]['Tot'] = np.zeros(c.timestep_number)
                    for tech in self.production[carrier][tech_name]:
                        if tech != 'Tot':  
                            self.production[carrier][tech_name]['Tot'] += self.production[carrier][tech_name][tech]