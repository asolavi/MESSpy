#%% ###########################################################################
"""
PRE PROCESSING
==============
"""
# Import modules: do not edit
from core import rec, location
from core import economics as eco
from core import constants as c
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import modules: change if willing to use different pre- and post-process modules
import preprocess_test as pre
import postprocess_test as pp

path = r'./input_test_4' 

name_studycase  = 'Ammonia' # illustrative name for saving the results of the studycase simulation

# Selecting the names of the .json files to be read as simulations input: 
file_tech_cost      = 'tech_cost'
file_energy_market  = 'energy_market'
file_general        = 'general'
file_studycase      = 'studycase'


# Opening input files:
with open(os.path.join(path,f"{file_studycase}.json"),'r')      as f: studycase     = json.load(f)
with open(os.path.join(path,f"{file_general}.json"),'r')        as f: general       = json.load(f)
with open(os.path.join(path,f"{file_tech_cost}.json"),'r')      as f: tech_cost     = json.load(f)
with open(os.path.join(path,f"{file_energy_market}.json"),'r')  as f: energy_market = json.load(f)


#%% ###########################################################################
"""
SOLVER - studycase simulation
======
"""
from scipy.optimize import brentq

# For ammonia synthesis with hydrogen buffer, an iteration on the level of hydrogen in the buffer is needed to correctly operate the control strategy

studycase['electricity_consumer']['H tank']['used capacity'] = studycase['electricity_consumer']['H tank']['max capacity']
def convergence_tank(initial_LOC): 
    studycase['electricity_consumer']['H tank']['initial LOC'] = initial_LOC
    sim = rec.REC(studycase,general,file_studycase,file_general,path) 
    try:
        sim.REC_power_simulation()
        LOC_0 = sim.locations['electricity_consumer'].technologies['H tank'].LOC[0]
        LOC_end = sim.locations['electricity_consumer'].technologies['H tank'].LOC[-1]
        delta_tank = LOC_0 - LOC_end
    except ValueError:
        delta_tank = -float('inf') 
    iterations.append((initial_LOC, delta_tank))
    return delta_tank

lower_bound = studycase['electricity_consumer']['H tank']['max capacity'] * studycase['electricity_consumer']['H tank']['min level']
upper_bound = studycase['electricity_consumer']['H tank']['max capacity']
iterations = [] 
try:
    initial_LOC = brentq(convergence_tank, lower_bound, upper_bound, xtol=2, maxiter=30)
    if iterations[-1][1] != 0:
        best_negative = min([x for x in iterations if x[1] < 0], key=lambda x: abs(x[1]), default=None)
        if best_negative:
            initial_LOC = best_negative[0]
except ValueError:
    if convergence_tank(lower_bound) == -float('inf') and convergence_tank(upper_bound) == -float('inf'):
        raise ValueError('Warning: the selected strategy for the ASR is full-time, but the H tank and/or the electrolyzer and/or the energy generation is/are too small to guarantee it')
    else:
        raise
studycase['electricity_consumer']['H tank']['initial LOC'] = initial_LOC
sim = rec.REC(studycase,general,file_studycase,file_general,path) 
sim.REC_power_simulation()
    
if 'initial LOC' in studycase['electricity_consumer']['H tank']:
    del studycase['electricity_consumer']['H tank']['initial LOC']
if 'used capacity' in studycase['electricity_consumer']['H tank']:
    del studycase['electricity_consumer']['H tank']['used capacity']
    
sim.tech_cost(tech_cost) # calculate the cost of all technologies
sim.save(name_studycase,'pkl')   # saving results in .pkl format (useful if you make postprocess using python)
sim.save(name_studycase,'csv')   # saving results in .csv (useful if you make postprocess using other languages/programmes)

    
#%% ###########################################################################
"""
POST PROCESS - LCOE calculation
================================
"""

LCOE = eco.LCOE('electricity_consumer',studycase,name_studycase,energy_market,path,revenues=False,refund=False,plot=True,print_=True)

#%% ###########################################################################
"""
POST PROCESS - PLOTTING
================================
some post-process are alredy avaiable as examples in postprocess_test
you should create your own postprocess_dev.py and create your own graphs
"""
# Here the main simulation results are read: balances, balances0 and economic. Balances are also available in the Variable Explorer panel: sim and sim0. Results are also available in .csv format in results/csv folder.
with open('results/pkl/balances_'+name_studycase+'.pkl', 'rb')              as f: balances  = pickle.load(f)
with open('results/pkl/production_'+name_studycase+'.pkl', 'rb')            as f: production  = pickle.load(f)
with open('results/pkl/consumption_'+name_studycase+'.pkl', 'rb')           as f: consumption  = pickle.load(f) 
with open('results/pkl/LOC_'+name_studycase+'.pkl', 'rb')                   as f: LOC  = pickle.load(f)
with open('results/pkl/LOP_'+name_studycase+'.pkl', 'rb')                   as f: LOP  = pickle.load(f)

# Balance results

pp.demand_plot(name_studycase, 'electricity_consumer')

pp.plot_energy_balances(name_studycase,'electricity_consumer',80,85,'electricity',width=0.9)
pp.plot_energy_balances(name_studycase,'electricity_consumer',220,225,'electricity',width=0.9)
pp.plot_energy_balances(name_studycase,'electricity_consumer',80,85,'ammonia',width=0.9)
pp.plot_energy_balances(name_studycase,'electricity_consumer',220,225,'ammonia',width=0.9)

pp.LOC_plot(name_studycase, studycase)
pp.LOP_plot(name_studycase)
pp.monthly_balances(name_studycase, 'electricity_consumer', ['electricity', 'hydrogen', 'ammonia'])
pp.print_and_plot_annual_energy_balances(name_studycase,'electricity_consumer', print_= True)






