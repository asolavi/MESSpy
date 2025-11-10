"""
MESSpy - preprocessing

"""
from core import constants as c
from CoolProp.CoolProp import PropsSI
import math

def change_peakP(structure,location_name,peakP):
    structure[location_name]['PV']['peakP'] = peakP
    return(structure)

def change_peakW(structure,location_name,peakW):
    structure[location_name]['wind']['Npower'] = peakW
    return(structure)

def change_batterysize(structure,location_name,batterysize):
    structure[location_name]['battery']['nominal capacity'] = batterysize
    return(structure)

def change_Elesize(structure,location_name,elesize):
    structure[location_name]['electrolyzer']['number of modules'] = elesize
    return(structure)

def change_Htanksize(structure,location_name,tanksize):
    structure[location_name]['H tank']['max capacity'] = tanksize
    return(structure)

def change_NH3tanksize(structure,location_name,tanksize):
    structure[location_name]['NH3 tank']['max capacity'] = tanksize
    return(structure)

def change_ASRsize(structure,location_name,ASRsize):
    structure[location_name]['ASR']['max_prod'] = ASRsize
    return(structure)

def change_PSAsize(structure,location_name):
    nitro = structure[location_name]['ASR']['max_prod'] * c.N2MOLMASS / (2*c.NH3MOLMASS)   # [kg/s]
    nitro_unit = structure[location_name]['PSA']['Nflowrate'] / 3600 * PropsSI('D', 'T', 273.15, 'P', 101325, 'N2')  # [kg/s]
    PSAsize = math.ceil(nitro / nitro_unit)
    structure[location_name]['PSA']['number of units'] = PSAsize
    return(structure)

def change_CCGTsize(structure,location_name,CCGTsize):
    structure[location_name]['CCGT']['Npower'] = CCGTsize
    return(structure)

def change_electricityprice(energy_market,electricity_price):
    energy_market['electricity']['purchase'] = electricity_price
    return(energy_market)

def change_windeleprice(energy_market,windele_price):
    energy_market['wind electricity']['purchase'] = windele_price
    return(energy_market)

def change_O2tanksize(structure,location_name,O2_tanksize):
    structure[location_name]['O2 tank']['max capacity'] = O2_tanksize
    return(structure)

def change_O2demand(structure,location_name,amount):
    structure[location_name]['oxygen demand']['amount'] = amount
    return(structure)

def change_O2tankprice(tech_cost,O2tank_price):
    tech_cost['O2 tank']['cost per unit'] = O2tank_price
    return(tech_cost)

def change_O2sellingprice(energy_market,O2price):
    energy_market['oxygen']['sale'] = O2price
    return(energy_market)