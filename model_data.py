#!/usr/bin/env python
# coding: utf-8

# ### Scenario Tree Design
# 
# #### Currently long term scenarios are split into the following lables: 
#     
# TECH : "Advanced" "Moderate" "Conservative"
# LOAD  : "3" :Net-Zero pathway-High electrification,  "2" :2024 Long term outlook  "1": 2021 Long term outloop
# CLIMATE : "SPP1", "SPP3", "SPP5"
# 
# All are organized above complimentarily, e.g. ( optimistic scenarioe are advanced tech devolopement, load electrification is high going towards net zero, and climate change is low and temperature is stable.) Based on the above to build the small test case we design the following 3 directions per stage (9 collapsable scenarios) based on the action taken :  
#         
# OPTIMISTIC:    "3","SPP1" and "Advanced"
# MODERATE :    "2" "SPP3" and "Moderate" 
# IDLE     :    "1","SPP5", and "Conservative"
#     
# From that we can derive the following scenarios: 
#     
# S1 : OPT, OPT
# S2 : OPT, MOD
# S3 : OPT, IDL
# S4 : MOD, OPT
# S5 : MOD, MOD
# S6 : MOD, IDL
# S7 : IDL, OPT
# S8 : IDL, MOD
# S9 : IDL, IDL
# 

# In[2]:


import pandas as pd
import numpy as np
import math
import pickle
import re

main_path = "/Data"

start_time = '2023-01-01'  
end_time   = '2023-12-31'  

# ORIGINAL------------------
with open(main_path + '/tech_costs_scenarios.pkl', 'rb') as f:
    tech_costs = pickle.load(f)
    
with open(main_path + '/load_scenarios.pkl', 'rb') as f:
    load_scen = pickle.load(f)

with open(main_path + '/climate_scenarios.pkl', 'rb') as f:
    climate_scen = pickle.load(f)
#-------------------------
    
### for different horizon tests
# with open(main_path + '/tech_costs_scenarios-144.pkl', 'rb') as f:
#     tech_costs = pickle.load(f)
    
# with open(main_path + '/load_scenarios-144.pkl', 'rb') as f:
#     load_scen = pickle.load(f)

# with open(main_path + '/climate_scenarios-144.pkl', 'rb') as f:
#     climate_scen = pickle.load(f)


load_scen = load_scen.reset_index()
load_scen.index += 1


OmegaS = {1,2,3,4,5,6,7,8,9}
OmegaT = set(range(24))

OmegaState = {1,2}
OmegaHzn = {1,2,3}  # Origianl

#OmegaHzn = {1,3,5}  # Horizon10
#OmegaHzn = {5,10,15}  # Horizon30
#OmegaHzn = {10,15,20}  # Horizon40


ind = {1:((1,1),(1,2)),
       2:((1,2),(1,3)),
       3:((1,3),(1,4)),
       4:((1,4),(1,5)),
       5:((1,5),(1,6)),
       6:((1,6),(1,7)),
       7:((1,7),(1,8)),
       8:((1,8),(1,9)),
       9:((2,1),(2,2)),
       10:((2,2),(2,3)),
       11:((2,4),(2,5)),
       12:((2,5),(2,6)),
       13:((2,7),(2,8)),
       14:((2,8),(2,9))}

# Path Probability per stage

Path_Prob = {1:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             2:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             3:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             4:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             5:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             6:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             7:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             8:{0:1, 1:1, 2:0.3333, 3:0.3333}, 
             9:{0:1, 1:1, 2:0.3333, 3:0.3333}} # Scenarion : {Stage: Probability}

# Probability of a full path taken

Scenario_Prob   = {1: 0.11108889,
                   2: 0.11108889,
                   3: 0.11108889,
                   4: 0.11108889,
                   5: 0.11108889,
                   6: 0.11108889,
                   7: 0.11108889,
                   8: 0.11108889,
                   9: 0.11108889}

# Scenario Labels: 

TECH_scen  ={1:{1:'Moderate', 2:'Advanced'    ,3:'Advanced'},
             2:{1:'Moderate', 2:'Advanced'    ,3:'Moderate'},
             3:{1:'Moderate', 2:'Advanced'    ,3:'Conservative'},
             4:{1:'Moderate', 2:'Moderate'    ,3:'Advanced'},
             5:{1:'Moderate', 2:'Moderate'    ,3:'Moderate'},
             6:{1:'Moderate', 2:'Moderate'    ,3:'Conservative'},
             7:{1:'Moderate', 2:'Conservative',3:'Advanced'},
             8:{1:'Moderate', 2:'Conservative',3:'Moderate'},
             9:{1:'Moderate', 2:'Conservative',3:'Conservative'}}


CLIMATE_scen = {1:{1:'SPP3', 2:'SPP5',3:'SPP5'},
                2:{1:'SPP3', 2:'SPP5',3:'SPP3'},
                3:{1:'SPP3', 2:'SPP5',3:'SPP1'},
                4:{1:'SPP3', 2:'SPP3',3:'SPP5'},
                5:{1:'SPP3', 2:'SPP3',3:'SPP3'},
                6:{1:'SPP3', 2:'SPP3',3:'SPP1'},
                7:{1:'SPP3', 2:'SPP1',3:'SPP5'},
                8:{1:'SPP3', 2:'SPP1',3:'SPP3'},
                9:{1:'SPP3', 2:'SPP1',3:'SPP1'}}
    
    
LOAD_scen ={ 1:{1:'2', 2:'3',3:'3'},
             2:{1:'2', 2:'3',3:'2'},
             3:{1:'2', 2:'3',3:'1'},
             4:{1:'2', 2:'2',3:'3'},
             5:{1:'2', 2:'2',3:'2'},
             6:{1:'2', 2:'2',3:'1'},
             7:{1:'2', 2:'1',3:'3'},
             8:{1:'2', 2:'1',3:'2'},
             9:{1:'2', 2:'1',3:'1'}}


# ### Load all the data required

# In[3]:


filename = main_path + 'aeso_downsized.xlsx'

bus_data               = pd.read_excel(filename ,sheet_name = 'bus',index_col = 'bus')
branch_data            = pd.read_excel(filename ,sheet_name = 'branch',index_col = 'branch')
existing_con           = pd.read_excel(filename ,sheet_name = 'units',index_col = 'unit')

re_zones                = pd.read_csv(main_path + '/step_7_near_points.xlsx')
re_zones2               = pd.read_csv(main_path + '/step_7_output.xlsx')
re_zones2.index        += 1

 
with open(main_path + '/P_dtr.pkl', 'rb') as f:
    F_DTR_base = pickle.load(f)
    
with open(main_path + '/POA_base.pkl', 'rb') as f:
    zt_mono_base = pickle.load(f)
    
with open(main_path + '/WIND_base.pkl', 'rb') as f:
    zt_wind_base = pickle.load(f)
    
with open(main_path + '/LOAD_base.pkl', 'rb') as f:
    load_profiles = pickle.load(f)

with open(main_path + '/Medoids.pkl', 'rb') as f:
    day_profiles = pickle.load(f)
    




# In[4]:


# ------------ Sets
Gn         = dict(existing_con.bus)  # unit to bus connection
OmegaBus   = set(bus_data.index)
OmegaG     = set(existing_con.index)  #  units
OmegaRow   = set(branch_data.index)

OmegaZw = set(re_zones2["WIND Index"].values)
OmegaZs = set(re_zones2["POA index"].values)


# ### Short Term Scenarios

# In[5]:


# ---------- Zone Data ------------

panel_efficiency = 0.22
wm2_to_pu_km2 = 1000000/100000000


Zone2Buss = {}
for i in re_zones2.index:
    Zone2Buss[re_zones2["POA index"].loc[i]] = re_zones.iloc[int(re_zones2.loc[i]["POA index"])]['bus'] + 1

re_zones2["POA bus"] = list(Zone2Buss.values())

Zone2Busw = {}
for i in re_zones2.index:
    Zone2Busw[re_zones2["WIND Index"].loc[i]] = re_zones.iloc[int(re_zones2.loc[i]["WIND Index"])]['bus'] + 1
    
re_zones2["WIND bus"] = list(Zone2Busw.values())

OmegaO = set(list(range(len(day_profiles))))

panel_efficiency = 0.22
wm2_to_pu_km2 = 1000000/100000000

zt_mono = {}

for i in zt_mono_base.keys():
    zt_mono[i] = zt_mono_base[i]*panel_efficiency*wm2_to_pu_km2


# In[6]:


# ---------- Stage Data and Parameters ---------------# 

num_stages = 3     # number of stages in total
y_per_stg = 10      # years implied every stage

OmegaState = {1,2} # set of investment stages  #original {1,2}
OmegaStg   = {2,3} # set of operational stages # original {2,3}

FOM_years  = {1:2, 2:1} # remaining stages from the day of deployment, this is to properly calculate Fixed O&M, since those are unchanging costs per year based on the installed capcaity. 


T = {1:80, 2:40, 3:0}  # Regolatory Emission Limit in Million Metric Tonnes.  

rho_d  = {0:91, 1:91, 2:91, 3:92}  # The percentage of time each operational conditiions. Since we are doing simple clustering only summer and winter both weather and demand will have the same percentage of the time.

# ------------- Bus Data ---------------- #

caplim = dict(zip(bus_data.index,bus_data.caplim)) # bus capacity installation limit in MW

VOM_SHD = 100  # $per 100MW or $1000 per MW, Source: https://www.aeso.ca/assets/Uploads/Attachment-1-Pricing-Approaches-in-Other-Jurisdictions-FINAL.pdf



# # ------------ Transmission Data ----------# 

Ln = dict(zip(branch_data.index, zip(branch_data.fbus,branch_data.tbus))) 
n0 = dict(zip(branch_data.index, branch_data.n0))
S_max = dict(zip(zip(branch_data.fbus, branch_data.tbus), branch_data["static rating"]/100))
S_max.update(dict(zip(zip(branch_data.tbus, branch_data.fbus), branch_data["static rating"]/100)))
Xik = dict(zip(branch_data.index,branch_data['X'])) # Line Series Reactance



# ### Existing Units Data Preperation

# In[7]:


# Existing 

existing_type = existing_con["type"].to_dict()

p_max = existing_con["pmax"].to_dict()
ramp  = existing_con["ramp"].to_dict()


available_costs = set(tech_costs['gas'].keys())
for i in tech_costs.keys():
    available_costs = available_costs.union(set(tech_costs[i].keys()))

existing_con = existing_con.reindex(columns = list(existing_con.columns) + list(available_costs), fill_value= None)

pd.options.mode.chained_assignment = None

for i in existing_con.index:
    for j in tech_costs.keys():
        if existing_con["type"][i] == j:
            for k in tech_costs[j].keys():
                existing_con[k].loc[i] = tech_costs[j][k].loc[0]["Moderate"]



# ### Determining Indecies of retrofittable existing units.  

# In[8]:


gas_indicies = set(existing_con[existing_con["type"]=="gas"].index)
coal_indicies = set(existing_con[existing_con["type"]=="coal"].index)
solar_indicies = set(existing_con[existing_con["type"]=="solar"].index)
wind_indicies = set(existing_con[existing_con["type"]=="wind"].index)
hydro_indicies = set(existing_con[existing_con["type"]=="hydro"].index)
biopower_indicies = set(existing_con[existing_con["type"]=="biopower"].index)


# In[9]:


# Source: https://atb.nrel.gov/electricity/2023/index

# Overnight Capital Cost (OCC) is added for the retrofit. Retrofit reduces the emission rate of coal and gas by 95% 

# Source: canada carbon price perdiction 

CO2e = {1:0.00005, 2:0.00017, 3:0.00020} # cost m$ per tonneCO2 over stages. 

######### Carbon pricing sensitivity test. 

factor = 8

CO2e= {k: v * factor for k, v in CO2e.items()}

case_name = "C02Price8"


# Source: https://www.eia.gov/tools/faqs/faq.php?id=74&t=11

Eme_gas  = 40.7  # tonnes of emissions per 100MW.
Eme_coal = 100   # tonnes of emissions per 100MW.

eme_gret = 40.7*0.05    # post retrofitting tonnes of emissions for both gas and coal based on selected technologies in tonnes of emissions per 100MW.
eme_cret = 100*0.05

EME_gas  = {}
EME_coal = {}

EME_gret = {}
EME_cret = {}

for i in CO2e:
    EME_gret[i] = eme_gret*CO2e[i]
    EME_cret[i] = eme_cret*CO2e[i]
    EME_gas[i] =  Eme_gas*CO2e[i]
    EME_coal[i] = Eme_coal*CO2e[i]

per_million = 1000000
kw_to_pu    = 1000*100   #kW to 100 MW
mw_to_pu    = 100        #MW to 100 MW


# ### Existing Technology Costs

# In[10]:


# Hydro doesn't matter because it has no variable cost as of now
# Solar and Wind will be the same as the new ones 
# Biopower is the only one that matters here 



CAP_bio     = tech_costs['biopower']['CAPEX']                  # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_bio     = tech_costs['biopower']['Fixed O&M']              # $m/100MW  output
VOM_bio     = tech_costs['biopower']['Variable O&M']
FUEL_bio    = tech_costs['biopower']['Fuel']       
HR_bio      = tech_costs['biopower']['Heat Rate']     

CAP_BIO       = {}           # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_BIO       = {}           # $m/100MW  output
VOM_BIO       = {}
FUEL_BIO      = {}
HR_BIO        = {}

for s in OmegaS:
    for y in OmegaHzn:
        CAP_BIO[s,y]   = CAP_bio[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_BIO[s,y]   = FOM_bio[TECH_scen[s][y]][y-1]* kw_to_pu / per_million             
        VOM_BIO[s,y]   = VOM_bio[TECH_scen[s][y]][y-1]* mw_to_pu / per_million
        HR_BIO[s,y]    =  HR_bio[TECH_scen[s][y]][y-1]
        FUEL_BIO[s,y]  = FUEL_bio[TECH_scen[s][y]][y-1]*mw_to_pu


# ### Coal and Gas Retrofitting Costs

# In[11]:


# ----- Coal Retrofitting

min_load_coal = 0.35

VOM_coal       = tech_costs['coal']['Variable O&M']

CAP_coal_retro = tech_costs['coal retro']['Additional OCC']
FOM_coal_retro = tech_costs['coal retro']["Fixed O&M"]
VOM_coal_retro = tech_costs['coal retro']['Variable O&M']


# ----- Gas Retfrofitting

CAP_gas_retro   = tech_costs['gas retro']['Additional OCC']
FOM_gas_retro   = tech_costs['gas retro']["Fixed O&M"]
VOM_gas_retro   = tech_costs['gas retro']['Variable O&M']



VOM_COAL = {}

CAP_CRET = {}
FOM_CRET = {}
VOM_CRET = {}
CAP_GRET = {}
FOM_GRET = {}
VOM_GRET = {}


for s in OmegaS:
    for y in OmegaHzn:
        VOM_COAL[s,y] = VOM_coal[TECH_scen[s][y]][y-1]* mw_to_pu / per_million
        
        CAP_CRET[s,y] = CAP_coal_retro[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_CRET[s,y] = FOM_coal_retro[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        VOM_CRET[s,y] = VOM_coal_retro[TECH_scen[s][y]][y-1]* mw_to_pu / per_million
        CAP_GRET[s,y] = CAP_gas_retro[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_GRET[s,y] = FOM_gas_retro[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        VOM_GRET[s,y] = VOM_gas_retro[TECH_scen[s][y]][y-1]* mw_to_pu / per_million


# ### New Gas with CCS 95% 

# In[12]:


# ------------ Gas New with retrofitting Units Data --------------# 

# Source: https://atb.nrel.gov/electricity/2023/index

CAP_gas       = tech_costs['gas']['CAPEX'] + tech_costs['gas retro']['Additional OCC']      # $/kW
FOM_gas       = tech_costs['gas retro']['Fixed O&M']   
VOM_gas       = tech_costs['gas']['Variable O&M']
HR_gas        = tech_costs['gas retro']['Heat Rate']
NOP_gas       = 1 + tech_costs['gas retro']['Net Output Penalty']


CAP_GAS       = {}
FOM_GAS       = {}             
VOM_GAS       = {}
HR_GAS        = {}
NOP_GAS       = {}


for s in OmegaS:
    for y in OmegaHzn:
        CAP_GAS[s,y]   = CAP_gas[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_GAS[s,y]   = FOM_gas[TECH_scen[s][y]][y-1]* kw_to_pu / per_million             
        VOM_GAS[s,y]   = VOM_gas[TECH_scen[s][y]][y-1]* mw_to_pu / per_million
        HR_GAS[s,y]    = HR_gas[TECH_scen[s][y]][y-1]
        NOP_GAS[s,y]   = NOP_gas[TECH_scen[s][y]][y-1]



# Source:  https://docs.wind-watch.org/US-footprints-Strata-2017.pdf

A_gas         = 70                           # km^2  
A_pu_gas      = 4.8                         # km^2 /100MW 
min_load_gas  = 0.5                         # factor



# ### SMR Unit Data

# In[13]:


# -------------SMR Unit Data ----------------# 

CAP_smr      = tech_costs['nuclear']['CAPEX']                  # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
CAP_smr["Advanced"] = CAP_smr["Moderate"]
FOM_smr      = tech_costs['nuclear']['Fixed O&M']              # $m/100MW  output
FOM_smr["Advanced"] = FOM_smr["Moderate"]
VOM_smr      = tech_costs['nuclear']['Variable O&M']
VOM_smr["Advanced"] = VOM_smr["Moderate"]
FUEL_smr      = tech_costs['nuclear']['Fuel']       
FUEL_smr["Advanced"] = FUEL_smr["Moderate"]
HR_smr      = tech_costs['nuclear']['Heat Rate']     
HR_smr["Advanced"] = HR_smr["Moderate"]

CAP_SMR       = {}           # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_SMR       = {}           # $m/100MW  output
VOM_SMR       = {}
FUEL_SMR      = {}
HR_SMR        = {}

for s in OmegaS:
    for y in OmegaHzn:
        CAP_SMR[s,y]   = CAP_smr[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_SMR[s,y]   = FOM_smr[TECH_scen[s][y]][y-1]* kw_to_pu / per_million             
        VOM_SMR[s,y]   = VOM_smr[TECH_scen[s][y]][y-1]* mw_to_pu / per_million
        HR_SMR[s,y]    =  HR_smr[TECH_scen[s][y]][y-1]
        FUEL_SMR[s,y]  = FUEL_smr[TECH_scen[s][y]][y-1]*mw_to_pu


# Source:  https://docs.wind-watch.org/US-footprints-Strata-2017.pdf

A_smr         = 70                         # km^2  
A_pu_smr      = 5                         # km^2 /100MW 
min_load_smr  = 0.2                       # factor


# ### H2 Turbine Data

# In[14]:


# ------------ H2 Turbine Data -------------# 

# source: Table 2 in https://www-sciencedirect-com.ezproxy.lib.ucalgary.ca/science/article/pii/S0306261921007261 

CAP_H2       = 132.0              # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_H2       = 1.4                # $m/100MW  output
VOM_H2       = 0.00087            # Qucik google serch average 

# Source:  https://docs.wind-watch.org/US-footprints-Strata-2017.pdf

A_h2         = 35                           # km^2  
A_pu_h2      = 4.8                         # km^2 /100MW 
min_load_h2  = 0.5                         # factor


# ### Solar Data

# In[15]:


# ------------ Solar Energy Data -------------# 

CAP_sol      = tech_costs['solar']['CAPEX']             # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_sol      = tech_costs['solar']['Fixed O&M']                   # $m/100MW  output

CUR_SOL      = 0.0004  # between SMR VOM and GAS VOM 


CAP_SOL       = {}           # $m/100MW  capacity  based on https://www.eia.gov/todayinenergy/detail.php?id=54519
FOM_SOL       = {}           # $m/100MW  output

for s in OmegaS:
    for y in OmegaHzn:
        CAP_SOL[s,y]   = CAP_sol[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_SOL[s,y]   = FOM_sol[TECH_scen[s][y]][y-1]* kw_to_pu / per_million             


    
# Source: estimates https://www.quora.com/How-much-area-of-land-does-a-100MW-solar-power-station-requier

A_solar = dict(zip(re_zones2["POA index"], re_zones2["POA area (km2)"]))  # as defined by HRDPS Data
A_pu_sol  = 1.8   # Area occupied per 100MW  of technology in km^2


# In[ ]:





# ### WIND Data

# In[16]:


# # ------------ Wind Energy Data -------------# 

CAP_wind    = tech_costs['wind']['CAPEX']
FOM_wind    = tech_costs['wind']['Fixed O&M']

CUR_WIND    = 0.0004      # between SMR VOM and GAS VOM

CAP_WIND = {}
FOM_WIND = {}

for s in OmegaS:
    for y in OmegaHzn:
        CAP_WIND[s,y]   = CAP_wind[TECH_scen[s][y]][y-1]* kw_to_pu / per_million
        FOM_WIND[s,y]   = FOM_wind[TECH_scen[s][y]][y-1]* kw_to_pu / per_million             

# # Source: This heavily depends on the configuration and turbine tech used and size, estimated from https://www.nrel.gov/docs/fy09osti/45834.pdf showing 

A_wind  = dict(zip(re_zones2["WIND Index"], re_zones2["WIND area (km2)"]))  # as defined by HRDPS Data
A_pu_wind = 2  # Area occupied per 100MW  of technology in km^2

# Existing technologies


# ### Pumped Hydro 

# In[17]:


# #-----------Pumped Hydro Parameters

# # Source: https://atb.nrel.gov/electricity/2023/index

CAP_pump    = tech_costs['pump']['OCC']
FOM_pump    = tech_costs['pump']['Fixed O&M']

VOM_PUMP    = 0

CAP_PUMP = {}
FOM_PUMP = {}

for s in OmegaS:
    for y in OmegaHzn:
        CAP_PUMP[s,y]   = CAP_pump[TECH_scen[s][y]][y-1]*1.264* kw_to_pu / per_million  # The 1.264 is the OCC to CAPEX ratio form other tech
        FOM_PUMP[s,y]   = FOM_pump[TECH_scen[s][y]][y-1]* kw_to_pu / per_million  

W_max  = 0.4 # Original 0.4    
sigmaT = 0.9 # original 0.9
sigmaP = 1.1  #original 1.1
VU_0 = 6 # Original 6
VU_min = 0
VU_max = 6 # Original 6
VL_0 = 0
VL_min = 0
VL_max = 6 # Original 6


# ### Battery Data

# In[18]:


#------------ Battery parameters

# Source: 4-hour deafult battery,  https://atb.nrel.gov/electricity/2023/index


CAP_batt    = tech_costs['battery']['OCC']
FOM_batt    = tech_costs['battery']["Fixed O&M"]

VOM_batt    = 0

CAP_BATT = {}
FOM_BATT = {}


for s in OmegaS:
    for y in OmegaHzn:
        CAP_BATT[s,y]   = CAP_batt[TECH_scen[s][y]][y-1]*1.264* kw_to_pu / per_million  # The 1.264 is the OCC to CAPEX ratio form other tech
        FOM_BATT[s,y]   = FOM_batt[TECH_scen[s][y]][y-1]* kw_to_pu / per_million  
        

CHmax  = 1
CHmin  = 0
DImax  = 1
DImin  = 0
EtaCh  = 0.9
EtaDi  = 1.1

SOCini = 0
SOCmax = 4

MaxDoD = 0.8   

SOCmin = (1 - MaxDoD)

γ_batt = 1 # desired operational period in years
γ_bshelf = 20 # Idle shelflife 
ϵ_batt = 0.70 # SOC max persetnage at which EOL is considered reached
δ_batt = (1 - ϵ_batt)/γ_batt   # yearly allowed degredation 
δ_bshelf =  1/ (γ_bshelf*8760) # hourly degredation 


# ### Transmission Line Data

# In[19]:


# ---------- New Transmission Line ------------

# Data here and sources can be found in wire_data.xlsx

# Cost data is Exploratory costs from  Transmission Cost Estimation Guide For MTEP: https://cdn.misoenergy.org/20220208%20PSC%20Item%2005c%20Transmission%20Cost%20Estimation%20Guide%20for%20MTEP22_Draft622733.pdf

EXP_cost = branch_data['Cost per km ($)']/1000000 # conductor cost in $m/km
FOM_LINE =  (0.03*EXP_cost).to_dict()  # $FOM per km per 100MW rating
CAP_LINE = branch_data["Total Cost ($mil)"].to_dict()

 


# ### DTR metering system 

# In[20]:


# ----------- DTR metering system -------------

CAP_DTR = {}
FOM_DTR = {}

DTR_span = 100  #this is the distance between dtr devices. 
CAP_dtr = 0.0033 # $m per device 

CAP_DTR = np.floor(branch_data.distance/100)*CAP_dtr

FOM_dtr_per_month = 0.000178

FOM_DTR = FOM_dtr_per_month*12


# ### SSSC Facts Data 

# In[21]:


# ----------- SSSC Facts system ---------------

CAP_SSSC = 5     #arbitrary
FOM_SSSC = 0.005  #arbitrary

V = 0.03    # max voltage capability of sssc device in p.u. (with respect to the p.u. voltage of system) 
C = 0.49   # Cut-in limit of SSSC device. 
M_f = 10   # Cut-in Big M  (set to maximum number of possible SSSC on a line) 

# ## Short Term Scenarios

# In[22]:


# (n,s,y,o)

LOAD = {}

for n in OmegaBus:
    for s in OmegaS:
        for y in OmegaHzn:
            for o in OmegaO: 
                if y == 1:
                    LOAD[n,o,y,s] = np.array(load_profiles[n,o])/100
                else:
                    LOAD[n,o,y,s] = np.array(load_profiles[n,o])*load_scen[LOAD_scen[s][y]][y]/100

zt_wind = {}

wind_scen = climate_scen['wind speed']

for z in OmegaZw:
    for s in OmegaS:
        for y in OmegaHzn:
            for o in OmegaO:
                if y ==1:
                    zt_wind[z,o,y,s] = np.array(zt_wind_base[z,o])
                else:
                    zt_wind[z,o,y,s] = np.array(zt_wind_base[z,o])*wind_scen[CLIMATE_scen[s][y]][y-1]


F_DTR = {}

dtr_scen = climate_scen['dtr']

for l in OmegaRow:
    for s in OmegaS:
        for y in OmegaHzn:
            for o in OmegaO:
                if y ==1:
                    F_DTR[l,o,y,s] = np.array(F_DTR_base[l,o])
                else:
                    F_DTR[l,o,y,s] = np.array(F_DTR_base[l,o])*dtr_scen[CLIMATE_scen[s][y]][y-1]





# In[23]:


# --------------- Economic Data & Objective Function Parameters---------------- # 

# Source: https://atb.nrel.gov/electricity/2023/index

NPV = {}
d_rate = 0.06

for y in OmegaState:
    NPV[y] = 1/(1+d_rate)**y

#--------------- Linarization Parameters        
            
MTrack = 500
MSSC   = 2  # < This should be the capacity of the line
        
L_f = -2  # lower bound of line flow
U_f = 2 # upper bound of line flow
 
M_sssc = max(-L_f,U_f) 



### Adjustment to the original results.  

OmegaRet = gas_indicies.union(coal_indicies)
OmegaVre = solar_indicies.union(wind_indicies)
inflow = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


#### Reserved for Different Horizon Tests: 

# def remap_idx2(d):
#     ys = sorted({k[1] for k in d})
#     ymap = {y: i+1 for i, y in enumerate(ys)}
#     new_d = {(k[0], ymap[k[1]]): v for k, v in d.items()}
#     return new_d

# def remap_y_in_4d(d):
#     ys = sorted({k[2] for k in d})
#     ymap = {y: i + 1 for i, y in enumerate(ys)}
#     new_d = {(k[0], k[1], ymap[k[2]], k[3]): v for k, v in d.items()}
#     return new_d


# CAP_GAS = remap_idx2(CAP_GAS)
# FOM_GAS = remap_idx2(FOM_GAS)
# CAP_SMR = remap_idx2(CAP_SMR)
# FOM_SMR = remap_idx2(FOM_SMR)
# CAP_PUMP = remap_idx2(CAP_PUMP)
# FOM_PUMP = remap_idx2(FOM_PUMP)
# CAP_BATT = remap_idx2(CAP_BATT)
# FOM_BATT = remap_idx2(FOM_BATT)
# CAP_WIND = remap_idx2(CAP_WIND)
# FOM_WIND = remap_idx2(FOM_WIND)
# CAP_SOL = remap_idx2(CAP_SOL)
# FOM_SOL = remap_idx2(FOM_SOL)
# CAP_GRET = remap_idx2(CAP_GRET)
# FOM_GRET = remap_idx2(FOM_GRET)
# CAP_CRET = remap_idx2(CAP_CRET)
# FOM_CRET = remap_idx2(FOM_CRET)
# VOM_GAS = remap_idx2(VOM_GAS)
# VOM_GRET = remap_idx2(VOM_GRET)
# VOM_COAL = remap_idx2(VOM_COAL)
# VOM_BIO = remap_idx2(VOM_BIO)
# VOM_SMR = remap_idx2(VOM_SMR)
# VOM_CRET = remap_idx2(VOM_CRET)

# LOAD = remap_y_in_4d(LOAD)
# zt_wind = remap_y_in_4d(zt_wind)
# F_DTR = remap_y_in_4d(F_DTR)



# CO2e = {1:0.0007, 2:0.00010, 3:0.00015} # 10
# CO2e = {1:0.00015, 2:0.00025, 3:0.00035} # 30
# CO2e = {1:0.00035, 2:0.00040, 3:0.0005} # 40

