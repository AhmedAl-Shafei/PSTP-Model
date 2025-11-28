from  pyomo.environ import *
from pyomo.opt import SolverFactory
import time
import pandas as pd
import numpy as np

from model_data import *

case_name = "all-factors"

m = ConcreteModel()

# Investment Variables

# ------- Sets : I realized it is important  

m.i_gas     = Var(OmegaBus, OmegaState, OmegaS, within = NonNegativeReals) 
m.i_smr     = Var(OmegaBus, OmegaState, OmegaS, within = NonNegativeReals)
m.i_h2      = Var(OmegaBus, OmegaState, OmegaS, within = NonNegativeReals)
m.i_solar   = Var(OmegaZs, OmegaState, OmegaS,  within = NonNegativeReals)
m.i_wind    = Var(OmegaZw, OmegaState,  OmegaS, within = NonNegativeReals)

m.x_pump    = Var(OmegaBus, OmegaState, OmegaS, within = Binary)
m.x_batt    = Var(OmegaBus, OmegaState, OmegaS, within = Binary) 

m.x_line    = Var(OmegaRow, OmegaState, OmegaS, within = Binary)
m.x_dtr     = Var(OmegaRow, OmegaState, OmegaS, within = Binary)
m.x_com     = Var(OmegaRow, OmegaState, OmegaS, within = Binary)
m.x_sssc    = Var(OmegaRow, OmegaState, OmegaS, within = NonNegativeIntegers)
m.x_retro   = Var(OmegaRet, OmegaState, OmegaS, within = Binary)


m.p_exist   = Var(OmegaG, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_gas     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_smr     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_h2      = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_ccs     = Var(OmegaRet, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.p_solar   = Var(OmegaZs, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_wind    = Var(OmegaZw, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_load    = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.theta     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS,OmegaO, within = Reals)
m.f         = Var(OmegaRow, OmegaT, OmegaStg, OmegaS,OmegaO, within = Reals)

# SSSC injection variable 
m.X_inj  = Var(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, within = Reals)

# hydropump Storage

m.p_pump    = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_turb    = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.w_pump    = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.w_turb    = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.r_up      = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.r_low     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.h_spill_up     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals) 
m.h_spill_dn     = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)


m.x_phase   = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary) # Hydro state of charge

# Battery variables 

m.p_ch      = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)
m.p_di      = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.x_state   = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary)   # Battery state of charge

m.s         = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)

m.Dcy       = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = NonNegativeReals)


# Linearlization variables
m.z = Var(OmegaRow, OmegaT, OmegaStg, OmegaS,OmegaO, within = Binary)
m.w = Var(OmegaRow, OmegaT, OmegaStg, OmegaS,OmegaO, within = Reals)

m.u  = Var(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary)
m.u_1 = Var(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary)
m.u_2 = Var(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary)




m.X_investment =  sum(Scenario_Prob[s]*sum(\
                + sum((CAP_GAS[s,y] + FOM_GAS[s,y]) *m.i_gas[n,y,s]  for n in OmegaBus)\
                + sum((CAP_SMR[s,y] + FOM_SMR[s,y]) *m.i_smr[n,y,s]  for n in OmegaBus)\
                + sum((CAP_H2  + FOM_H2) *m.i_h2[n,y,s]  for n in OmegaBus)\
                + sum((CAP_PUMP[s,y] + FOM_PUMP[s,y])*m.x_pump[n,y,s] for n in OmegaBus)\
                + sum((CAP_BATT[s,y] + FOM_BATT[s,y])*m.x_batt[n,y,s] for n in OmegaBus)\
                + sum((CAP_WIND[s,y] + FOM_WIND[s,y])*m.i_wind[z,y,s]  for z in OmegaZw)\
                + sum((CAP_SOL[s,y] + FOM_SOL[s,y]) *m.i_solar[z,y,s] for z in OmegaZs)\
                + sum((CAP_LINE[l] + FOM_LINE[y]) *m.x_line[l,y,s]\
                + (CAP_DTR[l] + FOM_DTR) *m.x_dtr[l,y,s]\
                + (CAP_SSSC+ FOM_SSSC)*m.x_sssc[l,y,s]for l in OmegaRow)\
                + sum((CAP_GRET[s,y]+ FOM_GRET[s,y])*m.x_retro[g,y,s] for g in gas_indicies)
                + sum((CAP_CRET[s,y]+ FOM_CRET[s,y])*m.x_retro[g,y,s] for g in coal_indicies)
                       for y in OmegaState) for s in OmegaS)

m.X_operation  =     sum(Scenario_Prob[s]*sum(sum(rho_d[o]*sum(\
                     sum((EME_gas[y] + VOM_GAS[s,y])*m.p_exist[g,t,y,s,o] 
                       + (EME_gret[y]+ VOM_GRET[s,y])*m.p_ccs[g,t,y,s,o] for g in gas_indicies)\
                    + sum((EME_coal[y] + VOM_COAL[s,y])*m.p_exist[g,t,y,s,o]\
                       + (EME_cret[y] + VOM_CRET[s,y])*m.p_ccs[g,t,y,s,o] for g in coal_indicies)\
                    + sum(VOM_BIO[s,y]*m.p_exist[g,t,y,s,o] for g in biopower_indicies)\
                    + sum(CUR_WIND*m.p_exist[g,t,y,s,o] for g in wind_indicies)\
                    + sum(CUR_SOL*m.p_exist[g,t,y,s,o] for g in solar_indicies)\
                   + sum(VOM_GAS[s,y]*m.p_gas[n,t,y,s,o]\
                       + VOM_SMR[s,y]*m.p_smr[n,t,y,s,o]\
                       + VOM_H2  *m.p_h2[n,t,y,s,o]\
                       + VOM_SHD *m.p_load[n,t,y,s,o] for n in OmegaBus) \
                   + sum(CUR_WIND*m.p_wind[z,t,y,s,o] for z in OmegaZw)\
                   + sum(CUR_SOL *m.p_solar[z,t,y,s,o] for z in OmegaZs)\
                      for t in OmegaT)\
                      for o in OmegaO)\
                      for y in OmegaStg) for s in OmegaS)

m.obj = Objective(expr = m.X_investment + m.X_operation, sense = minimize)




# ---------- Investment Constraints 


# 1) 

def pumpinv(m,n,s):
    return 1 >= sum(m.x_pump[n,y,s] for y in OmegaState)
m.pumpinv = Constraint(OmegaBus, OmegaS, rule = pumpinv)

def battinv(m,n,s):
    return 1 >= sum(m.x_batt[n,y,s] for y in OmegaState)
m.battinv = Constraint(OmegaBus, OmegaS, rule = battinv)

def lineinv(m,l,s):
    return 1 >= sum(m.x_line[l,y,s] for y in OmegaState)
m.lineinv = Constraint(OmegaRow, OmegaS, rule = lineinv)

def retroonce(m,g,s):
    return 1 >= sum(m.x_retro[g,y,s] for y in OmegaState)
m.retroonce = Constraint(OmegaRet, OmegaS, rule = retroonce)

def donutinv(m,l,s):
    return 1 >= sum(m.x_dtr[l,y,s] for y in OmegaState)
m.donutinv = Constraint(OmegaRow,OmegaS, rule = donutinv)



# 2) 

def nolinenodtr(m,l,y,s):
    return m.x_dtr[l,y,s] <= sum(m.x_line[l,τ-1,s] for τ in range(2,y+1)) + n0[l]
m.nolinenodtr = Constraint(OmegaRow, OmegaState, OmegaS, rule = nolinenodtr)

def nolinenosssc(m,l,y,s):
    return m.x_sssc[l,y,s] <= (sum(m.x_line[l,τ-1,s] for τ in range(2,y+1)) + n0[l])*M_f
m.nolinenosssc = Constraint(OmegaRow, OmegaState, OmegaS, rule = nolinenosssc)

# 3)

def gas_area(m,n,s):
    return sum(m.i_gas[n,y,s] for y in OmegaState)*A_pu_gas  <= A_gas
m.gas_area = Constraint(OmegaBus, OmegaS, rule = gas_area)

def smr_area(m,n,s):
    return sum(m.i_smr[n,y,s]*A_pu_smr for y in OmegaState) <= A_smr
m.smr_area = Constraint(OmegaBus, OmegaS, rule = smr_area)

def h2_area(m,n,s):
    return sum(m.i_h2[n,y,s]*A_pu_h2 for y in OmegaState) <= A_h2
m.h2_area = Constraint(OmegaBus, OmegaS, rule = h2_area)

def solar_area(m,z,s):
    return sum(m.i_solar[z,y,s]*A_pu_sol for y in OmegaState)  <= A_solar[z]
m.solar_area = Constraint(OmegaZs, OmegaS,  rule = solar_area)

def wind_area(m,z,s):
    return sum(m.i_wind[z,y,s]*A_pu_wind for y in OmegaState) <= A_wind[z]
m.wind_area = Constraint(OmegaZw, OmegaS, rule = wind_area)


# ------------ Operational Constraints 


# 1) 

def exist_max_output(m,g,t,y,s,o):
          return m.p_exist[g,t,y,s,o] <= existing_con["pmax"][g]
m.exist_max_output = Constraint(OmegaG - OmegaRet - OmegaVre, OmegaT, OmegaStg, OmegaS, OmegaO, rule = exist_max_output)

def emitters_max_output(m,g,t,y,s,o):
          return m.p_exist[g,t,y,s,o] <= existing_con["pmax"][g]*(1 - sum(m.x_retro[g,τ-1,s] for τ in range(2, y+1)))
m.emitters_max_output = Constraint(OmegaRet, OmegaT, OmegaStg, OmegaS, OmegaO, rule = emitters_max_output)

def retro_max_output(m,g,t,y,s,o):
          return m.p_ccs[g,t,y,s,o] <= p_max[g]*sum(m.x_retro[g,τ-1,s] for τ in range(2, y+1))
m.retro_max_output = Constraint(OmegaRet, OmegaT, OmegaStg, OmegaS, OmegaO, rule = retro_max_output)

def gas_max_output(m,n,t,y,s,o): 
    return m.p_gas[n,t,y,s,o] <= sum(m.i_gas[n,τ-1,s] for τ in range(2, y+1))
m.gas_max_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = gas_max_output)

def gas_min_output(m,n,t,y,s,o): 
    return min_load_gas*sum(m.i_gas[n,τ-1,s] for τ in range(2, y+1)) <= m.p_gas[n,t,y,s,o]
m.gas_min_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = gas_min_output)


def smr_max_output(m,n,t,y,s,o): 
    return m.p_smr[n,t,y,s,o] <= sum(m.i_smr[n,τ-1,s] for τ in range(2, y+1))
m.smr_max_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = smr_max_output)

def smr_min_output(m,n,t,y,s,o): 
    return min_load_smr*sum(m.i_smr[n,τ-1,s] for τ in range(2, y+1)) <= m.p_smr[n,t,y,s,o]
m.smr_min_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = smr_min_output)

def h2_max_output(m,n,t,y,s,o): 
    return m.p_h2[n,t,y,s,o] <= sum(m.i_h2[n,τ-1,s] for τ in range(2, y+1))
m.h2_max_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = h2_max_output)

def h2_min_output(m,n,t,y,s,o): 
    return min_load_h2*sum(m.i_h2[n,τ-1,s] for τ in range(2, y+1)) <= m.p_h2[n,t,y,s,o]
m.h2_min_output = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = h2_min_output)


# 2) 

def target_penetration(m,y,s,o):
    return   sum(sum(EME_gas[y]*m.p_exist[g,t,y,s,o] + EME_gret[y]*m.p_ccs[g,t,y,s,o]  for g in gas_indicies)\
            +sum(EME_coal[y]*m.p_exist[g,t,y,s,o] + EME_cret[y]*m.p_ccs[g,t,y,s,o]  for g in coal_indicies)\
             for t in OmegaT) <= T[y]*y_per_stg
m.target_penetration = Constraint(OmegaStg, OmegaS, OmegaO, rule = target_penetration) 


# 3) 

def load_shedding(m,n,t,y,s,o):
    return m.p_load[n,t,y,s,o] <= LOAD[n,o,y,s][t]
m.load_shedding = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO,  rule = load_shedding)


# 4)

def wind_curtailment(m,z,t,y,s,o):
    return m.p_wind[z,t,y,s,o] <= zt_wind[z,o,y,s][t]*(sum(m.i_wind[z,τ-1,s]  for τ in range(2,y+1)))
m.wind_curtailment = Constraint(OmegaZw, OmegaT, OmegaStg, OmegaS, OmegaO,  rule = wind_curtailment)


def solar_curtailment(m,z,t,y,s,o):
    return m.p_solar[z,t,y,s,o] <=  zt_mono[z,o][t]*(sum(m.i_solar[z,τ-1,s] for τ in range(2,y+1))) 
m.solar_curtailment = Constraint(OmegaZs, OmegaT, OmegaStg, OmegaS, OmegaO,  rule = solar_curtailment)


def windex_curtailment(m,g,t,y,s,o):
    return m.p_exist[g,t,y,s,o] <= existing_con["pmax"][g]*zt_wind[38293,o,y,s][t]
m.windex_curtailment = Constraint(wind_indicies, OmegaT, OmegaStg, OmegaS, OmegaO,  rule = windex_curtailment)


def solarex_curtailment(m,g,t,y,s,o):
    return m.p_exist[g,t,y,s,o] <=  existing_con["pmax"][g]*zt_mono[37685,o][t]
m.solarex_curtailment = Constraint(solar_indicies, OmegaT, OmegaStg, OmegaS, OmegaO,  rule = solarex_curtailment)

# 5) 

def coal_rampup(m,g,t,y,s,o):
          return m.p_exist[g,t,y,s,o] - m.p_exist[g,t-1,y,s,o] <= existing_con["ramp"][g]
m.coal_rampup = Constraint(OmegaRet, OmegaT-{0}, OmegaStg, OmegaS, OmegaO, rule = coal_rampup)

def coal_rampdown(m,g,t,y,s,o):
          return m.p_exist[g,t,y,s,o] - m.p_exist[g,t-1,y,s,o] >= - existing_con["ramp"][g]
m.coal_rampdown = Constraint(OmegaRet, OmegaT-{0}, OmegaStg, OmegaS, OmegaO, rule = coal_rampdown)

def rampuppccs(m,g,t,y,s,o):
    return m.p_ccs[g,t,y,s,o] - m.p_ccs[g,t-1,y,s,o] <= existing_con["ramp"][g]
m.rampuppccs = Constraint(OmegaRet, OmegaT-{0},OmegaStg, OmegaS,OmegaO, rule = rampuppccs)

def rampdownpccs(m,g,t,y,s,o):
    return m.p_ccs[g,t,y,s,o] - m.p_ccs[g,t-1,y,s,o] >= - existing_con["ramp"][g]
m.rampdownpccs = Constraint(OmegaRet, OmegaT-{0},OmegaStg, OmegaS, OmegaO, rule = rampdownpccs)


# 6) 

def nodbal_rule(m,n,t,y,s,o):
    return sum(m.f[l,t,y,s,o] for l in OmegaRow if Ln[l][1] == n) - sum(m.f[l,t,y,s,o] for l in OmegaRow if Ln[l][0] == n) \
         + sum(zt_wind[z,o,y,s][t]*(sum(m.i_wind[z,τ-1,s] for τ in range(2,y+1))) - m.p_wind[z,t,y,s,o] for z in OmegaZw if Zone2Busw[z] == n) \
         + sum(zt_mono[z,o][t]*(sum(m.i_solar[z,τ-1,s] for τ in range(2,y+1))) - m.p_solar[z,t,y,s,o] for z in OmegaZs if Zone2Buss[z] == n)\
         + sum(zt_wind[38293,o,y,s][t]*existing_con["pmax"][g] - m.p_exist[g,t,y,s,o] for g in wind_indicies if Gn[g] == n)\
         + sum(zt_mono[37685,o][t]*existing_con["pmax"][g] - m.p_exist[g,t,y,s,o] for g in solar_indicies if Gn[g] == n)\
         + sum(m.p_exist[g,t,y,s,o]  + m.p_ccs[g,t,y,s,o] for g in OmegaRet if Gn[g] == n)\
         + sum(m.p_exist[g,t,y,s,o] for g in OmegaG-OmegaRet-OmegaVre if Gn[g] == n)\
         + m.p_gas[n,t,y,s,o] + m.p_smr[n,t,y,s,o] + m.p_h2[n,t,y,s,o]  + m.p_load[n,t,y,s,o]\
         + m.p_turb[n,t,y,s,o] - m.p_pump[n,t,y,s,o] + m.p_di[n,t,y,s,o] - m.p_ch[n,t,y,s,o]\
         - LOAD[n,o,y,s][t] == 0 
m.nodbal = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = nodbal_rule)

# 7)


def thetalim_rule(m,n,t,y,s,o):  
          return m.theta[n,t,y,s,o] <= 0.3 
m.thetalim = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = thetalim_rule)

def thetalim2_rule(m,n,t,y,s,o):  
          return m.theta[n,t,y,s,o] >= -0.3
m.thetalim2 = Constraint(OmegaBus,OmegaT, OmegaStg, OmegaS, OmegaO, rule = thetalim2_rule)



# 8)

def flow_rule(m,l,t,y,s,o):
          return m.f[l,t,y,s,o] == (1/Xik[l])*(m.theta[Ln[l][0],t,y,s,o] - m.theta[Ln[l][1],t,y,s,o]) + m.X_inj[l,t,y,s,o]
m.flow = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = flow_rule)


# # 9) 


def soc_rule(m,n,t,y,s,o):
        return m.s[n,t+1,y,s,o] == m.s[n,t,y,s,o] + EtaCh*m.p_ch[n,t,y,s,o] - EtaDi*m.p_di[n,t,y,s,o]
m.soc = Constraint(OmegaBus, OmegaT-{23}, OmegaStg, OmegaS, OmegaO, rule = soc_rule) 

def SOCmax_rule(m,n,t,y,s,o):
    return m.s[n,t,y,s,o] <= SOCmax
m.SOCmax = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = SOCmax_rule) 

def SOCmin_rule(m,n,t,y,s,o):
    return m.s[n,t,y,s,o] >= SOCmin
m.SOCmin = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = SOCmin_rule) 

def max_di(m,n,t,y,s,o):
    return m.p_di[n,t,y,s,o] <= DImax*sum(m.x_batt[n,τ-1,s] for τ in range(2, y+1))
m.max_di = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_di)

def max_ch(m,n,t,y,s,o):
    return m.p_ch[n,t,y,s,o] <= CHmax*sum(m.x_batt[n,τ-1,s] for τ in range(2, y+1))
m.max_ch = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO,rule = max_ch)

def maxbat_linear1(m,n,t,y,s,o):
    return m.p_di[n,t,y,s,o] <= DImax*m.x_state[n,t,y,s,o]
m.maxbat_linear1 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO,rule = maxbat_linear1)

def maxbat_linear2(m,n,t,y,s,o):
    return m.p_ch[n,t,y,s,o] <= CHmax*(1 - m.x_state[n,t,y,s,o])
m.maxbat_linear2 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO,rule = maxbat_linear2)


def chstop(m,n,y,s,o):
    return m.p_ch[n,23,y,s,o] <= 0
m.chstop = Constraint(OmegaBus, OmegaStg, OmegaS, OmegaO, rule = chstop)


## Addition of Degradation 

def degcurv1(m,n,t,y,s,o): 
    return m.Dcy[n,t,y,s,o] >=  - 0.00102*m.s[n,t,y,s,o] + 0.00051
m.degcurv1 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = degcurv1)

def degcurv2(m,n,t,y,s,o): 
    return m.Dcy[n,t,y,s,o] >=  - 0.000151*m.s[n,t,y,s,o] + 0.0001502
m.degcurv2 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = degcurv2)

def deglim(m,n,y,s,o):
    return sum(m.Dcy[n,t,y,s,o] + δ_bshelf for t in OmegaT)   <= δ_batt
m.deglim = Constraint(OmegaBus, OmegaStg, OmegaS, OmegaO, rule = deglim)

## Addition of lifetime 

def battlife1(m,n,y,s):
    return sum(m.x_batt[n,τ,s] for τ in range(1,y+1)) <= 1
m.battlife1 = Constraint(OmegaBus, list(OmegaState)[:-γ_batt], OmegaS, rule = battlife1)


def battlife2(m,n,y,s):
    return sum(m.x_batt[n,τ,s] for τ in range(1,y+1)) <= 1
m.battlife2 = Constraint(OmegaBus, list(OmegaState)[2-γ_batt:], OmegaS, rule = battlife2)



# 10) 


def pumpstop(m,n,y,s,o):
    return m.w_pump[n,23,y,s,o] <= 0
m.pumpstop = Constraint(OmegaBus, OmegaStg, OmegaS, OmegaO, rule = pumpstop)


def turb_rule(m,n,t,y,s,o):
    return m.p_turb[n,t,y,s,o] == sigmaT*m.w_turb[n,t,y,s,o]
m.turb = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = turb_rule) 

def pump_rule(m,n,t,y,s,o):
    return m.p_pump[n,t,y,s,o] == sigmaP*m.w_pump[n,t,y,s,o]
m.pump = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = pump_rule) 


def volUini_rule(m,n,y,s,o):
    return m.r_up[n,1,y,s,o] == VU_0
m.volUini = Constraint(OmegaBus, OmegaStg, OmegaS, OmegaO, rule = volUini_rule) 

def volLini_rule(m,n,y,s,o):
    return m.r_low[n,1,y,s,o] == VL_0
m.volLini = Constraint(OmegaBus, OmegaStg, OmegaS, OmegaO, rule = volLini_rule)


def volUmax_rule(m,n,t,y,s,o):
    return m.r_up[n,t,y,s,o] <= VU_max
m.volUmax = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = volUmax_rule) 

def volLmax_rule(m,n,t,y,s,o):
    return m.r_low[n,t,y,s,o] >= VL_min
m.volLmax = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = volLmax_rule)

def volU_rule(m,n,t,y,s,o):
        return m.r_up[n,t+1,y,s,o] == m.r_up[n,t,y,s,o] + m.w_pump[n,t,y,s,o] - m.w_turb[n,t,y,s,o] + inflow[t] - m.h_spill_up[n,t,y,s,o]
m.volU = Constraint(OmegaBus, OmegaT-{23}, OmegaStg, OmegaS,  OmegaO, rule = volU_rule) 

def volL_rule(m,n,t,y,s,o):
        return m.r_low[n,t+1,y,s,o] == m.r_low[n,t,y,s,o] + m.w_turb[n,t,y,s,o] - m.w_pump[n,t,y,s,o] + inflow[t] - m.h_spill_dn[n,t,y,s,o]
m.volL = Constraint(OmegaBus, OmegaT-{23}, OmegaStg, OmegaS,  OmegaO, rule = volL_rule) 


m.zp = Var(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, within = Binary)

def max_turb(m,n,t,y,s,o):
    return m.w_turb[n,t,y,s,o] <= W_max*m.zp[n,t,y,s,o]
m.max_turb = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_turb)

def max_pump(m,n,t,y,s,o):
    return m.w_pump[n,t,y,s,o] <= W_max*sum(m.x_pump[n,τ-1,s] for τ in range(2, y+1)) - W_max*m.zp[n,t,y,s,o]
m.max_pump = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_pump)

def max_linear1(m,n,t,y,s,o):
    return m.zp[n,t,y,s,o] <= sum(m.x_pump[n,τ-1,s] for τ in range(2, y+1))
m.max_linear1 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_linear1)

def max_linear2(m,n,t,y,s,o):
    return m.zp[n,t,y,s,o] <= m.x_phase[n,t,y,s,o]
m.max_linear2 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_linear2)

def max_linear3(m,n,t,y,s,o):
    return m.zp[n,t,y,s,o] >= sum(m.x_pump[n,τ-1,s] for τ in range(2, y+1)) + m.x_phase[n,t,y,s,o] - 1
m.max_linear3 = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = max_linear3)




# #  11) 

m.anti_gas       = ConstraintList()
m.anti_smr       = ConstraintList()
m.anti_h2        = ConstraintList()
m.anti_solar     = ConstraintList()
m.anti_wind      = ConstraintList()
m.anti_batt      = ConstraintList()
m.anti_pump      = ConstraintList()
m.anti_line      = ConstraintList()
m.anti_retro     = ConstraintList()
m.anti_dtr       = ConstraintList()
m.anti_sssc      = ConstraintList()

for i in ind:
    for n in OmegaBus:
        m.anti_gas.add(m.i_gas[n,ind[i][0]] == m.i_gas[n,ind[i][1]])
        m.anti_smr.add(m.i_smr[n,ind[i][0]] == m.i_smr[n,ind[i][1]])
        m.anti_h2.add(m.i_h2[n,ind[i][0]]   == m.i_h2[n,ind[i][1]])
        m.anti_batt.add(m.x_batt[n,ind[i][0]]   == m.x_batt[n,ind[i][1]])
        m.anti_pump.add(m.x_pump[n,ind[i][0]]   == m.x_pump[n,ind[i][1]])

    for z in OmegaZs:
        m.anti_solar.add(m.i_solar[z,ind[i][0]] == m.i_solar[z,ind[i][1]])
    for z in OmegaZw: 
        m.anti_wind.add(m.i_wind[z,ind[i][0]] == m.i_wind[z,ind[i][1]])
    for l in OmegaRow:
        m.anti_line.add(m.x_line[l,ind[i][0]] == m.x_line[l,ind[i][1]])
        m.anti_dtr.add(m.x_dtr[l,ind[i][0]] == m.x_dtr[l,ind[i][1]])
        m.anti_sssc.add(m.x_sssc[l,ind[i][0]] == m.x_sssc[l,ind[i][1]])
    for g in OmegaRet:
        m.anti_retro.add(m.x_retro[g,ind[i][0]] == m.x_retro[g,ind[i][1]])


# # 12) 

def cap1_rule(m,l,t,y,s,o):
          return m.f[l,t,y,s,o]  - S_max[Ln[l]]*(sum(m.x_line[l,τ-1,s] for τ in range(2,y+1)) + sum(m.x_com[l,τ-1,s] for τ in range(2,y+1))+ n0[l] - sum(m.x_dtr[l,τ-1,s]*n0[l] for τ in range(2,y+1))) \
          - F_DTR[l,o,y,s][t]*(sum(m.x_com[l,τ-1,s] for τ in range(2,y+1)) + n0[l]*sum(m.x_dtr[l,τ-1,s] for τ in range(2,y+1))) <= 0
m.cap1 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cap1_rule)
          
def cap2_rule(m,l,t,y,s,o):
          return - m.f[l,t,y,s,o]  - S_max[Ln[l]]*(sum(m.x_line[l,τ-1,s] for τ in range(2,y+1)) + sum(m.x_com[l,τ-1,s] for τ in range(2,y+1))+ n0[l] - sum(m.x_dtr[l,τ-1,s]*n0[l] for τ in range(2,y+1))) \
          - F_DTR[l,o,y,s][t]*(sum(m.x_com[l,τ-1,s] for τ in range(2,y+1)) + n0[l]*sum(m.x_dtr[l,τ-1,s] for τ in range(2,y+1))) <= 0
m.cap2 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cap2_rule)


def caplin1(m,l,y,s):
    return m.x_com[l,y,s] <= m.x_line[l,y,s]
m.caplin1 = Constraint(OmegaRow, OmegaState, OmegaS, rule = caplin1)

def caplin2(m,l,y,s):
    return m.x_com[l,y,s] <= m.x_dtr[l,y,s]
m.caplin2 = Constraint(OmegaRow, OmegaState, OmegaS, rule = caplin2)

def caplin3(m,l,y,s):
    return m.x_com[l,y,s] >= m.x_line[l,y,s] + m.x_dtr[l,y,s] - 1
m.caplin3 = Constraint(OmegaRow, OmegaState, OmegaS, rule = caplin3)



# # 13) 

def sssc_linear1(m,l,t,y,s,o):
    return - sum(m.x_sssc[l,τ-1,s] for τ in range(2,y+1))*V*abs(1/Xik[l]) <= m.X_inj[l,t,y,s,o]
m.sssc_linear1 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = sssc_linear1)

def sssc_linear2(m,l,t,y,s,o):
    return  sum(m.x_sssc[l,τ-1,s] for τ in range(2,y+1))*V*abs(1/Xik[l]) >= m.X_inj[l,t,y,s,o]
m.sssc_linear2 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = sssc_linear2)


def cutin_sssc1a(m,l,t,y,s,o):
    return -M_f*V*abs(1/Xik[l])*(m.u_1[l,t,y,s,o] + m.u_2[l,t,y,s,o]) <= m.X_inj[l,t,y,s,o]
m.cutin_sssc1a = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc1a)

def cutin_sssc1b(m,l,t,y,s,o):
    return M_f*V*abs(1/Xik[l])*(m.u_1[l,t,y,s,o] + m.u_2[l,t,y,s,o]) >= m.X_inj[l,t,y,s,o]
m.cutin_sssc1b = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc1b)

def cutin_sssc3(m,l,t,y,s,o):
    return m.f[l,t,y,s,o] >= C - M_sssc*(1 - m.u_1[l,t,y,s,o])
m.cutin_sssc3 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc3)

def cutin_sssc4(m,l,t,y,s,o):
    return m.f[l,t,y,s,o] <= C + M_sssc*m.u_1[l,t,y,s,o]
m.cutin_sssc4 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc4)

def cutin_sssc5(m,l,t,y,s,o):
    return - C >= m.f[l,t,y,s,o] - (1 - m.u_2[l,t,y,s,o]) * M_sssc
m.cutin_sssc5 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc5)

def cutin_sssc6(m,l,t,y,s,o):
    return - C <= m.f[l,t,y,s,o] + m.u_2[l,t,y,s,o]* M_sssc
m.cutin_sssc6 = Constraint(OmegaRow, OmegaT, OmegaStg, OmegaS, OmegaO, rule = cutin_sssc6)

# 14) Xphase 

def xphasestable(m,n,t,y,s,o):
    return sum(m.x_pump[n,τ-1,s] for τ in range(2, y+1)) >= m.x_phase[n,t,y,s,o] 
m.xphasestable = Constraint(OmegaBus, OmegaT, OmegaStg, OmegaS, OmegaO, rule = xphasestable)

opt = SolverFactory('gurobi')
#opt.options['Solfiles'] = 'solution'
#opt.options['NonConvex'] = 2
opt.options['MIPGap'] = 0.01
opt.options['DualReductions'] = 0
# opt.options['NodeMethod'] =    # (-1 auto, 0 primal, 1 simplex, 2 barrier

# opt.options['PoolSolutions'] = 2
# opt.options['PoolSearchMode'] = 2

opt.options['MIPFocus'] = 2  # Focus on finding feasible solution, proving optimality or imporving the bound
opt.options['Cuts'] = 3  # 3 aggressvive
opt.options["ImproveStartTime"] = 300
opt.options['TimeLimit'] = 50000
# opt.options['Threads']   = 20
#opt.options['Heuristics'] = 0.2 # percentage of time the algorithm resorts to heuristics 
#opt.options['Threads'] = 8


timepre = time.time()
results = opt.solve(m, load_solutions=True, tee = True)
timepos = time.time()
print(timepos-timepre)

# m.obj.display()

results.write()


for variable in m.component_objects(Var, active=True):
    df = pd.DataFrame(variable.get_values().values(), index = variable.get_values().keys())
    df.to_csv("monolithic_test_results/unlinked_hydro_{}-{}.csv".format(variable, case_name), index=True)
    
with open('monolithic_test_results/unlinked_hydro_results_{}.txt'.format(case_name), 'w') as f:
    f.write(repr(results))


