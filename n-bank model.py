
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import PolynomialFeatures
import joblib

import os
import warnings
warnings.filterwarnings('ignore')


# load hydrogen property models made by machine learning
poly_features = PolynomialFeatures(degree=2, include_bias=False) # for using the property models

load_Density = joblib.load(os.path.abspath("Property models") + "\\Density.pkl")
load_Internal_Energy = joblib.load(os.path.abspath("Property models") + "\\Internal Energy.pkl")
load_Enthalpy = joblib.load(os.path.abspath("Property models") + "\\Enthalpy.pkl")
load_Cv = joblib.load(os.path.abspath("Property models") + "\\Cv.pkl")

M = 2.016/1000 # [kg/mol]
R = 8.314*(10**-5) # [m3*bar/mol*K]

class Hydrogen_Fueling():
    global M
    global R
    
    def __init__(self, Line_T_init, Tank_V, P_tank_init, T_amb, APRR, P_st_low, P_st_mid, P_st_high, P_st_high2, n_bank, stop_moment):
        self.Line_T_init = Line_T_init
        self.Tank_V = Tank_V
        self.P_tank_init = P_tank_init
        self.T_amb = T_amb
        self.P_st_low = P_st_low
        self.P_st_mid = P_st_mid
        self.P_st_high = P_st_high
        self.P_st_high2 = P_st_high2
        self.APRR = APRR
        self.n_bank = n_bank
        self.stop_moment = stop_moment
        
    # init Cp, Cv
    def Heat_capacity(self): # [J/kg*K]
        A = 33.066178
        B = -11.363417
        C = 11.432816
        D = -2.772874
        E = -0.158558

        Cp = (A + B*(self.Line_T_init/1000) + C*((self.Line_T_init/1000)**2) + D*((self.Line_T_init/1000)**3) + E/((self.Line_T_init/1000)**2)) / M # [J/kg*K]
        Cv = (Cp - ((R*(10**5))/M)) # [J/K]
        return Cp, Cv
    
    
    # heat transfer in the fueling line
    def Heat_transfer(self, D_in_st, D_out_st, D_in_ve, D_out_ve, L_st, L_ve):
        
        def heat_trans(Di,Do,L):
            k = 1.5 # [W/m*K]
            S = 2 * np.pi / (np.log(Do / Di)) # [-]
            q = k * S * L * (self.T_amb - self.Line_T_init) # [W]
            return q
        
        heat = heat_trans(D_in_st, D_out_st, L_st)
        return heat
    
    
    def logic(self):
        Q = self.Heat_transfer(0.01, 0.03, 0.0013, 0.01, 4.479, 5.538)
        Cp = self.Heat_capacity()[0]
        Cv = self.Heat_capacity()[1]

        # initial value of the fueling for calculation
        Line_T_list = [self.Line_T_init]
        Line_P_list = [self.P_st_low]
        P_tank_list = [self.P_tank_init]
        T_tank_list = [self.T_amb - 273]
        
        SOC = 0
        i = 0
        mass = 0
        Filter_P_drop_init = 0
        MfM_P_drop_init = 0
        
        bank1_time = []
        m_list = []
        mass_list = []
        Line_ρ_list = []
        Valve_P_drop_list = []
        SOC_list = []
        internal_energy_list = []


        try:
            """ --------------------------------- The first bank ---------------------------------- """
            while SOC < 90:
                i = i + 1
                
                # mass flow rate
                c1 = 0.013834957
                c2 = 0.010681871
                d1 = -0.000174506 
                d2 = -0.000236707
                P1 = 15*(10**6) # [Pa]
                P2 = 20*(10**6) # [Pa]
    
                m = ((Line_P_list[i-1] - P_tank_list[i-1])*(10**6))/(10**4) * (c1 - ((P2 - (self.APRR*i)*(10**6)) / (P2 - P1)) * (c1-c2)) * np.exp((d1 - ((P2 - (self.APRR*i)*(10**6)) / (P2 - P1)) * (d1-d2)))
                m_list.append(m) # [g/s]
                
                
                # SOC
                mass = mass + m_list[i-1]
                mass_list.append(mass)
                
                SOC = (self.P_tank_init/70*40.2) + (mass/1000)/(40.2*self.Tank_V/1000) * 100
                SOC_list.append(SOC)
                
                
                ######################################## fueling line ########################################
                # Temp. Pres. values of fueling line
                Line_TP_list = pd.DataFrame([Line_T_list[i-1], Line_P_list[i-1]]).transpose()
                Line_TP_poly = poly_features.fit_transform(Line_TP_list)
                
                
                # density
                Line_ρ = load_Density.predict(Line_TP_poly) * M * 1000 # [mol/L] -> [kg/m3]
                Line_ρ_list.append(Line_ρ[0])
                
                
                # heat transfer
                Line_T = self.Line_T_init + (Q/180) / ((m_list[i-1]/1000) * Cp)
                self.Line_T_init = Line_T
                Line_T_list.append(Line_T)
                 
                
                # pressure loss
                k_v = 0.75 # [m3/h]
                Area = 1 # [m2]
                filter_k_p = 100 # [dimensionless]
                mass_flow_meter_k_p = 2.8 # [dimensionless]
                 
                vol_flow_rate = (m_list[i-1]/1000) / Line_ρ_list[i-1]
                Valve_P_drop = ((36*(vol_flow_rate*1000)/k_v)**2)/1000
                Valve_P_drop_list.append(Valve_P_drop)
                
                if Valve_P_drop_list[i-2] >= Valve_P_drop_list[i-1]:
                    Valve_P_drop_list[i-1] = Valve_P_drop_list[i-2]
    
                Filter_P_drop = Filter_P_drop_init + 0.5 * filter_k_p * Line_ρ_list[i-1] * (((m_list[i-1]/1000) / (Area * Line_ρ_list[i-1]))**2)
                Filter_P_drop_init = Filter_P_drop
                
                MfM_P_drop = MfM_P_drop_init + 0.5 * mass_flow_meter_k_p * Line_ρ_list[i-1] * (((m_list[i-1]/1000) / (Area * Line_ρ_list[i-1]))**2)
                MfM_P_drop_init = MfM_P_drop
                
                Line_P = self.P_st_low - Filter_P_drop - MfM_P_drop - Valve_P_drop_list[i-1]              
                Line_P_list.append(Line_P)
                
                
                # Cp, Cv
                A = 33.066178
                B = -11.363417
                C = 11.432816
                D = -2.772874
                E = -0.158558
                
                Cp = (A + B*(self.Line_T_init/1000) + C*((Line_T_list[i-1]/1000)**2) + D*((Line_T_list[i-1]/1000)**3) + E /((Line_T_list[i-1]/1000)**2)) / M # [J/kg*K]
                Cv = (Cp - ((R*(10**5))/M)) # [J/K]
                ######################################## fueling line ########################################
                
                
                
                ######################################## fueling tank ########################################
                # compressibility factor
                a = [0.05888460, -0.06136111, -0.002650473, 0.002731125, 0.001802374, -0.001150707, 0.9588528*(10**-4), -0.1109040*(10**-6), 0.1264403*(10**-9)]
                b = [1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4]
                c = [1, 1, 2, 2, 2.42, 2.63, 3, 4, 5]
                
                Z = 0
                for w in range(9):
                    Z = Z + a[w] * ((100/Line_T_list[i-2])**b[w]) * (Line_P_list[i-1]**c[w])
                Z = Z + 1
                
                
                # innter temperature 
                Tank_TP_list = pd.DataFrame([(T_tank_list[i-1]+273), P_tank_list[i-1]]).transpose()
                Tank_TP_poly = poly_features.fit_transform(Tank_TP_list)
    
                ld_U = load_Internal_Energy.predict(Tank_TP_poly)[0] # [KJ/mol]
                ld_H = load_Enthalpy.predict(Tank_TP_poly)[0] # [KJ/mol]
                ld_Cv = load_Cv.predict(Tank_TP_poly)[0] # [J/mol*K]
                
                internal_energy = ld_U / M * 1000 # [J/kg]
                internal_energy_list.append(internal_energy)
                enthalpy = ld_H / M * 1000 # [J/kg]
                CV = ld_Cv / M # [J/kg*K]
                
                Tank_Q = enthalpy * (m_list[i-1]/1000) * i - ((m_list[i-1]/1000 * internal_energy_list[i-1]) - (m_list[i-2]/1000 * internal_energy_list[i-2])) * i # [J]
                ΔT = Tank_Q / (CV * mass)   
                
                T_tank = T_tank_list[i-1] + ΔT
                T_tank_list.append(T_tank)
                
                
                # inner pressure
                ρ_tank = mass/(self.Tank_V) # [kg/m3]
                
                P_tank = self.P_tank_init + (Z * ρ_tank * R * (T_tank_list[i-1] + 273) / M) / 10 # [bar -> MPa]
                P_tank_list.append(P_tank)
                ######################################## fueling tank ########################################
    
    
                """ Value of the first bank """
                print(i, "s", "SOC : ", SOC, "[%]")
                print(i, "s", "Massflow Rate : ", m_list[i-1], "[g/s]")
                print(i, "s", "Line Temperature : ", Line_T_list[i-1] - 273, "[℃]")
                print(i, "s", "Line Pressure : ", Line_P_list[i-1], "[MPa]")
                print(i, "s", "Density : ", Line_ρ_list[i-1], "[kg/m3]")
                print(i, "s", "Tank Temperature : ", T_tank_list[i-1], "[℃]")
                print(i, "s", "Tank Pressure : ", P_tank_list[i-1], "[MPa]")
                print('\n')
                
                bank1_time.append(i)
                
                # switch moment of the bank
                if i > 50:
                    if m < self.stop_moment:
                        break
            
            
            
            """ --------------------------------- After the first bank ---------------------------------- """
            bank2_time = []
            bank3_time = []
            bank4_time = []
            bank5_time = []      
            rep = ["q", "w", "e", "r"]
            for i in range(self.n_bank-1):
                
                rep[i] = 0
                while SOC < 90:
                    rep[i] = rep[i] + 1

                    # mass flow rate
                    c1 = 0.013834957
                    c2 = 0.010681871
                    d1 = -0.000174506 
                    d2 = -0.000236707
                    P1 = 15*(10**6) # [pa]
                    P2 = 20*(10**6) # [pa]
                    
                    m = ((Line_P - P_tank)*(10**6))/(10**4) * (c1 - ((P2 - (self.APRR*rep[i])*(10**6)) / (P2 - P1)) * (c1-c2)) * np.exp((d1 - ((P2 - (self.APRR*rep[i])*(10**6)) / (P2 - P1)) * (d1-d2)))
                    m_list.append(m) # [g/s]

                    
                    # SOC
                    mass = mass + m
                    mass_list.append(mass)
                    
                    SOC = (self.P_tank_init/70*40.2) + (mass/1000)/(40.2*self.Tank_V/1000) * 100
                    SOC_list.append(SOC)
                    
                    
                    ######################################## fueling line ########################################
                    # Temp. Pres. values of fueling line
                    Line_TP_list = pd.DataFrame([Line_T, Line_P]).transpose()
                    Line_TP_poly = poly_features.fit_transform(Line_TP_list)
                    
                    
                    # density
                    Line_ρ = load_Density.predict(Line_TP_poly)[0] * M * 1000 # [mol/L] -> [kg/m3]
                    Line_ρ_list.append(Line_ρ)
                    
                    
                    # heat transfer
                    Line_T = self.Line_T_init + (Q/180) / ((m/1000) * Cp)
                    self.Line_T_init = Line_T
                    Line_T_list.append(Line_T)
                    
                    
                    # pressure loss
                    k_v = 0.75 # [m3/h]
                    Area = 1 # [m2]
                    filter_k_p = 100 # [dimensionless]
                    mass_flow_meter_k_p = 2.8 # [dimensionless]
                     
                    vol_flow_rate = (m/1000) / Line_ρ
                    Valve_P_drop = ((36*(vol_flow_rate*1000)/k_v)**2)/1000
                    Valve_P_drop_list.append(Valve_P_drop)
                    
                    if Valve_P_drop_list[rep[i]-2] >= Valve_P_drop_list[rep[i]-1]:
                        Valve_P_drop_list[rep[i]-1] = Valve_P_drop_list[rep[i]-2]
        
                    Filter_P_drop = Filter_P_drop_init + 0.5 * filter_k_p * Line_ρ * (((m/1000) / (Area * Line_ρ))**2)
                    Filter_P_drop_init = Filter_P_drop
                    
                    MfM_P_drop = MfM_P_drop_init + 0.5 * mass_flow_meter_k_p * Line_ρ * (((m/1000) / (Area * Line_ρ))**2)
                    MfM_P_drop_init = MfM_P_drop
                    
                    if i == 0:
                        Line_P = self.P_st_mid - Filter_P_drop - MfM_P_drop - Valve_P_drop
                        Line_P_list.append(Line_P)
                        
                    elif i == 1:
                        Line_P = self.P_st_high - Filter_P_drop - MfM_P_drop - Valve_P_drop
                        Line_P_list.append(Line_P)
                        
                    elif i == 2:
                        Line_P = self.P_st_high2 - Filter_P_drop - MfM_P_drop - Valve_P_drop
                        Line_P_list.append(Line_P)
                        
                    
                    # Cp, Cv
                    A = 33.066178
                    B = -11.363417
                    C = 11.432816
                    D = -2.772874
                    E = -0.158558
                    
                    Cp = (A + B*(self.Line_T_init/1000) + C*((Line_T/1000)**2) + D*((Line_T/1000)**3) + E /((Line_T/1000)**2)) / M # [J/kg*K]
                    Cv = (Cp - ((R*(10**5))/M)) # [J/K]
                    ######################################## fueling line ########################################
                    
                    
                    
                    ######################################## fueling tank ########################################
                    # compressibility factor
                    a = [0.05888460, -0.06136111, -0.002650473, 0.002731125, 0.001802374, -0.001150707, 0.9588528*(10**-4), -0.1109040*(10**-6), 0.1264403*(10**-9)]
                    b = [1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4]
                    c = [1, 1, 2, 2, 2.42, 2.63, 3, 4, 5]
                    
                    Z = 0
                    for w in range(9):
                        Z = Z + a[w] * ((100/Line_T)**b[w]) * (Line_P**c[w])
                    Z = Z + 1
                    
                    
                    # inner temperature
                    Tank_TP_list = pd.DataFrame([(T_tank + 273), P_tank]).transpose()
                    Tank_TP_poly = poly_features.fit_transform(Tank_TP_list)
                    
                    ld_U = load_Internal_Energy.predict(Tank_TP_poly)[0] # [KJ/mol]
                    ld_H = load_Enthalpy.predict(Tank_TP_poly)[0] # [KJ/mol]
                    ld_Cv = load_Cv.predict(Tank_TP_poly)[0] # [J/mol*K]
                    
                    internal_energy = ld_U / M * 1000 # [J/kg]
                    internal_energy_list.append(internal_energy)
                    enthalpy = ld_H / M * 1000 # [J/kg]
                    CV = ld_Cv / M # [J/kg*K]
                    
                    Tank_Q = enthalpy * (m/1000) * rep[i] - ((m/1000 * internal_energy) - (m/1000 * internal_energy)) * rep[i] # [J]
                    ΔT = Tank_Q / (CV * mass)   
                    
                    T_tank = T_tank_list[-1] + ΔT
                    T_tank_list.append(T_tank)
                    
                    
                    # inner pressure
                    ρ_tank = mass/(self.Tank_V) # [kg/m3]
                    
                    P_tank = self.P_tank_init + (Z * ρ_tank * R * (T_tank + 273) / M) / 10 # [bar -> MPa]
                    P_tank_list.append(P_tank)
                    ######################################## fueling tank ########################################
                    
                    """ Value 결과값 """
                    
                    # Value of the second bank
                    if i == 0:
                        print(bank1_time[-1] + rep[i], "s", "SOC : ", SOC, "[%]")
                        print(bank1_time[-1] + rep[i], "s", "Massflow Rate : ", m, "[g/s]")
                        print(bank1_time[-1] + rep[i], "s", "Line Temperature : ", Line_T - 273, "[℃]")
                        print(bank1_time[-1] + rep[i], "s", "Line Pressure : ", Line_P, "[MPa]")
                        print(bank1_time[-1] + rep[i], "s", "Density : ", Line_ρ, "[kg/m3]")
                        print(bank1_time[-1] + rep[i], "s", "Tank Temperature : ", T_tank, "[℃]")
                        print(bank1_time[-1] + rep[i], "s", "Tank Pressure : ", P_tank, "[MPa]")
                        print('\n')
            
                        bank2_time.append(bank1_time[-1] + rep[i])
                    
                    # Value of the third bank
                    elif i == 1:
                        print(bank2_time[-1] + rep[i], "s", "SOC : ", SOC, "[%]")
                        print(bank2_time[-1] + rep[i], "s", "Massflow Rate : ", m, "[g/s]")
                        print(bank2_time[-1] + rep[i], "s", "Line Temperature : ", Line_T - 273, "[℃]")
                        print(bank2_time[-1] + rep[i], "s", "Line Pressure : ", Line_P, "[MPa]")
                        print(bank2_time[-1] + rep[i], "s", "Density : ", Line_ρ, "[kg/m3]")
                        print(bank2_time[-1] + rep[i], "s", "Tank Temperature : ", T_tank, "[℃]")
                        print(bank2_time[-1] + rep[i], "s", "Tank Pressure : ", P_tank, "[MPa]")
                        print('\n')
            
                        bank3_time.append(bank2_time[-1]+ rep[i])
                    
                    # Value of the fourth bank
                    elif i == 2:
                        print(bank3_time[-1] + rep[i], "s", "SOC : ", SOC, "[%]")
                        print(bank3_time[-1] + rep[i], "s", "Massflow Rate : ", m, "[g/s]")
                        print(bank3_time[-1] + rep[i], "s", "Line Temperature : ", Line_T - 273, "[℃]")
                        print(bank3_time[-1] + rep[i], "s", "Line Pressure : ", Line_P, "[MPa]")
                        print(bank3_time[-1] + rep[i], "s", "Density : ", Line_ρ, "[kg/m3]")
                        print(bank3_time[-1] + rep[i], "s", "Tank Temperature : ", T_tank, "[℃]")
                        print(bank3_time[-1] + rep[i], "s", "Tank Pressure : ", P_tank, "[MPa]")
                        print('\n')
            
                        bank4_time.append(bank3_time[-1] + rep[i])
                  
                    # Value of the fifth bank
                    elif i == 3:
                        print(bank4_time[-1] + rep[i], "s", "SOC : ", SOC, "[%]")
                        print(bank4_time[-1] + rep[i], "s", "Massflow Rate : ", m, "[g/s]")
                        print(bank4_time[-1] + rep[i], "s", "Line Temperature : ", Line_T - 273, "[℃]")
                        print(bank4_time[-1] + rep[i], "s", "Line Pressure : ", Line_P, "[MPa]")
                        print(bank4_time[-1] + rep[i], "s", "Density : ", Line_ρ, "[kg/m3]")
                        print(bank4_time[-1] + rep[i], "s", "Tank Temperature : ", T_tank, "[℃]")
                        print(bank4_time[-1] + rep[i], "s", "Tank Pressure : ", P_tank, "[MPa]")
                        print('\n')
            
                        bank5_time.append(bank4_time[-1] + rep[i])
    
                    # switch moment of each bank
                    if i != (self.n_bank - 2):
                        if rep[i] > 60:
                            if m < self.stop_moment:
                                break
                    
                    total_time = bank1_time + bank2_time + bank3_time + bank4_time + bank5_time
                
        except:
            pass
                
        
        """"""""""""""""""""""""""""""""""""""""""" Results plot """""""""""""""""""""""""""""""""""""""""""""""""""  
        # limitation line
        # limit_flow = []
        # limit_temp = []
        # limit_pres = []
        
        # for i in total_time:
        #     Flow_limit = 60
        #     Temp_limit = 85
        #     Pres_limit = 87.5
            
        #     limit_flow.append(Flow_limit)
        #     limit_temp.append(Temp_limit)
        #     limit_pres.append(Pres_limit)
            
        # plt.plot(total_time, limit_flow, color='orange')
        # plt.plot(total_time, limit_temp, color='dodgerblue')
        # plt.plot(total_time, limit_pres, color='green')
        
        # results
        # plt.plot(total_time, SOC_list[:len(total_time)], label = 'SOC', c='darkred')
        # plt.plot(total_time, Line_P_list[:len(total_time)], label = 'Line Pressure')
        plt.plot(total_time, m_list[:len(total_time)], label = 'Massflow', color='orange')
        plt.plot(total_time, T_tank_list[:len(total_time)], label = 'Tank Temperature', color='dodgerblue')
        plt.plot(total_time, P_tank_list[:len(total_time)], label = 'Tank Pressure', color='green') 
 
        plt.xlim(0, len(total_time)+9)
        plt.ylim(0, 100)
        plt.xlabel('Time [s]')
        # plt.ylabel('SOC [%], Pressure [MPa], Massflow [g/s]')
        plt.ylabel('Massflow [g/s], Temperature [℃], Pressure [MPa]')
        plt.legend()
        plt.grid()
        plt.show()  
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        return SOC_list, m_list, Line_T_list, Line_P_list, Line_ρ_list, T_tank_list, P_tank_list, total_time, bank1_time, bank2_time, bank3_time, bank4_time

# Execution code
a = Hydrogen_Fueling(233, 156.6, 5, 298, 28.5/60, 45, 87.5, 87.5, 87.5, 3, 7)
result = a.logic()


""" Optimization """
# def Optimization(stop_moment, low_bank, mid_bank, high_bank, high_bank2, bank_number):
    
#     a = Hydrogen_Fueling(233, 156.6, 5, 298, 28.5/60, low_bank, mid_bank, high_bank, high_bank2, int(round(bank_number)), stop_moment)
#     result = a.logic()
    
#     ##########################################################################
#     Power = 132 # [kW]
    
#     firs_bank_Energy = result[8][-1] * Power * 0.5 # [kJ]
#     secd_bank_Energy = (result[9][-1] - result[8][-1]) * Power # [kJ]
    
#     if int(round(bank_number)) == 2:
#         n_bank_energy = firs_bank_Energy + secd_bank_Energy
    
#     elif int(round(bank_number)) == 3:
#         thrd_bank_Energy = (result[10][-1] - result[9][-1]) * Power # [kJ]
#         n_bank_energy = firs_bank_Energy + secd_bank_Energy + thrd_bank_Energy
        
#     else:
#         thrd_bank_Energy = (result[10][-1] - result[9][-1]) * Power # [kJ]
#         four_bank_Energy = (result[11][-1] - result[10][-1]) * Power # [kJ]
#         n_bank_energy = firs_bank_Energy + secd_bank_Energy + thrd_bank_Energy + four_bank_Energy
#     ##########################################################################
    
#     time = result[7][-1]

#     return - n_bank_energy * time


# pbounds = {'stop_moment': (10, 30),
#                 'bank_number': (2,4),
#                 'low_bank': (40, 87.5),
#                 'mid_bank': (40, 87.5),
#                 'high_bank': (40, 87.5),
#                 'high_bank2': (40, 87.5),
#                 }


# bo=BayesianOptimization(f=Optimization, pbounds=pbounds, verbose=2, random_state=1)
# bo.maximize(init_points=1000, n_iter=200, acq='ei', xi=0.01)
# print(bo.max)

# # Best_P_tank_init = int(bo.max['params']['P_tank_init'])
# # Best_T_amb = int(bo.max['params']['T_amb'])
# Best_stop_moment = int(bo.max['params']['stop_moment'])
# Best_low_pres = int(bo.max['params']['low_bank'])
# Best_mid_pres = int(bo.max['params']['mid_bank'])
# Best_high_pres = int(bo.max['params']['high_bank'])
# Best_high_pres2 = int(bo.max['params']['high_bank2'])
# Best_bank_num = int(round(bo.max['params']['bank_number']))

# a = Hydrogen_Fueling(233, 156.6, 5, 298, 28.5/60, Best_low_pres, Best_mid_pres, Best_high_pres, Best_high_pres2, Best_bank_num, Best_stop_moment)
# result = a.logic()


# ##########################################################################
# Power = 132 # [kW]

# firs_bank_Energy = result[8][-1] * Power * 0.5 # [kJ]
# secd_bank_Energy = (result[9][-1] - result[8][-1]) * Power # [kJ]
# thrd_bank_Energy = (result[10][-1] - result[9][-1]) * Power # [kJ]

# n_bank_energy = (firs_bank_Energy + secd_bank_Energy + thrd_bank_Energy)
# ##########################################################################

# time = result[7][-1]

# print("충전시간 =", time, "소모 에너지 =", n_bank_energy, "탱크변경시점 =", int(bo.max['params']['stop_moment']))
# print("충전시간 =", time, "탱크변경시점 =", Best_stop_moment, "공급탱크 개수 =", Best_bank_num)
# print("Low bank 압력 =", Best_low_pres, "Mid bank 압력 =", Best_mid_pres, "High bank 압력 =", Best_high_pres)


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
# limit_flow = []
# limit_temp = []
# limit_pres = []

# # for i in total:
# #     Flow_limit = 60
# #     Temp_limit = 85
# #     Pres_limit = 87.5
    
# #     limit_flow.append(Flow_limit)
# #     limit_temp.append(Temp_limit)
# #     limit_pres.append(Pres_limit)


# # plt.plot(total, SOC_list[:len(total)], label = 'SOC', c='darkred')
# # plt.plot(total, Line_P_list[:len(total)], label = 'Line Pressure')
# plt.plot(result[7], result[1][:len(result[7])], label = 'Massflow', color='orange')
# plt.plot(result[7], result[5][:len(result[7])], label = 'Tank Temperature', color='dodgerblue')
# plt.plot(result[7], result[6][:len(result[7])], label = 'Tank Pressure', color='green') 

# # plt.plot(total, limit_flow, color='orange')
# # plt.plot(total, limit_temp, color='dodgerblue')
# # plt.plot(total, limit_pres, color='green')


# plt.xlim(0, len(result[7])+9)
# plt.ylim(0, 100)
# plt.xlabel('Time [s]')
# # plt.ylabel('SOC [%], Pressure [MPa], Massflow [g/s]')
# plt.ylabel('Massflow [g/s], Temperature [℃], Pressure [MPa]')
# plt.legend()
# plt.grid()
# plt.show()  
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



