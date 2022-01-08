import math
import os
import sys
import random
import numpy as np
import pandas as pd
from scipy.integrate import odeint

class kinetic_modeling():
    def __init__(self):
        self.null  = 0
        # unit charge q
        self.q=1.0*1.60218e-19
        # Boltzmann coefficient kb
        self.kb=1.38064853e-23
        # conversion factor from kcal to j
        self.kcal2j = 4184.0
        # Avogadro number mol**-1
        self.Navg   = 6.02e23

        # Gating charge of releasing 1st Na to the outside of the cell
        self.don1=0.75
        # Gating charges of binding 3 Na from the inside of the cell
        self.din1=0.0
        self.din2=0.0
        self.din3=0.25
        # Gating charges of binding 2 K from the outside of the cell
        self.dok1=-0.46
        self.dok2=-0.27
        # Gating charges of releasing 2K to the inside of the cell
        self.dik1=0.0
        self.dik2=0.0
        # The total charge moved is 1, the release of 2,3 Na to the outside is computed by 1 - what's known
        # Just for initialization, set the 2 unknown to 999
        self.don2 = 999
        self.don3 = 999
        # Make a list of all the gating charges
        self.gating_charges = [self.don1,self.don2,self.don3,self.din1,self.din2,self.din3,self.dok1,self.dok2,self.dik1,self.dik2]

        # base activation energy
        self.lnA = 12

        self.vdink =  0
        self.vdinn =  0

    def mymodel(self,x,t,coefficients):
            lfctr1,rfctr1a,rfctr1b,rfctr1c,rfctr2a,rfctr2b,rfctr2c,rfctr3a,rfctr3b,rfctr3c,\
            rfctr4a,rfctr4b,rfctr4c,rfctr5a,rfctr5b,rfctr5c,rfctr6a,rfctr6b,rfctr6c,\
            rfctr7a,rfctr7b,rfctr7c,rfctr8a,rfctr8b,rfctr8c,rfctr9a,rfctr9b,rfctr9c,rfctr10a,rfctr10b,rfctr10c = coefficients
            dx0 = (rfctr1a*x[1]+rfctr1b*x[0]+rfctr1c*x[9])/lfctr1
            dx1 = rfctr2a*x[2]+rfctr2b*x[1]+rfctr2c*x[0]
            dx2 = rfctr3a*x[3]+rfctr3b*x[2]+rfctr3c*x[1]
            dx3 = rfctr4a*x[4]+rfctr4b*x[3]+rfctr4c*x[2]
            dx4 = rfctr5a*x[5]+rfctr5b*x[4]+rfctr5c*x[3]
            dx5 = rfctr6a*x[6]+rfctr6b*x[5]+rfctr6c*x[4]
            dx6 = rfctr7a*x[7]+rfctr7b*x[6]+rfctr7c*x[5]
            dx7 = rfctr8a*x[8]+rfctr8b*x[7]+rfctr8c*x[6]
            dx8 = rfctr9a*x[9]+rfctr9b*x[8]+rfctr9c*x[7]
            dx9 = rfctr10a*x[0]+rfctr10b*x[9]+rfctr10c*x[8]
            return [dx0,dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9]

    def update_vol_dependent_ks(self,all_k_params,gating_charges,vmp,q,kbt):
        """
            update kon and offs that are voltage dependent
        """
        # unpacking all_k_params
        kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
        kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2 = all_k_params

        #kp1c,kn1c,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b = subset_ks
        don1,don2,don3,din1,din2,din3,dok1,dok2,dik1,dik2 = gating_charges

        # update selected ks
        kp1c=kp1c*math.exp(0.5*din3*vmp*q/kbt)
        kn1c=kn1c*math.exp(0-0.5*din3*vmp*q/kbt)
        
        kp3=kp3*math.exp(0.7*don1*vmp*q/kbt)
        kn3=kn3*math.exp(-0.3*don1*vmp*q/kbt)

        kp4b=kp4b*math.exp(0.5*don2*vmp*q/kbt)
        kn4b=kn4b*math.exp(0-0.5*don2*vmp*q/kbt)

        kp4a=kp4a*math.exp(0.5*don3*vmp*q/kbt)
        kn4a=kn4a*math.exp(0-0.5*don3*vmp*q/kbt)

        kp5a=kp5a*math.exp(0*dok1*vmp*q/kbt)
        kn5a=kn5a*math.exp(-1.0*dok1*vmp*q/kbt)

        kp5b=kp5b*math.exp(0*dok2*vmp*q/kbt)
        kn5b=kn5b*math.exp(-1.0*dok2*vmp*q/kbt)
        updated_all_k_params = [ kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
                                 kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2]
        
        return updated_all_k_params

    def update_gating_charges(self,don2):
        """
            Update gating charges (mostly don2 and don3) based on the new don2
        """
         # update gating charge based on input don2
        don1,don2_tmp,don3_tmp,din1,din2,din3,dok1,dok2,dik1,dik2 = self.gating_charges
        don23tot = 1.0 - dok1 - dok2 -dik1 - dik2 - din1 -din2 - din3 - don1
        don3 = don23tot - don2
        gating_charges = [ don1,don2,don3,din1,din2,din3,dok1,dok2,dik1,dik2 ]
        return gating_charges

    def change_units(self,v,concentrations):
        """
            convert the concentrations from mV to V and mM to M
        """
        vmp = v*1e-3
        cno,cni,cko,cki,catp,cadp,cpi = np.array(concentrations,dtype=float)*1e-3
        return vmp,cno,cni,cko,cki,catp,cadp,cpi

    def prepare_parameters(self,temp,v,concentrations,all_k_params,don2):
        """
           compute the left and right coefficients in the set of ODEs
        """
        # convert the concentrations from mV to V and mM to M
        vmp,cno,cni,cko,cki,catp,cadp,cpi = self.change_units(v,concentrations)
        
        # update gating charge based on input don2
        gating_charges = self.update_gating_charges(don2)

        # Update ks of voltage dependent steps
        kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
        kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2 = self.update_vol_dependent_ks(all_k_params,gating_charges,vmp,q=self.q,kbt=self.kb*temp)

        # compute the coefficients in the ODEs 
        lfctr1=1+cni/keqn1+cni**2/keqn1/keqn2+cki/keqk1+cki**2/keqk1/keqk2
        rfctr1a=kn1c
        rfctr1b=-kp1c*cni**3/keqn1/keqn2-kn7*cki**2/keqk1/keqk2
        rfctr1c=kp7
        rfctr2a=kn2*cadp
        rfctr2b=-kp2-kn1c
        rfctr2c=kp1c*cni**3/keqn1/keqn2
        rfctr3a=kn3*cno
        rfctr3b=-kp3-kn2*cadp
        rfctr3c=kp2
        rfctr4a=kn4b*cno
        rfctr4b=-kp4b-kn3*cno
        rfctr4c=kp3
        rfctr5a=kn4a*cno
        rfctr5b=-kp4a-kn4b*cno
        rfctr5c=kp4b
        rfctr6a=kn5a
        rfctr6b=-kp5a*cko-kn4a*cno
        rfctr6c=kp4a
        rfctr7a=kn5b
        rfctr7b=-kp5b*cko-kn5a
        rfctr7c=kp5a*cko
        rfctr8a=kn6*cpi
        rfctr8b=-kp6-kn5b
        rfctr8c=kp5b*cko
        rfctr9a=kna
        rfctr9b=-kpa*catp-kn6*cpi
        rfctr9c=kp6
        rfctr10a=kn7*cki**2/keqk1/keqk2
        rfctr10b=-kp7-kna
        rfctr10c=kpa*catp

        coeff_list = [lfctr1,rfctr1a,rfctr1b,rfctr1c,rfctr2a,rfctr2b,rfctr2c,rfctr3a,rfctr3b,rfctr3c,rfctr4a,rfctr4b,rfctr4c,\
                      rfctr5a,rfctr5b,rfctr5c,rfctr6a,rfctr6b,rfctr6c,rfctr7a,rfctr7b,rfctr7c,rfctr8a,rfctr8b,rfctr8c,\
                      rfctr9a,rfctr9b,rfctr9c,rfctr10a,rfctr10b,rfctr10c]
        return coeff_list

    def get_flux(self,temp,v,concentrations,react_conc,all_k_params,don2):
        """
            At a given condiction: temp,v,concentrations and reactant concentrations, 
            calculate the flux in the system as the average of all edges
        """
        # convert the concentrations from mV to V and mM to M
        vmp,cno,cni,cko,cki,catp,cadp,cpi = self.change_units(v,concentrations)
        
        # update gating charge based on input don2
        gating_charges = self.update_gating_charges(don2)

        # Update ks of voltage dependent steps
        kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
        kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2 = self.update_vol_dependent_ks(all_k_params,gating_charges,vmp,q=self.q,kbt=self.kb*temp)

        # Unpack react_conc
        flux = [  kp1c*cni*react_conc[0]*cni**2/keqn1/keqn2 - kn1c*react_conc[1], 
                  kp2*react_conc[1] - kn2*cadp*react_conc[2],
                  kp3*react_conc[2] - kn3*cno*react_conc[3],
                  kp4b*react_conc[3] - kn4b*cno*react_conc[4],
                  kp4a*react_conc[4] - kn4a*cno*react_conc[5],
                  kp5a*cko*react_conc[5] - kn5a*react_conc[6],
                  kp5b*cko*react_conc[6] - kn5b*react_conc[7],
                  kp6*react_conc[7] - kn6*cpi*react_conc[8],
                  kpa*catp*react_conc[8] - kna*react_conc[9],
                  kp7*react_conc[9] - kn7*cki**2*react_conc[0]/keqk1/keqk2  ]

        return flux

    def get_cycle_energy(self,v,temp,concentrations,dg_atp_0=-31):
        """
            Get the energy of the entire cycle based on 
            2K_o + 3Na_i + ATP <----> 2K_i + 3Na_o + ADP + Pi 
            ATP <----> ADP + Pi, dG_0 = -31 kj/mol
            2K_o  <----> 2K_i      2*qVmp
            3Na_i <---->  + 3Na_o  -3*qVmp
        """
        # convert the concentrations from mV to V and mM to M
        vmp,cno,cni,cko,cki,catp,cadp,cpi = self.change_units(v,concentrations)

        RT  = self.kb * temp * self.Navg / self.kcal2j
        
        # energies are all in kcal/mol

        # For ATP hydrolysis
        dG_atp = dg_atp_0/self.kcal2j + RT*math.log(cadp*cpi/catp) 

        # For transporting 3Na from inside to outside against concentration gradients
        dG_Na = RT*math.log((cno/cni)**3)

        # For transporting 2K from outside to inside against concentration gradients
        dG_K = RT*math.log((cki/cko)**2)

        # For transporting a charge, negative sign in front because membrane potential is defined as inside - outside, pushing
        # a positive ion from inside to outside in a negative TM potential has a dG > 0
        dG_q = -vmp*self.q *self.Navg / self.kcal2j

        E_ref = dG_atp + dG_Na + dG_K + dG_q
        return E_ref

    def get_model_cycle_energy(self,v,temp,concentrations,all_k_params,don2):
        """
            Get the energy from each step along the cycle using the kon/offs 
        """
        vmp = list(self.change_units(v,concentrations))[0]

        RT  = self.kb * temp * self.Navg / self.kcal2j

        # update gating charge based on input don2
        gating_charges = self.update_gating_charges(don2)

        # Update ks of voltage dependent steps
        kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
        kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2 = self.update_vol_dependent_ks(all_k_params,gating_charges,vmp,q=self.q,kbt=self.kb*temp)

        # Get a list of Keq starting from k5a, i.e., First K+ binding to E2 from the outside
        Keq_list = [ kp5a/kn5a, kp5b/kn5b, kp6/kn6, kpa/kna, kp7/kn7, keqk2, keqk1, 1/keqn2, 1/keqn1, kp1c/kn1c, kp2/kn2, kp3/kn3, kp4b/kn4b, kp4a/kn4a ]

        per_step_energies = [ RT*math.log(Keq) for Keq in Keq_list ]
        total_cycle_energy = np.sum(np.array(per_step_energies,dtype=float))
        return total_cycle_energy

    def read_exp_flux(self,flux_name='exp_flux_data.csv'):
        """
            Read in experimental V, and CNao dependent flux data
        """
        df = pd.read_csv(flux_name)
        vmps = df['V in mV']
        return vmps,df

    def read_exp_qslow_na(self,fname='exp_qslow_na.csv'):
        df = pd.read_csv(fname)
        vmps = df['V in mV']
        return vmps,df

    def read_exp_qslow_k(self,fname='exp_qslow_k.csv'):
        df = pd.read_csv(fname)
        vmps = df['V in mV']
        return vmps,df

    def random_move(self,all_k_params,don2,nbr=5):
        """
            randomly pick nbr out of all the parameters and change them
        """
        all_params = [ ele for ele in all_k_params]
        all_params.append(don2)
        indices = range(0,len(all_params),1)
        indices_to_be_changed = random.sample(indices,k=nbr)
        ## define the ranges of the parameters
        sample_range = random.randint(1,99)/10.0
        sample_range_don2 = random.randint(1,73)/10.0    
        sample_exp_ranges = [ random.randint(6,6),
                              random.randint(3,5),
                              random.randint(1,4),
                              random.randint(5,5),
                              random.randint(1,4),
                              random.randint(1,4),
                              random.randint(1,6),
                              random.randint(1,6),
                              random.randint(3,5),
                              random.randint(2,5),
                              random.randint(2,5),
                              random.randint(-2,3),
                              random.randint(2,5),
                              random.randint(1,5),
                              random.randint(4,6),
                              random.randint(4,6),
                              random.randint(5,7),
                              random.randint(1,4),
                              random.randint(1,4),
                              random.randint(1,4),
                              random.randint(-8,-2),
                              random.randint(-8,-2),
                              random.randint(-8,-2),
                              random.randint(-8,-2),
                              random.randint(-1,-1) ] 
        for i in indices_to_be_changed[:nbr]:
            if i == len(all_params)-1: # don2
                don2 = sample_range_don2 * 10**sample_exp_ranges[i]
            else:  # the rest of ks
                all_params[i] = sample_range * 10**sample_exp_ranges[i]
        new_all_k_params = all_params[:-1]
        new_don2 = don2
        return new_all_k_params,new_don2

    def calc_pump_states_concentrations(self,temp,v,concentrations,all_k_params,don2,init_states_conc,fraction,time=np.linspace(0.0,0.5,10000)):
        """
           calculate steady state concentration for all the states along the cycle 
           at a given Vmp and outside Na/K concentration
           v and concentrations inputs are in mV and mM
           output array
           rows: # of time points
           columns: 0,1,2,3,4,5,6,7,8,9 StateID
        """
        coeff_list = self.prepare_parameters(temp,v,concentrations,all_k_params,don2)
        fin_states_conc_array = odeint(self.mymodel,init_states_conc,time,args=(coeff_list,))
        istart = int(round(np.shape(fin_states_conc_array)[0]*(1-fraction)))
        fin_states_conc = np.average(fin_states_conc_array[istart:,:],axis=0)   # use the last 30% to compute state conc avg
        return fin_states_conc

    def calc_total_state_population(self,all_k_params,state_conc,concentrations):
        """
            compute total pump population taking into consideration the hidden states:
            Na2E1ATP, NaE1ATP, KE1ATP, and K2E1ATP
            For debugging purposes
        """
        cno,cni,cko,cki,catp,cadp,cpi = np.array(concentrations,dtype=float)*1e-3
        e1atp =state_conc[0]
        na_e1atp = cni*e1atp/all_k_params[-4]
        na2_e1atp = cni**2*e1atp/all_k_params[-4]/all_k_params[-3]
        k_e1atp = cki*e1atp/all_k_params[-2]
        k2_e1atp = cki**2*e1atp/all_k_params[-2]/all_k_params[-1]
        tot_population = np.sum(state_conc) + na2_e1atp + na_e1atp + k_e1atp + k2_e1atp
        return tot_population

    def calc_slow_charge_ext_na(self,holding_states_conc,fin_states_conc,don2):
        """
           calculate slow charge movement based on the state population change and the gating charges
           holding_states_conc/fin_states_conc are 1D np.array with 10 elements
        """
        gating_charges = self.update_gating_charges(don2)
        don1,don2,don3 = gating_charges[:3]
        charge_moved  =  (holding_states_conc[2] - fin_states_conc[2]) * (don1 + don2 +don3) + \
                         (holding_states_conc[3] - fin_states_conc[3]) * (don2 + don3 ) + \
                         (holding_states_conc[4] - fin_states_conc[4]) * don3  
        return charge_moved

    def calc_slow_charge_ext_k(self,holding_states_conc,fin_states_conc):
        """
            calculate slow charge movement based on the state population change and the gating charges
            holding_states_conc/fin_states_conc are 1D np.array with 10 elements
        """
        dok1,dok2 = self.dok1,self.dok2
        charge_moved  =  (holding_states_conc[7] - fin_states_conc[7]) * dok2 + \
                         (holding_states_conc[8] - fin_states_conc[8]) * (dok1+dok2) + \
                         (holding_states_conc[9] - fin_states_conc[9]) * (dok1+dok2) 
        charge_moved = 0-charge_moved # flip sign 
        return charge_moved

    def calc_energy_dev_from_ref(self,all_k_params,don2,vs=range(-120,80,20),cnos=[1.5,50,100,150],concentrations=[9999, 50, 5.4, 140, 9.8, 0.05,1.833],temp=310.0):
        """
            calculating deviation of cycle energy for a given parameter set
            concentrations take the format of cno,cni,cko,cki,catp,cadp,cpi
        """
        energy_ref_list, energy_from_ks_list = [],[]
        for v in vs:
            for cno in cnos:
                concentrations[0] = cno
                energy_ref_list.append(self.get_cycle_energy(v,temp,concentrations))
                energy_from_ks_list.append(self.get_model_cycle_energy(v,temp,concentrations,all_k_params,don2))
        energy_ref_array, energy_from_ks_array = np.array(energy_ref_list,dtype=float), np.array(energy_from_ks_list,dtype=float)
        energy_dev_sq_avg = np.average((energy_ref_array-energy_from_ks_array)**2)
        normalizing_factor = np.std(energy_ref_array)**2
        norm_energy_dev = energy_dev_sq_avg/normalizing_factor
        return norm_energy_dev

    def calc_flux_dev_from_ref(self,all_k_params,don2,fraction,cnos=[1.5,50,100,150],concentrations=[9999, 50, 5.4, 140, 9.8, 0.05,1.833],\
                               init_states_conc = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),temp=310.0):
        """
            calculating deviation of flux for a given parameter set from the reference flux
        """
        ## Read in reference flux
        vmps,df = self.read_exp_flux()
        flux_data_ref = np.array(df.iloc[:, 1:])  # n_vmp,4 shaped array
        model_flux = []
        for vmp in vmps:
            for cno in cnos:
                concentrations[0] = cno
                fin_states_conc = self.calc_pump_states_concentrations(temp,vmp,concentrations,all_k_params,don2,init_states_conc,fraction,time=np.linspace(0.0,2.0,10000))
                flux_list = self.get_flux(temp,vmp,concentrations,fin_states_conc,all_k_params,don2)
                flux_array = np.array(flux_list,dtype=float)
                flux_avg = np.average(flux_array)
                flux_std = np.std(flux_array)
                if flux_std > 0.5: 
                    raise ValueError("The computed flux along the edges are not all equal! Not steady state!")
                model_flux.append(flux_avg)
        model_flux_array = np.array(model_flux,dtype=float).reshape(-1,4)
        flux_dev_sq_avg = np.average((flux_data_ref - model_flux_array )**2)
        normlizing_factor = np.std(flux_data_ref)**2
        norm_flux_dev = flux_dev_sq_avg/normlizing_factor
        return norm_flux_dev

    def calc_qslow_na_dev_from_ref(self,all_k_params,don2,fraction,cnos=[50,100,200,400],concentrations=[9999,0,0,0,0,0,0],\
                                   init_states_conc=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),holding_v=-40.0,temp=298.0):
        """
            calculating deviation of slow charge movement in external Na+ binding
            Initially everything is in (Na)3E1~ATP
        """ 
        vmps, df = self.read_exp_qslow_na()
        df_isna = df.isna()   # try to identify Nans
        col_names = df.columns[1:]
        qslow_list,qslow_ref_list = [],[]
        for i,cno in enumerate(cnos):
            col_name = col_names[i]
            # states concentration at holding voltage
            concentrations[0] = cno
            holding_states_conc = self.calc_pump_states_concentrations(temp,holding_v,concentrations,all_k_params,don2,init_states_conc,fraction)        
            for vmp in vmps:
                ## Check if the value is nan in the df
                cell_is_na = df_isna[df['V in mV'] == vmp][col_name]
                if cell_is_na.item() == True: 
                    continue # skip this calculation because there's no experimental measurements here
                else:
                    fin_states_conc = self.calc_pump_states_concentrations(temp,vmp,concentrations,all_k_params,don2,init_states_conc,fraction)
                    qslow = self.calc_slow_charge_ext_na(holding_states_conc,fin_states_conc,don2)
                    qslow_ref = df[df['V in mV'] == vmp][col_name]
                    qslow_list.append(qslow)
                    qslow_ref_list.append(qslow_ref)
        qslow_array = np.array(qslow_list,dtype=float)
        qslow_ref_array = np.array(qslow_ref_list,dtype=float)
        qslow_dev_sq_avg = np.average((qslow_array-qslow_ref_array)**2)
        normalizing_factor = np.std(qslow_ref_array)**2
        norm_qslow_dev = qslow_dev_sq_avg/normalizing_factor
        return norm_qslow_dev

    def calc_qslow_k_dev_from_ref(self,all_k_params,don2,fraction,ckos=[1,2,4,8],concentrations=[0,0,999,0,0,0,25.0],\
                                   init_states_conc=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),holding_v=0.0,temp=298.0):
        """
            calculating deviation of slow charge movement in external K+ binding
            Initially everything is in E2(K2)
        """ 
        vmps, df = self.read_exp_qslow_k()
        df_isna = df.isna()   # try to identify NaNs
        col_names = df.columns[1:]
        qslow_list,qslow_ref_list = [],[]
        for i,cko in enumerate(ckos):
            col_name = col_names[i]
            # states concentration at holding voltage
            concentrations[2] = cko
            holding_states_conc = self.calc_pump_states_concentrations(temp,holding_v,concentrations,all_k_params,don2,init_states_conc,fraction)        
            for vmp in vmps:
                ## Check if the value is nan in the df
                cell_is_na = df_isna[df['V in mV'] == vmp][col_name]
                if cell_is_na.item() == True: 
                    continue # skip this calculation because there's no experimental measurements here
                else:
                    fin_states_conc = self.calc_pump_states_concentrations(temp,vmp,concentrations,all_k_params,don2,init_states_conc,fraction)
                    qslow = self.calc_slow_charge_ext_k(holding_states_conc,fin_states_conc)
                    qslow_ref = df[df['V in mV'] == vmp][col_name]
                    qslow_list.append(qslow)
                    qslow_ref_list.append(qslow_ref)
        qslow_array = np.array(qslow_list,dtype=float)
        qslow_ref_array = np.array(qslow_ref_list,dtype=float)
        qslow_dev_sq_avg = np.average((qslow_array-qslow_ref_array)**2)
        normalizing_factor = np.std(qslow_ref_array)**2
        norm_qslow_dev = qslow_dev_sq_avg/normalizing_factor
        return norm_qslow_dev

    def monte_carlo_cycles(self,total_steps,tolerance,guessed_ks,guessed_delta,fraction,log_csv_name,tfactor=0.1):
        """
           To run MC sampling
        """

        ## calculate error use the guessed ks and gating charges
        energy_dev = self.calc_energy_dev_from_ref(guessed_ks,guessed_delta)
        flux_dev   = self.calc_flux_dev_from_ref(guessed_ks,guessed_delta,fraction) 
        qslow_na_dev = self.calc_qslow_na_dev_from_ref(guessed_ks,guessed_delta,fraction)
        qslow_k_dev  = self.calc_qslow_k_dev_from_ref(guessed_ks,guessed_delta,fraction)
        prev_total_dev = energy_dev + flux_dev + qslow_na_dev + qslow_k_dev

        #print(energy_dev, flux_dev, qslow_na_dev, qslow_k_dev)

        record_list = []

        record = [0, True, round(prev_total_dev,2)]
        record.extend([ele for ele in guessed_ks])
        record.extend([guessed_delta])
        record_list.append(record)

        istep = 1
        while istep <= total_steps:

            if prev_total_dev <= tolerance: break

            new_all_k_params, new_don2 = self.random_move(guessed_ks,guessed_delta)
            #print(new_all_k_params,new_don2)
            new_energy_dev = self.calc_energy_dev_from_ref(new_all_k_params,new_don2)
            new_flux_dev   = self.calc_flux_dev_from_ref(new_all_k_params,new_don2,fraction) 
            new_qslow_na_dev = self.calc_qslow_na_dev_from_ref(new_all_k_params,new_don2,fraction)
            new_qslow_k_dev  = self.calc_qslow_k_dev_from_ref(new_all_k_params,new_don2,fraction)
            total_dev = new_energy_dev + new_flux_dev + new_qslow_na_dev + new_qslow_k_dev
            
            if total_dev <= prev_total_dev: 
                is_accepted = True
            else:
                rand_nbr =  random.uniform(0,1)
                prob = math.exp(-(total_dev-prev_total_dev)/tfactor)
                if rand_nbr < prob:
                    is_accepted = True
                else: is_accepted = False

            # if accepted, update the ks and the don2 and prev_total_dev
            if is_accepted == True: 
                guessed_ks, guessed_delta = new_all_k_params, new_don2
                prev_total_dev = total_dev
            
            record = [istep, is_accepted, round(total_dev,2)]
            record.extend([ele for ele in guessed_ks])
            record.extend([guessed_delta])
            record_list.append(record)

            istep += 1
        
        df = pd.DataFrame(record_list, columns =['MC_step', 'Is_accepted', 'total_deviation','kp1c','kn1c','kp2','kn2','kp3',\
        'kn3','kp4b','kn4b','kp4a','kn4a','kp5a','kn5a','kp5b','kn5b','kp6','kn6','kpa','kna','kp7 ','kn7','kdn1','kdn2',\
        'kdk1','kdk2','don2'])
        df.to_csv(log_csv_name)
        return True


        




