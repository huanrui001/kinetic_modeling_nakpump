import os,sys,math
import random
import re
import numpy as np
from scipy.integrate import odeint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
from kinetic_modeling_tools import kinetic_modeling 

km = kinetic_modeling()

def parse_args():
    parser = ArgumentParser(description='''
    Written by Huan Rui, 2021
    This program will ---- TBA
    # A set of ODEs to solve and get the concentration of E1/E2                                                                                                        
    # x(0) : E1.ATP                                                                                                                                                    
    # x(1) : Na3.E1.ATP                                                                                                                                                
    # x(2) : (Na3).E1~P                                                                                                                                                
    # x(3) : E2-P.Na2                                                                                                                                                  
    # x(4) : E2-P.Na                                                                                                                                                   
    # x(5) : E2-P                                                                                                                                                      
    # x(6) : E2-P.K                                                                                                                                                    
    # x(7) : E2-P.K2                                                                                                                                                   
    # x(8) : E2(K2)                                                                                                                                                    
    # x(9): E2(K2).ATP              

    ''', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-f',
                        '--fraction',
                        help='Use the last xx fraction of the time series to calculate properties',
                        default=0.3)
    parser.add_argument('-kg',
                        '--guessed_ks',
                        nargs=24,
                        metavar=('kp1c','kn1c','kp2','kn2','kp3','kn3','kp4b','kn4b','kp4a','kn4a','kp5a','kn5a','kp5b','kn5b',\
                                 'kp6','kn6','kpa','kna','kp7','kn7','keqn1','keqn2','keqk1','keqk2'),
                        type=float,
                        help='initial guess of the on/off rates in the order of:\
                            [ kp1c,kn1c,kp2,kn2,kp3,kn3,kp4b,kn4b,kp4a,kn4a,kp5a,kn5a,kp5b,kn5b,\
                              kp6,kn6,kpa,kna,kp7,kn7,keqn1,keqn2,keqk1,keqk2 ]')
    parser.add_argument('-dg',
                        '--guessed_delta',
                        type=float,
                        help='Guessed value of don2',
                        default=0.27 )
    parser.add_argument('-n',
                        '--total_steps',
                        type=int,
                        help='total monte carlo steps for run',
                        default=10)
    parser.add_argument('-e',
                        '--tolerance',
                        type=float,
                        help='tolerance - stopping criteria',
                        default=0.4)
    parser.add_argument('-l',
                        '--log_csv_name',
                        dest='log_csv_name',
                        help='log file name csv to store all the ks from each accepted run',
                        default='monte_carlo_run_log.csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    fraction = args.fraction
    guessed_ks = args.guessed_ks
    guessed_delta = args.guessed_delta
    total_steps = args.total_steps
    tolerance = args.tolerance
    log_csv_name = args.log_csv_name

    km.monte_carlo_cycles(total_steps,tolerance,guessed_ks,guessed_delta,fraction,log_csv_name)
    

if __name__ == "__main__":
    main()
