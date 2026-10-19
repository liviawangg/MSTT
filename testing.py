from linear import PropagateSatelliteLinearMan
from second_order import PropagateSatelliteSecondMan
import numpy as np

# RUN THE ALGORITHMS FROM HERE
#Input format: number of points, time between points, record data?

P_0_std = np.diag([10**2]*3 + [1e-3**2]*3 + [5e-3**2]*3 + [50**2])
P_0_sparse = np.diag([10**2]*3 + [1e-3**2]*3 + [5e-3**2]*3 + [10**2])

# -------- LINEAR TEST CASES --------
def standard_case_linear():
    propagator = PropagateSatelliteLinearMan(180, 10, P_0_init=P_0_std, record=True)
    propagator.do_calc()

def sparse_case_linear():
    propagator = PropagateSatelliteLinearMan(10, 180, P_0_init=P_0_sparse, t1_guess = 915, record=True)
    propagator.do_calc()

def short_arc_case_linear():
    propagator = PropagateSatelliteLinearMan(150, 2, P_0_init=P_0_std, t1_truth=151, t1_guess=161, t2=300, record=True)
    propagator.do_calc()

def geo_case_linear():
    # t2 = 18000
    propagator = PropagateSatelliteLinearMan(
        300, 60,                                        
        P_0_init=P_0_std,
        observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
        target_coe  =(17993.0215, 0.7301, 0, 0, 0, 170),
        t1_truth=4485.25, 
        t1_guess=4535.25,
        dv_truth=(0.0,   -1.4773091, 0.0),
        dv_guess=(0.005, -1.4723091, 0.005)
    )
    propagator.do_calc()

# -------- SECOND ORDER TEST CASES --------
def standard_case():
    propagator = PropagateSatelliteSecondMan(180, 10, P_0_init=P_0_std, record=True)
    propagator.do_calc()

def sparse_case():
    propagator = PropagateSatelliteSecondMan(10, 180, P_0_init=P_0_sparse, t1_guess = 955, rdm_seed=22, record=True)
    propagator.do_calc()

def short_arc_case():
    propagator = PropagateSatelliteSecondMan(150, 2, P_0_init=P_0_std, t1_truth=151, t1_guess=161, t2=300, record=True)
    propagator.do_calc()

def geo_case_second():
    # t2 = 18000
    propagator = PropagateSatelliteSecondMan(
        300, 60,                                        
        P_0_init=P_0_std,
        observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
        target_coe  =(17993.0215, 0.7301, 0, 0, 0, 170),
        t1_truth=4485.25, 
        t1_guess=4535.25,
        dv_truth=(0.0,   -1.4773091, 0.0),
        dv_guess=(0.005, -1.4723091, 0.005)
    )
    propagator.do_calc()

# CHANGE THESE TO RUN EACH TEST
if __name__ == "__main__":
    print("-------- LINEAR TESTS --------\n")
    # print("Running linear standard case...")
    # standard_case_linear()

    print("Running linear sparse case...")
    sparse_case_linear()

    # print("Running linear short arc case...")
    # short_arc_case_linear()

    # print("Running linear GEO case...")
    # geo_case_linear()


    print("-------- SECOND ORDER TESTS --------\n")
    # print("Running second order standard case...")
    # standard_case()
    
    print("Running second order sparse case...")
    sparse_case()
    
    # print("Running second order short arc case...")
    # short_arc_case()

    # print("Running second order GEO case...")
    # geo_case_second()
    