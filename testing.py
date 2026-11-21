from linear import PropagateSatelliteLinearMan
from second_order_2comp import PropagateSatelliteSecondManProj
import numpy as np

P_0_std = np.diag([10**2]*3 + [1e-3**2]*3 + [5e-3**2]*3 + [50**2])
P_0_sparse = np.diag([10**2]*3 + [1e-3**2]*3 + [5e-3**2]*3 + [10**2])

rdm_seed = 27

# -------- LINEAR TEST CASES --------
def standard_case_linear():
    propagator = PropagateSatelliteLinearMan(180, 10, P_0_init=P_0_std, rdm_seed=rdm_seed, record=True)
    propagator.do_calc()

def sparse_case_linear():
    propagator = PropagateSatelliteLinearMan(10, 180, P_0_init=P_0_sparse, rdm_seed=rdm_seed, t1_guess = 955, record=True)
    propagator.do_calc()

def short_arc_case_linear():
    propagator = PropagateSatelliteLinearMan(150, 2, P_0_init=P_0_std, rdm_seed=rdm_seed, t1_truth=151, t1_guess=161, t2=300, record=True)
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
        dv_guess=(0.005, -1.4723091, 0.005),
        rdm_seed=rdm_seed
    )
    propagator.do_calc()

# -------- SECOND ORDER TEST CASES --------
def standard_case():
    propagator = PropagateSatelliteSecondManProj(180, 10, P_0_init=P_0_std, rdm_seed=rdm_seed, record=True, project = False)
    propagator.do_calc()

def standard_case_proj():
    propagator = PropagateSatelliteSecondManProj(180, 10, P_0_init=P_0_std, rdm_seed=rdm_seed, record=True)
    propagator.do_calc()

def sparse_case():
    propagator = PropagateSatelliteSecondManProj(10, 180, P_0_init=P_0_sparse, t1_guess = 955, rdm_seed=22, record=True, project = False)
    propagator.do_calc()

def sparse_case_proj():
    propagator = PropagateSatelliteSecondManProj(10, 180, P_0_init=P_0_sparse, t1_guess = 955, rdm_seed=22, record=True)
    propagator.do_calc()

def short_arc_case():
    propagator = PropagateSatelliteSecondManProj(150, 2, P_0_init=P_0_std, t1_truth=151, t1_guess=161, t2=300, rdm_seed=rdm_seed, record=True, project = False)
    propagator.do_calc()

def short_arc_case_proj():
    propagator = PropagateSatelliteSecondManProj(150, 2, P_0_init=P_0_std, t1_truth=151, t1_guess=161, t2=300, rdm_seed=rdm_seed, record=True)
    propagator.do_calc()

def geo_case_second():
    propagator = PropagateSatelliteSecondManProj(
        300, 60,                                        
        P_0_init=P_0_std,
        observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
        target_coe  =(17993.0215, 0.7301, 0, 0, 0, 170),
        t1_truth=4485.25, 
        t1_guess=4535.25,
        dv_truth=(0.0,   -1.4773091, 0.0),
        dv_guess=(0.005, -1.4723091, 0.005),
        rdm_seed=rdm_seed,
        project = False
    )
    propagator.do_calc()

def geo_case_second_proj():
    propagator = PropagateSatelliteSecondManProj(
        300, 60,                                        
        P_0_init=P_0_std,
        observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
        target_coe  =(17993.0215, 0.7301, 0, 0, 0, 170),
        t1_truth=4485.25, 
        t1_guess=4535.25,
        dv_truth=(0.0,   -1.4773091, 0.0),
        dv_guess=(0.005, -1.4723091, 0.005),
        rdm_seed=rdm_seed,
        project = False
    )
    propagator.do_calc()

if __name__ == "__main__":
    print("CORRECTED VERSION")
    print("-------- STANDARD CASE --------\n")
    print("LINEAR")
    standard_case_linear()

    print("\nSECOND ORDER")
    standard_case()

    print("\nSECOND ORDER PROJ")
    standard_case_proj()

    # print("-------- SPARSE CASE --------\n")
    # print("LINEAR")
    # sparse_case_linear()

    # print("\nSECOND ORDER")
    # sparse_case()

    # print("\nSECOND ORDER PROJ")
    # sparse_case_proj()
    
    # print("-------- SHORT ARC CASE --------\n")
    # print("LINEAR")
    # short_arc_case_linear()

    # print("\nSECOND ORDER")
    # short_arc_case()

    # print("\nSECOND ORDER PROJ")
    # short_arc_case_proj()

    # print("-------- GEO CASE --------\n")
    # print("LINEAR")
    # geo_case_linear()

    # print("\nSECOND ORDER PROJ")
    # geo_case_second_proj()

    # print("\nSECOND ORDER")
    # geo_case_second()


    