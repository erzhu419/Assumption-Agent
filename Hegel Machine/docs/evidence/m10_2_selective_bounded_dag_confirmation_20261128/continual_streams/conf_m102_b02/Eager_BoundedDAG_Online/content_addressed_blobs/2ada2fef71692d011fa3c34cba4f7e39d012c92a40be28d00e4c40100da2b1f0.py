import numpy as np

def run_node(inputs: dict) -> dict:
    design=inputs['node_00']['trajectory_design']; time=inputs['node_01']; m=time['modes']; n=len(m)
    w=np.linalg.lstsq(design[:n],m,rcond=1e-10)[0]
    return {'state_pred':(design[n:]@w)@time['basis'].T}
