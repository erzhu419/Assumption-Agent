import numpy as np


def run_node(inputs:dict)->dict:
    a=inputs['node_00']; t=inputs['node_01']; modes=t['modes']; n=len(modes)
    x=np.arcsinh(2*np.c_[a['b'],a['c']]); dot=x@x.T/x.shape[1]
    d=np.maximum(np.diag(dot)[:,None]+np.diag(dot)-2*dot,0.)
    k=(1+dot)**3+np.exp(-.05*d)
    return {'state_pred':(k[n:,:n]@np.linalg.solve(k[:n,:n]+.002*np.eye(n),modes))@t['basis'].T}
