"""Fit scalar and dynamical readouts on the calibrated operator features."""
import numpy as np


def run_node(inputs: dict) -> dict:
    node=inputs['node_00']; z,c,d=[node[k] for k in ('coordinates','soft','curvature')]
    readout=inputs['node_01']; targets,basis=readout['targets'],readout['basis']
    baseline=inputs['node_02']['fit']; n=len(targets)
    a=np.c_[np.ones(len(c)),c,d,(c*d).sum(1),c[:,0]*d[:,1]]
    pred=a[n:]@np.linalg.lstsq(a[:n],targets,rcond=1e-9)[0]
    pred=.99*pred+.01*baseline
    y=pred[:,0]; state=(basis@np.linalg.lstsq(basis,pred[:,1:].T,rcond=1e-9)[0]).T
    end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
