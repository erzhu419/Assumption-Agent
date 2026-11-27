import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m99_c02_s00_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; pulses=a['campaign_diagnostics']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]);
def response(z):
    axes=[]
    for j,t in enumerate('xy'):
        n=s['n'+t]; h=s['L'+t]/(n-1); k=s['kappa_'+t]; v=s['vel_'+t]; r=s['robin_'+t]
        A=m._axis(k,v,h,r).toarray()
        for side,row,col,wrap in [(0,0,1,n-1),(1,n-1,n-2,0)]:
            d,b,c=z[j,side]; A[row,row]=d*k[row]/h**2+b*r/h+c*k[row]/h**2; A[row,col]=-d*k[row]/h**2; A[row,wrap]=-c*k[row]/h**2
        axes.append(csc_matrix(A))
    return m._responses(s,axes)[0] @ W

def metric(e):
    e=e.reshape(len(s['mu_shifts']),len(s['sensor_ij']),len(pulses)); return np.max(np.sqrt(np.mean(e**2,axis=1))),np.max(np.abs(e))
for name,row in [('robin',[1,1,0]),('reflecting',[1,0,0]),('half_cell_reflecting',[2,0,0]),('periodic',[1,0,1])]:
    z=np.tile(row,(2,2,1)); e=response(z)-Y; print(name,metric(e), 'signed by shift',e.reshape(3,-1).mean(axis=1))
for wrap in [False,True]:
    for hold in range(3):
        fit=[i for i in range(3) if i!=hold]; init=np.tile([1.,0.,0.],(2,2,1)); mask=np.ones_like(init,dtype=bool); mask[:,:,2]=wrap
        def build(b):
            out=init.copy(); out[mask]=b; return out
        def f(b): return (response(build(b))-Y)[:,fit].ravel()
        opt=least_squares(f,init[mask],bounds=(np.tile([0.05,0,0],(2,2,1))[mask],np.tile([10,5,5],(2,2,1))[mask]),gtol=1e-13,ftol=1e-12,xtol=1e-12)
        z=build(opt.x); err=response(z)-Y; print('fit wrap',wrap,'hold',hold,'params [neighbor diffusion, Robin, wrap] xlo xhi ylo yhi',z.reshape(-1,3).round(6).tolist(),'training RMS',np.sqrt(np.mean(opt.fun**2)),'held max shift RMS',np.max(np.sqrt(np.mean(err[:,hold].reshape(3,-1)**2,axis=1))))
