import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags,eye,kron,coo_matrix
from scipy.sparse.linalg import splu
from model98 import _calibration,_axis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=_calibration(s['sensor_calibration_sweeps']);p=s['reference_tracer_pulses']; y=np.column_stack([f(q['y_pulse_raw']) for q in p]);nx,ny=s['nx'],s['ny']; ids=s['sensor_ij']@np.array([1,nx]); modes=np.column_stack([np.tensordot(q['w_pulse'],s['source_modes'],axes=1).ravel(order='F') for q in p]); shifts=s['mu_shifts']; ns=len(ids)

def response(w, modes=modes):
    axes=[]
    for a,par in zip('xy',w.reshape(2,2,5)):
        k=s['kappa_'+a];v=s['vel_'+a];r=s['robin_'+a];n=len(k);h=s['L'+a]/(n-1)
        leak,inner,wrap,second,stretch=par.T
        side=[0,n-1]; scale=k[side]/h**2
        A=_axis(k,v,h,r,[*leak,*inner]); rows=np.repeat(side,4); cols=np.array([0,n-1,2,1,n-1,0,n-3,n-2]);
        diag=(wrap+second)*scale; vals=np.column_stack([diag,-wrap*scale,-second*scale,np.zeros(2)]).ravel(); A=A+coo_matrix((vals,(rows,cols)),shape=(n,n));axes.append(A)
    # Stretch permits a geometrical rescaling of the tangential operator on each wall.
    ax,ay=axes; stretch=w.reshape(4,5)[:,4]; sx=diags(np.r_[stretch[0],np.zeros(nx-2),stretch[1]]);sy=diags(np.r_[stretch[2],np.zeros(ny-2),stretch[3]])
    A=kron(eye(ny)+sy,ax)+kron(ay,eye(nx)+sx)+diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc')
    return np.vstack([splu((A+mu*I).tocsc()).solve(modes)[ids] for mu in shifts])

if __name__=='__main__':
 from scipy.linalg import svdvals
 families={'wrap':[0,1,2],'second':[0,1,3],'tangent':[0,1,4],'all':[0,1,2,3,4]}
 for name,cols in families.items():
  for paired in (True,False):
   n=2 if paired else 4
   base=np.tile([0.,1.,0.,0.,0.],(4,1)); mask=np.zeros((n,5),bool);mask[:,cols]=True
   l=np.tile([-.65,.05,-1.,-1.,-.85],(n,1))[mask];u=np.tile([4.,6.,5.,5.,4.],(n,1))[mask]
   def unpack(x):
    z=base[:n].copy(); z[mask]=x;return np.repeat(z,2,axis=0) if paired else z
   def run(train=slice(None)):
    def fun(x): return (response(unpack(x))-y)[:,train].ravel()
    starts=[base[:n][mask].copy()]; starts[0][::len(cols)]=.06
    opt=min((least_squares(fun,q,bounds=(l,u),max_nfev=75,ftol=2e-9) for q in starts),key=lambda q:q.cost)
    e=response(unpack(opt.x))-y
    return unpack(opt.x),np.sqrt(np.mean(e.reshape(3,ns,2)**2,axis=1)).max(axis=0),svdvals(opt.jac)[-1]
   w,e,sv=run(); print(name,paired,'joint worst by pulse',e,'smallest SV',sv,'params',np.round(w,5).tolist(),flush=True)
   if e.max()<.02: print('holdout worst shift',run(0)[1][1],run(1)[1][0],flush=True)
