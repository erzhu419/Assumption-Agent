import sys,pickle,numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
nx,ny=s['nx'],s['ny']; ix=sp.eye(nx,format='csc'); iy=sp.eye(ny,format='csc'); eye=sp.eye(nx*ny,format='csc')
ijk=s['sensor_ij']; sensors=ijk[:,0]+nx*ijk[:,1]
modes=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1)
ps=s['reference_tracer_pulses']; rhs=modes@np.stack([p['w_pulse'] for p in ps],axis=1); target=np.column_stack([p['y_pulse_raw'] for p in ps]); z=s['Z_nuisance']; proj=z@np.linalg.pinv(z)

def axis(q,suffix):
    k=s['kappa_'+suffix]; v=s['vel_'+suffix]; h=s['L'+suffix]/(len(k)-1); rob=s['robin_'+suffix]
    faces=(k[:-1]+k[1:])/(2*h*h); lo=-faces[:-1]-np.maximum(v[1:-1],0)/h; hi=-faces[1:]-np.maximum(-v[1:-1],0)/h
    ds=q[:2]*k[[0,-1]]/h**2; leaks=q[2:4]*rob/h; wraps=q[4:]*k[[0,-1]]/h**2
    a=sp.diags((np.r_[lo,-ds[1]],np.r_[ds[0]+leaks[0]+wraps[0],-lo-hi,ds[1]+leaks[1]+wraps[1]],np.r_[-ds[0],hi]),(-1,0,1),format='lil'); a[0,-1]=-wraps[0]; a[-1,0]=-wraps[1]; return a.tocsc()

def predict(q,rhs=rhs):
    a=sp.kron(iy,axis(q[:6],'x'),format='csc')+sp.kron(axis(q[6:],'y'),ix,format='csc')+sp.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    return np.concatenate([splu(a+mu*eye).solve(rhs)[sensors] for mu in s['mu_shifts']])

def report(name,e):
    print(name,'pulse rms',np.sqrt(np.mean(e**2,axis=0)).round(9),'shift rms',np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1)).round(9),'nuisance fraction',round(np.linalg.norm(proj@e)/max(np.linalg.norm(e),1e-15),5),flush=True)

if __name__=='__main__':
    for x in (0,1):
      for y in (0,1):
        report(f'Robin {x,y}',predict(np.array([1,1,x,x,0,0,1,1,y,y,0,0]))-target)
    b=np.array([1.,1.,0.4,0.4,0,0,1.,1.,0.4,0.4,0,0]); inds=[[2,3,8,9],[0,1,2,3,6,7,8,9],list(range(12))]
    for ids in inds:
      for nuisance in (False,True):
        for hold in (-1,0,1,2):
          keep=[j for j in range(3) if j!=hold]
          def expand(v):
            q=b.copy(); q[ids]=v; return q
          def err(v):
            e=predict(expand(v))-target
            return e-proj@e if nuisance else e
          f=least_squares(lambda v: err(v)[:,keep].ravel(),b[ids],bounds=(-0.25,8),ftol=1e-12,gtol=1e-12,xtol=1e-12)
          print('fit',len(ids),nuisance,hold,expand(f.x).round(7))
          report(' residual',err(f.x))
