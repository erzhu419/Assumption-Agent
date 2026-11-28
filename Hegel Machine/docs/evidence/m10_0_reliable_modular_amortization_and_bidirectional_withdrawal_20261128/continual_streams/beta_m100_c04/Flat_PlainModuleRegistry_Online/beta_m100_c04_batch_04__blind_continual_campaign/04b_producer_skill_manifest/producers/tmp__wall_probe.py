import sys,pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,p):
    # p = (left coupling factor, right coupling factor, left Robin factor, right Robin factor)
    q=(k[1:]+k[:-1])*(.5/h**2)
    a=-q.copy(); b=-q.copy()
    a[:-1]-=np.maximum(v[1:-1],0)/h
    b[1:]-=np.maximum(-v[1:-1],0)/h
    b[0]=-p[0]*k[0]/h**2; a[-1]=-p[1]*k[-1]/h**2
    d=np.r_[-b[0]+p[2]*r/h,-a[:-1]-b[1:],-a[-1]+p[3]*r/h]
    return sparse.diags((a,d,b),(-1,0,1),format='csc')

def matrices(s,p):
    nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
    ax,ay=[axis(s['kappa_'+j],s['vel_'+j],s['L'+j]/(n-1),s['robin_'+j],q) for j,n,q in zip('xy',(nx,ny),(p[:4],p[4:]))]
    L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    eye=sparse.eye(nx*ny,format='csc'); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
    obs=[]
    for shift in s['mu_shifts']:
        field=splu(L+shift*eye).solve(rhs)
        obs.append(field[rows])
    return np.vstack(obs),field.T.reshape(-1,ny,nx).transpose(0,2,1)

def load():
    i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s=calibrate_upstream(dict(i['study'],reference_tracer_pulses=i['campaign_diagnostics']))
    w,y=(np.array([row[k] for row in s['reference_tracer_pulses']]).T for k in ('w_pulse','y_pulse_raw'))
    return s,w,y

if __name__=='__main__':
    s,w,y=load(); np.set_printoptions(precision=7,suppress=True)
    for name,p in [('open',[1]*8),('sealed',[1,1,0,0]*2),('double+open',[2,2,1,1]*2),('double+sealed',[2,2,0,0]*2)]:
        err=matrices(s,p)[0]@w-y
        print(name,'rel by pulse',np.linalg.norm(err,axis=0)/np.linalg.norm(y,axis=0))
    for held in [-1,0,1,2]:
        tr=[i for i in range(3) if i!=held]
        fit=least_squares(lambda p:(matrices(s,p)[0]@w[:,tr]-y[:,tr]).ravel(),[1]*8,bounds=(0,30),gtol=1e-12)
        err=matrices(s,fit.x)[0]@w-y
        print('held',held,'p',fit.x,'rel',np.linalg.norm(err,axis=0)/np.linalg.norm(y,axis=0),'jac singulars',np.linalg.svd(fit.jac,compute_uv=False))
    print('aux mean and std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
