import itertools, pickle, sys
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,kind):
    n=len(k); i=np.arange(n)
    f=(k+np.roll(k,-1))/(2*h*h)
    a=np.roll(f,1)+np.maximum(v,0)/h
    b=f+np.maximum(-v,0)/h
    A=np.zeros((n,n)); A[i,(i-1)%n]=-a; A[i,(i+1)%n]=-b; A[i,i]=a+b
    if kind!='periodic':
        for e,j in ((0,1),(n-1,n-2)):
            A[e,:]=0
            coef= (0.5*(k[e]+k[j]) if kind.startswith('face') else k[e])/h**2
            if kind=='upwind': coef=(b[e] if e==0 else a[e])
            A[e,j]=-coef
            A[e,e]=coef+(r/h if kind in ('robin','face_robin') else 0)
    return csc_matrix(A)

def load():
    item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
    p=s['reference_tracer_pulses']; W=np.stack([q['w_pulse'] for q in p],axis=1); Y=np.stack([q['y_pulse_raw'].ravel() for q in p],axis=1)
    return s,W,Y

def forward(s,ax,ay,rhs):
    nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc')
    A=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    ij=s['sensor_ij']; ids=ij[:,0]+nx*ij[:,1]
    return np.concatenate([splu(A+mu*I).solve(rhs)[ids] for mu in s['mu_shifts']])

if __name__=='__main__':
    s,W,Y=load(); nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; modes=s['source_modes']; rhs=modes.reshape(len(modes),-1,order='F').T@W
    def get(d,k): return axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d],k)
    kinds=['periodic','robin','sealed','face_robin','face_sealed','upwind']; scores=[]
    for x,y in itertools.product(kinds,repeat=2):
        R=forward(s,get('x',x),get('y',y),rhs)-Y
        Q=R-Z@np.linalg.lstsq(Z,R,rcond=None)[0]
        scores.append((np.sqrt(np.mean(R**2)),x,y,np.sqrt(np.mean(Q**2)),np.round(np.sqrt(np.mean(R.reshape(3,-1,3)**2,axis=1)),7).tolist()))
    print('pulse signal rms',np.sqrt(np.mean(Y**2,axis=0)))
    for row in sorted(scores)[:12]: print(row)
    print('matched laws',[(x,y,round(a,7),round(b,7)) for a,x,y,b,_ in scores if x==y])
