import sys,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,p):
    n=len(k); face=(k[1:]+k[:-1])/(2*h*h)
    lo=-face.copy(); hi=-face.copy()
    lo[:-1]-=np.maximum(v[1:-1],0)/h; hi[1:]-=np.maximum(-v[1:-1],0)/h
    hi[0]=-k[0]/h**2*p[0]; lo[-1]=-k[-1]/h**2*p[3]
    wrap0=k[0]/h**2*p[2]; wrap1=k[-1]/h**2*p[5]
    diag=np.r_[-hi[0]+wrap0+r/h*p[1],-lo[:-1]-hi[1:],-lo[-1]+wrap1+r/h*p[4]]
    A=sparse.diags([lo,diag,hi],[-1,0,1],format='csc')
    return A+sparse.csc_matrix(([-wrap0,-wrap1],([0,n-1],[n-1,0])),shape=(n,n))

def forward(s,p,modes=None):
    n,m=s['nx'],s['ny']
    a=axis(s['kappa_x'],s['vel_x'],s['Lx']/(n-1),s['robin_x'],p[:6]); b=axis(s['kappa_y'],s['vel_y'],s['Ly']/(m-1),s['robin_y'],p[6:])
    I=sparse.eye(n*m,format='csc'); L=(sparse.kron(sparse.eye(m),a)+sparse.kron(b,sparse.eye(n))+sparse.diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    rhs=np.column_stack([x.ravel(order='F') for x in (s['source_modes'] if modes is None else modes)])
    idx=s['sensor_ij'][:,0]+n*s['sensor_ij'][:,1]
    obs=[]
    for shift in s['mu_shifts']:
        u=splu(L+shift*I).solve(rhs); obs.append(u[idx])
    return np.vstack(obs),u.T.reshape((-1,m,n)).transpose(0,2,1)

if __name__=='__main__':
    i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; d=i['campaign_diagnostics']; W=np.stack([x['w_pulse'] for x in d]); modes=np.einsum('ab,bij->aij',W,s['source_modes']); Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=x['y_pulse_raw']))['y_obs_raw'] for x in d]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0];
    def errors(p): return forward(s,p,modes)[0]-Y
    def stats(e):return np.sqrt(np.mean(e**2,axis=0)).round(9)
    for name,t in [('Robin',[1,1,0]),('reflecting',[1,0,0]),('periodic',[1,0,1]),('half Robin',[1,.5,0]),('double diffusion',[2,1,0])]:
        e=errors(t*4); print(name,stats(e),stats(e-Q@(Q.T@e)))
    for nt,cols in [('loss',[1,4,7,10]),('loss/diff',[0,1,3,4,6,7,9,10]),('loss/diff/wrap',list(range(12)))]:
        start=np.tile([1.,.5,0.],4); low=np.tile([.0,0.,0.],4); high=np.tile([6.,20.,5.],4)
        def expand(x):
            p=start.copy(); p[cols]=x; return p
        for hold in [-1,0,1,2]:
            train=[j for j in range(3) if j!=hold]; fit=least_squares(lambda x:errors(expand(x))[:,train].ravel(),start[cols],bounds=(low[cols],high[cols]),gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=200); p=expand(fit.x); print(nt,'hold',hold,'p',p.round(6),'rms',stats(errors(p)))
    print('pulse magnitude',stats(Y))
