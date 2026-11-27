import sys
sys.path.insert(0,'/holdout_task')
import pickle
from itertools import product
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream

def axis(s,a,ends):
    k,v=s['kappa_'+a],s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1)
    A=np.zeros((n,n)); ii=np.arange(1,n-1)
    def up(i):
        l,r=(i-1)%n,(i+1)%n
        m=(k[i]+k[l])/(2*h*h)+np.maximum(v[i],0)/h
        p=(k[i]+k[r])/(2*h*h)+np.maximum(-v[i],0)/h
        A[i,l]=-m; A[i,r]=-p; A[i,i]=m+p
    up(ii)
    for side,bc in enumerate(ends):
        i=(n-1)*side; j=1 if side==0 else n-2
        if bc=='P': up(i)
        else:
            A[i,j]=-k[i]/h**2; A[i,i]=-A[i,j]+(0 if bc=='N' else (s['robin_'+a] if bc=='R' else bc)/h)
    return csc_matrix(A)

def forward(s,bcs):
    nx,ny=s['nx'],s['ny']; ax,ay=[axis(s,a,b) for a,b in zip('xy',bcs)]
    L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); I=eye(nx*ny,format='csc')
    rhs=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); i,j=s['sensor_ij'].T
    return np.vstack([splu(L+t*I).solve(rhs)[i+nx*j] for t in s['mu_shifts']])

if __name__=='__main__':
    with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
    s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
    W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']
    q=np.linalg.qr(Z,mode='reduced')[0]
    scores=[]
    for b in product('RNP',repeat=4):
        r=(forward(s,(b[:2],b[2:]))@W-Y); proj=r-q@(q.T@r)
        rm=np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1,W.shape[1])**2,axis=1)); scores.append((np.sqrt(np.mean(r*r)),''.join(b),np.sqrt(np.mean(proj*proj)),rm.max(),rm.round(7).tolist()))
    for row in sorted(scores)[:12]: print(row)
    print('uniform candidates',[(b,r,p,ws) for r,b,p,ws,_ in scores if len(set(b))==1])
