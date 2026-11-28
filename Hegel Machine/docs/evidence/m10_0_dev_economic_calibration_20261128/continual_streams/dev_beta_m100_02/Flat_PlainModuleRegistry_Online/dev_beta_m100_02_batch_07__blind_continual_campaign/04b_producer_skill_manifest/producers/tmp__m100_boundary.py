import pickle, json, sys
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
from itertools import product
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,kind):
    n=len(k); face=(k[1:]+k[:-1])/(2*h*h)
    lo=-face-np.maximum(v[1:],0)/h; hi=-face-np.maximum(-v[:-1],0)/h
    if kind in ('P','PR'):
        wrap=(k[0]+k[-1])/(2*h*h)
        wl=wrap+max(v[0],0)/h; wr=wrap+max(-v[-1],0)/h
    else:
        wl=wr=0.
        hi[0]=-k[0]/h**2; lo[-1]=-k[-1]/h**2
    diag=np.r_[-hi[0]+wl,-lo[:-1]-hi[1:],-lo[-1]+wr]
    if kind in ('R','PR','RN'): diag[[0,-1]]+=r/h if kind!='RN' else [r/h,0.]
    if kind=='NR': diag[-1]+=r/h
    if kind=='D': diag[[0,-1]]+=k[[0,-1]]/h**2
    return diags([lo,diag,hi],[-1,0,1],format='csc')+csc_matrix(([-wl,-wr],([0,n-1],[n-1,0])),shape=(n,n))

def responses(s,kind,fields=False):
    nx,ny=s['nx'],s['ny']; m=len(s['source_modes']); ix,iy=np.asarray(s['sensor_ij']).T
    a,b=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(n-1),s['robin_'+d],k) for d,n,k in zip('xy',(nx,ny),kind)]
    L=kron(eye(ny),a,format='csc')+kron(b,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    rhs=s['source_modes'].transpose(2,1,0).reshape(nx*ny,m)
    out=[]
    for mu in s['mu_shifts']:
        u=splu(L+mu*eye(nx*ny,format='csc')).solve(rhs); out.append(u[ix+nx*iy])
    return (np.vstack(out),u.T.reshape(m,ny,nx).transpose(0,2,1)) if fields else np.vstack(out)

if __name__=='__main__':
    base='/holdout_task/'
    item=pickle.load(open(base+'public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open(base+'_upstream_param_payload.json'))
    Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in item['campaign_diagnostics']]); W=np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']]); Z=s['Z_nuisance']; print('raw calibrated pulse range',Y.min(),Y.max())
    scored=[]
    for kinds in product(('R','N','P','PR','RN','NR','D'),repeat=2):
        err=responses(s,kinds)@W-Y
        res=err-Z@np.linalg.lstsq(Z,err,rcond=None)[0]
        scored.append((np.linalg.norm(err)/np.sqrt(err.size),kinds,np.sqrt(np.mean(err**2,axis=0)),np.linalg.norm(res)/np.sqrt(res.size)))
    for row in sorted(scored)[:15]: print(row)
    print('baseline',next(x for x in scored if x[1]==('R','R')))
    for i in range(Y.shape[1]):
        winner=min(scored,key=lambda x:np.mean(np.delete(x[2],i)**2)); print('LOPO',i,winner[1],winner[2][i])
