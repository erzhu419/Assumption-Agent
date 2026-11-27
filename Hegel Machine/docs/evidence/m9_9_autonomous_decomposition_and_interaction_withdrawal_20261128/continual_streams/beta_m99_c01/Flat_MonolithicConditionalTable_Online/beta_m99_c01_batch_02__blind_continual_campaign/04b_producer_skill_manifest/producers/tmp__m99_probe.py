import pickle, itertools
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import root
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; n=len(x)
pairs=list(itertools.combinations(range(4),2)); triples=list(itertools.combinations(range(4),3))
def feat(z,kind):
    p=[np.ones(len(z)),*z.T]
    if kind in ('square','quadratic','cubic'): p.extend((z*z).T)
    if kind in ('interaction','quadratic','cubic'): p.extend(z[:,i]*z[:,j] for i,j in pairs)
    if kind=='cubic':
        p.extend((z**3).T)
        p.extend(z[:,i]**2*z[:,j] for i,j in itertools.permutations(range(4),2))
        p.extend(z[:,i]*z[:,j]*z[:,k] for i,j,k in triples)
    return np.column_stack(p)
print('sweep nonzero counts',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('single bank shapes',{k:np.shape(v) for k,v in sw['single_bank_sweeps'].items()})
models={}
for name in ['linear','square','interaction','quadratic','cubic']:
    A=feat(x,name); C=np.linalg.lstsq(A,y,rcond=None)[0]; e=A@C-y
    h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]**2,axis=1)
    loo=e/np.maximum(1-h[:,None],1e-10)
    print(name,'rank',np.linalg.matrix_rank(A), 'rmse',np.sqrt(np.mean(e**2)), 'LOO RMSE',np.sqrt(np.mean(loo**2)), 'max LOO rosette',np.max(np.sqrt(np.mean(loo**2,axis=1))))
    if name in ('linear','quadratic'): print('coeff\n',C.round(6))
    models[name]=C
nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); modes=s['source_modes']; weights=np.array([p['w_pulse'] for p in pulses]).T; rhs=modes.reshape(len(modes),-1,order='F').T@weights
si,sj=s['sensor_ij'].T; sensors=si+nx*sj
raw=np.column_stack([p['y_pulse_raw'] for p in pulses]); z=s['Z_nuisance']; Q=np.linalg.qr(z)[0]; proj=lambda e:e-Q@(Q.T@e)
def axis(k,v,h,r,mode):
    a=(k[:-1]+k[1:])/2/h**2; n=len(k)
    lo=-a[:-1]-np.maximum(v[1:-1],0)/h; hi=-a[1:]-np.maximum(-v[1:-1],0)/h
    mat=diags((np.r_[lo,0],np.r_[0,-lo-hi,0],np.r_[0,hi]),(-1,0,1)).tolil()
    for i,j,f in [(0,1,0),(n-1,n-2,-1)]:
        d=k[i]/h**2
        if mode=='face': d=a[f]
        if mode=='reflect2': d*=2
        mat[i,i]=d+(r/h if mode in ('robin','face') else 0); mat[i,j]=-d
        if mode=='periodic':
            d0=(k[0]+k[-1])/2/h**2; mat[i,n-1-i]=-d0; mat[i,i]+=d0
        if mode=='upwind':
            dv=max((-1 if i==0 else 1)*v[i],0)/h; mat[i,i]+=dv; mat[i,j]-=dv
        if mode=='dirichlet': mat[i,i]=1/h**2; mat[i,j]=0
    return mat.tocsc()
def S_for(ax,ay):
    L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    return np.concatenate([splu(L+mu*I).solve(rhs)[sensors] for mu in s['mu_shifts']])
settings=['robin','reflect','reflect2','periodic','face','upwind','dirichlet']; axes=[]
for k,n0 in [('x',nx),('y',ny)]: axes.append({m:axis(s['kappa_'+k],s['vel_'+k],s['L'+k]/(n0-1),s['robin_'+k],m) for m in settings})
for name in ['linear','quadratic','cubic']:
    C=models[name]; ycal=np.column_stack([np.concatenate([root(lambda u:(feat(u[None],name)@C)[0]-v,(v-C[0])@np.linalg.inv(C[1:5])).x for v in raw[:,i].reshape(-1,4)]) for i in range(raw.shape[1])]); out=[]
    for a,b in itertools.product(settings,repeat=2):
        e=S_for(axes[0][a],axes[1][b])-ycal
        out.append((np.sqrt(np.mean(e**2)),a,b,np.sqrt(np.mean(e**2,axis=0)).round(7).tolist(),np.sqrt(np.mean(proj(e)**2))))
    print('wall candidates',name,sorted(out)[:8])
