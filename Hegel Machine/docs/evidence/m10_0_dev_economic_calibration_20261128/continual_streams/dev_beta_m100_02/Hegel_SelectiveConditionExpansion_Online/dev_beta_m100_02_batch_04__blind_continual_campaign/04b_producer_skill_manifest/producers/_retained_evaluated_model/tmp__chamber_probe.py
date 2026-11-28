import sys,pickle,itertools,json
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream

def axis(k,v,h,r,p):
    q=(k[:-1]+k[1:])/(2*h*h)
    lo=np.r_[-q[:-1]-np.maximum(v[1:-1],0)/h,0.]
    hi=np.r_[0.,-q[1:]-np.maximum(-v[1:-1],0)/h]
    d=np.r_[0.,-lo[:-1]-hi[1:],0.]
    e=[k[0]/h**2,k[-1]/h**2]
    a=diags([lo,d,hi],[-1,0,1],format='lil')
    for j,i,t,u in [(0,0,1,-1),(1,len(k)-1,len(k)-2,0)]:
        b,c,l=p[j*3:j*3+3]; a[i,i]=(b+c)*e[j]+l*r/h; a[i,t]=-b*e[j]; a[i,u]=-c*e[j]
    return a.tocsc()

def obs(s, p, modes):
    nx,ny=s['nx'],s['ny']; ix=eye(nx,format='csc'); iy=eye(ny,format='csc')
    ax,ay=[axis(s['kappa_'+z],s['vel_'+z],s['L'+z]/(s['n'+z]-1),s['robin_'+z],p[i*6:i*6+6]) for i,z in enumerate('xy')]
    a=kron(iy,ax,format='csc')+kron(ay,ix,format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; rhs=np.stack([m.ravel(order='F') for m in modes],axis=1)
    ident=eye(nx*ny,format='csc'); out=[]
    for shift in s['mu_shifts']:
        u=splu(a+shift*ident).solve(rhs); out.append(u[idx])
    return np.concatenate(out)

if __name__=='__main__':
    b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(b['study'],reference_tracer_pulses=b['campaign_diagnostics'])); pulses=s['reference_tracer_pulses']; y=np.array([d['y_pulse_raw'] for d in pulses]).T; modes=np.tensordot([d['w_pulse'] for d in pulses],s['source_modes'],axes=1)
    grammar={'R':(1,0,1),'N':(1,0,0),'P':(1,1,0),'H':(2,0,0),'B':(2,0,2)}
    results=[]
    rms=lambda a: np.sqrt(np.mean(a*a,axis=0))
    for ks in itertools.product(grammar,repeat=4):
        p=np.array([grammar[k] for k in ks]).ravel(); err=obs(s,p,modes)-y; results.append((np.linalg.norm(err),''.join(ks),p,rms(err)))
    for n,k,p,e in sorted(results)[:8]: print(k,e)
    for k in ['RRRR','NNNN','PPPP']: print('baseline',k,next(e for n,j,p,e in results if j==k))
    p0=sorted(results)[0][2]; np.set_printoptions(precision=7,suppress=True)
    for hold in [-1,0,1,2]:
        sel=np.arange(len(pulses))!=hold
        f=least_squares(lambda p:(obs(s,p,modes[sel])-y[:,sel]).ravel(),p0,bounds=(-0.15,8),max_nfev=70)
        print('fit',hold,'params',f.x.reshape(-1,3),'rmse',rms(obs(s,f.x,modes)-y),'singular values',np.linalg.svd(f.jac,compute_uv=False))
    for x in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
        a=np.array(x['facility_telemetry_trace']);print('telemetry',x['completed_campaign_short_name'],a.shape,np.round(np.median(a,axis=0),2),np.round(a[-4:].mean(axis=0),2))
