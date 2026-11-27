import itertools,pickle,numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; p=s['reference_tracer_pulses']; B=np.array([np.tensordot(a['w_pulse'],s['source_modes'],axes=1).ravel(order='F') for a in p]).T; Y=np.array([a['y_pulse_raw'] for a in p]).T

def axis(k,v,h,mean,adv,sign):
    k=(k[:-1]+k[1:])/2 if mean==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); d=k/h**2; v=v*sign/h; a=(v[:-1]+v[1:])/2; pos=np.maximum(a,0); neg=np.minimum(a,0)
    if adv==0: lo=-d-pos; hi=-d+neg; mid=np.r_[d+pos,0]+np.r_[0,d-neg]
    if adv==1: lo=-d-pos; hi=-d+neg; mid=-np.r_[hi,0]-np.r_[0,lo]
    if adv==2:
        lo=-d-np.maximum(v[1:],0); hi=-d+np.minimum(v[:-1],0); mid=-np.r_[hi,0]-np.r_[0,lo]
    if adv==3: lo=-d-np.maximum(v[:-1],0); hi=-d+np.minimum(v[1:],0); mid=np.r_[d+np.maximum(v[:-1],0),0]+np.r_[0,d-np.minimum(v[1:],0)]
    return diags([lo,mid,hi],[-1,0,1],format='csc')

ends=np.array([np.tile(np.arange(nx)==0,ny),np.tile(np.arange(nx)==nx-1,ny),np.repeat(np.arange(ny)==0,nx),np.repeat(np.arange(ny)==ny-1,nx)],dtype=float).T
E=eye(nx*ny,format='csc')
def predict(A,r):
    AA=A+diags(ends@r,format='csc'); return np.concatenate([splu(AA+mu*E).solve(B)[idx] for mu in s['mu_shifts']])

def candidates():
 for grid,mean,adv,sign in itertools.product(range(2),range(2),range(4),(1,-1)):
    xx=s['Lx']/(nx-grid); yy=s['Ly']/(ny-grid)
    A=kron(eye(ny),axis(s['kappa_x'],s['vel_x'],xx,mean,adv,sign),format='csc')+kron(axis(s['kappa_y'],s['vel_y'],yy,mean,adv,sign),eye(nx),format='csc')+V
    yield (grid,mean,adv,sign), A, np.array([xx,xx,yy,yy])
if __name__=='__main__':
 results=[]; nom=np.repeat([s['robin_x'],s['robin_y']],2); Z=s['Z_nuisance']; print('pulse nuisance coefficients against zero',np.linalg.lstsq(Z,Y,rcond=None)[0]);
 for key,A,h in candidates():
    known=[]
    for label,r in [('direct',nom),('flux',nom/h),('ghost',nom/h**2)]: known.append((np.std(predict(A,r)-Y),label))
    fit=least_squares(lambda r:(predict(A,r)-Y).ravel(),nom/h,bounds=(0,250)); results.append((np.std(fit.fun),key,fit.x,np.array(fit.x*h),min(known)))
 for v in sorted(results,key=lambda v:v[0])[:12]: print(v)
 print('nominal',nom)
