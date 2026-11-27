import sys,pickle,itertools
import numpy as np
from scipy.sparse import csc_matrix,eye,kron,diags
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(s,a,kind):
    k,v=s['kappa_'+a],s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1)
    A=np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1) and kind!='periodic':
            j=1 if i==0 else n-2
            A[i,j]=-k[i]/h**2
            A[i,i]=-A[i,j]+(s['robin_'+a]/h if kind=='open' else 0)
        else:
            j,l=(i-1)%n,(i+1)%n
            A[i,j]=-(k[i]+k[j])/(2*h**2)-max(v[i],0)/h
            A[i,l]=-(k[i]+k[l])/(2*h**2)-max(-v[i],0)/h
            A[i,i]=-A[i,j]-A[i,l]
    return csc_matrix(A)

def response(s,ax,ay):
    nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T
    op=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    Id=eye(nx*ny,format='csc')
    rhs=np.array([m.ravel(order='F') for m in s['source_modes']]).T
    return np.vstack([splu(op+mu*Id).solve(rhs)[i+nx*j] for mu in s['mu_shifts']])

def data():
    item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
    p=s['reference_tracer_pulses']; W=np.array([q['w_pulse'] for q in p]).T; Y=np.array([q['y_pulse_raw'] for q in p]).T
    return s,W,Y

if __name__=='__main__':
    s,W,Y=data(); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
    for x,y in itertools.product(['open','reflect','periodic'],repeat=2):
        E=response(s,axis(s,'x',x),axis(s,'y',y))@W-Y
        R=E-Q@(Q.T@E)
        print(x,y,'raw RMSE',np.round(np.sqrt(np.mean(E**2,axis=0)),7),'profiled RMSE',np.round(np.sqrt(np.mean(R**2,axis=0)),7),'shift RMSE',np.round([np.sqrt(np.mean(t**2)) for t in np.split(E,len(s['mu_shifts']))],7))
