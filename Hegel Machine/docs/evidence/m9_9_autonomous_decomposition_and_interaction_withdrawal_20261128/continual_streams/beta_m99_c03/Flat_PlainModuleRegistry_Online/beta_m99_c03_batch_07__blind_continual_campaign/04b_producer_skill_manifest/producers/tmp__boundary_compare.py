import sys, pickle, itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from _verified_upstream import calibrate_upstream

def axis(k,v,r,L,b):
    n=len(k); h=L/(n-1); i=np.arange(n); p=(i-1)%n; q=(i+1)%n
    l=(k+k[p])/(2*h*h)+np.maximum(v,0)/h
    u=(k+k[q])/(2*h*h)+np.maximum(-v,0)/h
    if b!='P':
        l[0]=u[-1]=0
        u[0]=k[0]/h**2; l[-1]=k[-1]/h**2
    d=l+u
    if b=='R': d[[0,-1]]+=r/h
    return sparse.csc_matrix((np.r_[d,-l,-u],(np.tile(i,3),np.r_[i,p,q])),shape=(n,n))

def forward(s,bs):
    nx,ny=s['nx'],s['ny']; a,b=[axis(s['kappa_'+c],s['vel_'+c],s['robin_'+c],s['L'+c],z) for c,z in zip('xy',bs)]
    A=sparse.kron(sparse.eye(ny),a,format='csc')+sparse.kron(b,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    I=sparse.eye(nx*ny,format='csc'); rhs=np.column_stack([x.ravel(order='F') for x in s['source_modes']]); i,j=s['sensor_ij'].T
    return np.vstack([splu(A+t*I).solve(rhs)[i+nx*j] for t in s['mu_shifts']])

def fit(s,X,y,Z):
    P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    A=X-Z@(P@X); b=y-Z@(P@y); ridge=s['source_ridge']; m=X.shape[1]; budget=s['total_budget']
    H=A.T@A+ridge*np.eye(m); c=A.T@b
    opt=minimize(lambda w: .5*w@H@w-c@w,np.full(m,budget/m),jac=lambda w:H@w-c,method='SLSQP',bounds=[(0,s['u_box'])]*m,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(m)}],options={'ftol':1e-14})
    w=opt.x
    return w,P@(y-X@w),.5*(np.linalg.norm(A@w-b)**2+ridge*(w@w))

if __name__=='__main__':
    raw=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; s=calibrate_upstream(raw)
    y=s['y_obs_raw']; Z=s['Z_nuisance']; sensor=np.arange(len(y))%len(s['sensor_ij']); groups=sensor//4
    for bs in itertools.product('RFP',repeat=2):
        X=forward(s,bs); w,g,J=fit(s,X,y,Z); walk=[]
        for fold in np.unique(groups):
            te=groups==fold; tr=~te
            wf,gf,_=fit(s,X[tr],y[tr],Z[tr]); walk.append(np.mean((X[te]@wf+Z[te]@gf-y[te])**2)**.5)
        print(''.join(bs),'objective',round(J,6),'train',round(np.std(y-X@w-Z@g),6),'group CV rms/max',np.round([np.mean(np.square(walk))**.5,max(walk)],6),'weights',w.round(3))
