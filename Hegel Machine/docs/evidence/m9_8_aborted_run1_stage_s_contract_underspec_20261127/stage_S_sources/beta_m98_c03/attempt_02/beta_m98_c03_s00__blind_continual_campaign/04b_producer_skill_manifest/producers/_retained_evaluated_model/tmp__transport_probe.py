import itertools,pickle
import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
nx,ny=s['nx'],s['ny']; coords=s['sensor_ij']; ids=coords[:,0]+nx*coords[:,1]; shifts=s['mu_shifts']; pulses=s['reference_tracer_pulses']; M=len(s['source_modes']); f=s['source_modes'].reshape(M,-1,order='F').T@np.stack([p['w_pulse'] for p in pulses],axis=1); y=np.stack([p['y_pulse_raw'] for p in pulses],axis=1); rx,ry=s['robin_x'],s['robin_y']

def axis(k,v,h,km,vm,form):
    n=len(k); j=np.arange(n-1); A=np.zeros((n,n)); d={'ar':(k[:-1]+k[1:])*.5,'ha':2*k[:-1]*k[1:]/(k[:-1]+k[1:]),'geo':np.sqrt(k[:-1]*k[1:])}[km]/h**2
    a={'face':(v[:-1]+v[1:])*.5,'left':v[:-1],'right':v[1:]}[vm]/h
    A[j,j]+=d; A[j+1,j+1]+=d; A[j,j+1]-=d; A[j+1,j]-=d
    if form=='flux':
        A[j,j]+=np.maximum(a,0); A[j+1,j]-=np.maximum(a,0); A[j,j+1]+=np.minimum(a,0); A[j+1,j+1]-=np.minimum(a,0)
    elif form=='velocity':
        p=np.maximum(v[1:]/h,0); q=np.maximum(-v[:-1]/h,0); A[j,j]+=q; A[j,j+1]-=q; A[j+1,j+1]+=p; A[j+1,j]-=p
    elif form=='facevelocity':
        p=np.maximum(a,0); q=np.maximum(-a,0); A[j,j]+=q; A[j,j+1]-=q; A[j+1,j+1]+=p; A[j+1,j]-=p
    elif form=='none': pass
    return A

def init(delt,km,vm,form):
    hx=s['Lx']/(nx-delt);hy=s['Ly']/(ny-delt)
    ax=axis(s['kappa_x'],s['vel_x'],hx,km,vm,form);ay=axis(s['kappa_y'],s['vel_y'],hy,km,vm,form)
    B=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    xx=np.tile(np.arange(nx),ny); yy=np.repeat(np.arange(ny),nx)
    return B,np.stack([xx==0,xx==nx-1,yy==0,yy==ny-1],axis=1).astype(float),hx,hy

def predict(B,d):
    return np.concatenate([solve(B+np.diag(d+mu),f,check_finite=False)[ids] for mu in shifts])

def boundary(hx,hy):
    h=np.repeat([hx,hy],2); r=np.repeat([rx,ry],2); k=np.array([s['kappa_x'][0],s['kappa_x'][-1],s['kappa_y'][0],s['kappa_y'][-1]]); v=np.array([-s['vel_x'][0],s['vel_x'][-1],-s['vel_y'][0],s['vel_y'][-1]])/h
    for name,c in {'r':r,'r/h':r/h,'kr/h':k*r/h,'r/k':r/k,'r/(h+r*h*h/2k)':r/(h+r*h*h/(2*k)),'kr/(h+rh*h/2)':k*r/(h+r*h*h/2),'2r/h':2*r/h,'r/h2':r/h**2,'kr/h2':k*r/h**2,'robin-extrap':k/(h*(1/r+h/2))}.items():
        for an,ad in [('zero',0),('out',np.maximum(v,0)),('signed',v),('in',np.maximum(-v,0))]:
            yield name,an,c+ad

if __name__=='__main__':
    ranks=[];free=[]
    for config in itertools.product([0,1,-1],['ar','ha','geo'],['face','left','right'],['flux','velocity','facevelocity','none']):
        B,E,hx,hy=init(*config)
        fit=least_squares(lambda c:(predict(B,E@c)-y).ravel(),np.ones(4)*3,bounds=(-.25,40),gtol=1e-9)
        free.append((np.sqrt(np.mean(fit.fun**2)),config,fit.x))
        for bn,an,c in boundary(hx,hy):
            e=predict(B,E@c)-y
            ranks.append((np.sqrt(np.mean(e**2)),config,bn,an,c))
    print('BEST exact'); print(*sorted(ranks,key=lambda z:z[0])[:12],sep='\n'); print('BEST profiled'); print(*sorted(free,key=lambda z:z[0])[:18],sep='\n')
