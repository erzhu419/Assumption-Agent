from chamber_probe import *
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def wall(k,v,h,r,kind):
    a=axis(k,v,h,r,0,0).tolil(); n=len(k)
    for i,j,b in [(0,1,n-1),(n-1,n-2,0)]:
        if kind=='reflect': continue
        if kind=='robin': a[i,i]+=r/h; continue
        if kind=='double': a[i,i]*=2; a[i,j]*=2; continue
        if kind=='half': a[i,i]*=.5; a[i,j]*=.5; continue
        if kind=='dirichlet': a.rows[i]=[i]; a.data[i]=[1e6]; continue
        if kind in ('periodic','wraplocal','wraponly'):
            fm=(k[i]+k[b])/2/h**2 if kind!='wraplocal' else k[i]/h**2
            a[i,b]=-fm-max((1 if i==0 else -1)*v[i],0)/h
            if kind=='periodic': a[i,j]=-(k[i]+k[j])/2/h**2-max((-1 if i==0 else 1)*v[i],0)/h
            if kind=='wraponly': a[i,j]=0
            a[i,i]=-a[i,b]-a[i,j]; continue
        t=np.asarray(kind)[0 if i==0 else 1]; f=k[i]/h**2
        a[i,j]=-f*t[0]; a[i,b]=-f*t[1]; a[i,i]=f*(t[0]+t[1])+r/h*t[2]
    return a.tocsc()

def predict(px,py):
    nx,ny=s['nx'],s['ny']; x=wall(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],px); y=wall(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],py)
    L=kron(eye(ny),x,format='csc')+kron(y,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc')
    Q=np.tensordot(W.T,s['source_modes'],axes=1).reshape((3,-1),order='F').T
    ii,jj=s['sensor_ij'].T; ids=ii+nx*jj
    return np.vstack([splu(L+shift*I).solve(Q)[ids] for shift in s['mu_shifts']])

def proj(R): return R-Z@np.linalg.lstsq(Z,R,rcond=None)[0]

def errors(px,py): return proj(predict(px,py)-Y)

if __name__=='__main__':
    kinds=['reflect','robin','double','half','dirichlet','periodic','wraplocal','wraponly']
    rankings=[]
    for px,py in itertools.product(kinds,repeat=2):
        err=errors(px,py); rankings.append((np.sqrt(np.mean(err**2)),px,py,np.sqrt(np.mean(err**2,axis=0))))
    for r in sorted(rankings)[:14]: print('wall',r)
    for test in [0,1,2,None]:
        train=[i for i in range(3) if i!=test]
        def ev(p): return errors(*p.reshape(2,2,3))
        a=least_squares(lambda p:ev(p)[:,train].ravel(),np.tile([1.,.1,.1],4),bounds=(np.tile([0.,0.,-.35],4),np.tile([4.,4.,3.],4)),max_nfev=100)
        print('fit',test,a.x.reshape(2,2,3).round(5),'pulse errors',np.sqrt(np.mean(ev(a.x)**2,axis=0)))
    diff=predict('reflect','reflect')-Y
    print('reflect raw mean and std per shift',[(np.mean(a,axis=0),np.std(a,axis=0)) for a in np.split(diff,3)])
    print('reflect residual by sensor coordinate (projected)',[(tuple(k),r) for k,r in zip(s['sensor_ij'],np.sqrt(np.mean(proj(diff).reshape(3,32,3)**2,axis=(0,2))).round(4))])
