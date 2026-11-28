from probe import *
from itertools import combinations_with_replacement

def poly(x,d):
    ix=list(combinations_with_replacement(range(x.shape[1]),d))
    return np.stack([np.prod(x[:,j],axis=1) for j in ix],1),ix

def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
q,iq=poly(u,2); d,id_=poly(u,3)
x=np.c_[np.ones(len(u)),u,q];w=fit(x,v1)
print('v1 quadratic weights',np.round(w,4))
for name,f in [('linear+sq',np.c_[u,u*u]),('linear+sq+cubes',np.c_[u,u*u,u**3]),('linear+sq+sine',np.c_[u,u*u,np.sin(u)]),('all q',np.c_[u,q])]:
    a=np.c_[np.ones(len(u)),f]; w2=fit(a,v2); h=np.sum((a@np.linalg.pinv(a))*np.eye(len(u)),1);print('v2',name,rms(a@w2-v2),rms((a@w2-v2)/(1-h[:,None])))
qz,_=poly(z,2); c1=np.c_[np.ones(len(z)),z,qz]@w;c2=np.c_[np.ones(len(z)),z,z*z]@fit(np.c_[np.ones(len(u)),u,u*u],v2)
f=np.c_[z,b,c1,c2]; names=[f'{g}{j}' for g in ['z','b','c','d'] for j in range(4)]
print('linear coefficients y,tr horizon 8',np.round(fit(np.c_[np.ones(len(z)),f],np.c_[y,tr[:,-1]]),4))

def curve(k):return np.c_[1-np.exp(-k*t),t]
kopt=least_squares(lambda k:(curve(k)@fit(curve(k),tr.T)-tr.T).ravel(),[.55]).x[0]
K=curve(kopt); amp=fit(K,tr.T).T; print('time fit',kopt,rms(amp@K.T-tr), 'amp linear rms',err(f,amp))
for i in range(8):
    rr=least_squares(lambda x: x[0]*(-np.expm1(-x[1]*t))+x[2]*t-tr[i],[tr[i,-1],.5,0],bounds=([-100,.05,-100],[100,4,100])).x
    print('row',i,np.round(np.r_[z[i],b[i],c1[i],c2[i],y[i],rr],3))

cols=[f]; nn=names.copy()
for tag,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a)),('tanh2',lambda a:np.tanh(2*a)),('sin2',lambda a:np.sin(2*a)),('cos2',lambda a:np.cos(2*a))]:
    cols.append(fn(f));nn.extend([f'{tag}({n})' for n in names])
a,idx=poly(f,2);cols.append(a);nn.extend([f'{names[i]}*{names[j]}' for i,j in idx]);a=np.column_stack(cols)
for label,Y in [('y',y),('traj8',tr[:,-1]),('amp0',amp[:,0]),('amp1',amp[:,1])]:
    trn=np.arange(52); te=np.arange(52,72);aa=a[trn]-a[trn].mean(0);yy=Y[trn]-Y[trn].mean();sel=[]
    for j in range(18):
        tmp=aa-aa[:,sel]@fit(aa[:,sel],aa) if sel else aa; score=np.abs(tmp.T@yy)/np.maximum(np.linalg.norm(tmp,axis=0),1e-12);score[sel]=-1;sel.append(np.argmax(score));m=np.c_[np.ones(len(a)),a[:,sel]]; ww=fit(m[trn],Y[trn]);r=m@ww-Y
        if j in (3,5,7,9,11,14,17):print('greedy',label,j+1,round(np.mean(abs(r[te])),5),round(np.mean(abs(r[trn])),5),[nn[k] for k in sel])
