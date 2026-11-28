import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']; good=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5
X=X[good];Y=Y[good]; inds=np.arange(len(X)); axis=np.count_nonzero(X,axis=1)==1
print('single-bank range',[(j, c['single_bank_sweeps']['raw_readout'][c['single_bank_sweeps']['bank_id']==j].min(),c['single_bank_sweeps']['raw_readout'][c['single_bank_sweeps']['bank_id']==j].max()) for j in range(4)])

def basis(x,k):
    return np.column_stack([np.prod(x**p,axis=1) for p in product(range(k+1),repeat=2) if sum(p)<=k])

def f(p,x,kind):
    u,v=x.T; a=u+v; b=u-v
    if kind.startswith('poly'): return basis(x,int(kind[4:]))@p
    if kind=='add': return p[0]+p[1]*np.tanh(p[2]*u)+p[3]*np.tanh(p[4]*v)
    if kind=='mix': return p[0]+p[1]*np.tanh(p[2]*a)+p[3]*np.tanh(p[4]*b)
    if kind=='mixlin': return f(p,x,'mix')+p[5]*a+p[6]*b
    if kind=='mixshift': return p[0]+p[1]*np.tanh(p[2]*a+p[5])+p[3]*np.tanh(p[4]*b+p[6])
    if kind=='mixquad': return f(p,x,'mixshift')+p[7]*a+p[8]*b+p[9]*a*a+p[10]*b*b+p[11]*a*b
    if kind=='mixrot': return p[0]+p[1]*np.tanh(p[2]*u+p[4]*v+p[3])+p[5]*np.tanh(p[6]*u+p[7]*v+p[8])
    if kind=='addrot': return f(p,x,'mixrot')
    if kind=='soft': return p[0]+p[1]*a/np.sqrt(1+(p[2]*a)**2)+p[3]*b/np.sqrt(1+(p[4]*b)**2)
    if kind=='rat': return p[0]+p[1]*a/(1+(p[2]*a)**2)+p[3]*b/(1+(p[4]*b)**2)

def fit(x,y,kind):
    if kind.startswith('poly'): return np.linalg.lstsq(basis(x,int(kind[4:])),y,rcond=None)[0]
    p={'add':[0,1.5,1,.3,1],'mix':[0,1,1,.6,1],'mixlin':[0,1,1,.6,1,0,0],'mixshift':[0,1,1,.6,1,0,0],'mixquad':[0,1,1,.6,1,0,0,0,0,0,0,0],'mixrot':[0,1,1,0,1,.6,1,-1,0],'addrot':[0,1.5,1,0,0,.3,0,1,0],'soft':[0,1,1,.6,1],'rat':[0,1,.25,.6,.25]}[kind]
    return least_squares(lambda p:f(p,x,kind)-y,p,max_nfev=300).x

if __name__=='__main__':
 rng=np.random.default_rng(2026); fold=np.empty(len(X),int)
 for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
 for kind in ['poly1','poly3','poly5','add','mix','mixlin','mixshift','mixquad','mixrot','addrot','soft','rat']:
    pred=np.empty_like(Y); params=[]
    for j in range(4):
       x=X[:,[j,j^2]]; y=Y[:,j]; p=fit(x,y,kind); params.append(p)
       for k in range(5):
          tr=fold!=k; pred[~tr,j]=f(fit(x[tr],y[tr],kind),x[~tr],kind)
    e=pred-Y; print(kind,'rms',np.sqrt((e**2).mean(0)).round(6),'axis',round(np.sqrt(np.mean(e[axis]**2)),6),'mixed',round(np.sqrt(np.mean(e[~axis]**2)),6),'max',round(abs(e).max(),6))
    if kind in ['mixshift','mixquad','mixrot','addrot']: print(' parameters', np.array(params).round(5))
