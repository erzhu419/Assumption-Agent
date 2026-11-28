import itertools
import pickle
import runpy
import numpy as np

st = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
d = runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':st})
z, y = d['z_tr'], d['train_y']
n = len(y)
a,b,c,e = z.T
q = np.column_stack((a*b,a*e,b*c,c*e))
base = np.c_[np.ones(n),q]
coef = np.linalg.lstsq(base,y,rcond=None)[0]
r = y-base@coef
folds = [np.arange(n-20,n), *np.array_split(np.random.default_rng(713).permutation(n),4), np.argsort(np.linalg.norm(q,axis=1))[-n//4:]]

def evaluate(name,X,target=y,inverse=lambda p:p):
    errs=[]
    for te in folds:
        tr=np.setdiff1d(np.arange(n),te)
        p=inverse(X[te]@np.linalg.lstsq(X[tr],target[tr],rcond=None)[0])
        errs.append(np.mean(np.abs(y[te]-p)))
    full=inverse(X@np.linalg.lstsq(X,target,rcond=None)[0]); print(name,X.shape[1], 'fit',round(np.mean(abs(y-full)),6),'slices',np.round(errs,6),'worst',round(max(errs),6))

def monomials(X,k):
    return np.column_stack([np.prod(X[:,i],axis=1) for i in itertools.combinations_with_replacement(range(X.shape[1]),k)])

print('response range', y.min(),y.max(),'baseline residual range',r.min(),r.max())
evaluate('bilinear',base)
evaluate('full quadratic z',np.c_[np.ones(n),z,monomials(z,2)])
for degree in [2,3]:
    evaluate(f'polynomial q degree {degree}',np.c_[base,*[monomials(q,k) for k in range(2,degree+1)]])
for freq in [.35,.7,1.,1.4,2.]:
    evaluate(f'q sincos {freq}',np.c_[base,np.sin(freq*q),np.cos(freq*q)])
for scale in [1.5,2.,3.,4.,6.,10.,20.]:
    if scale>np.max(np.abs(y)):
        evaluate(f'scaled tanh {scale}',base,np.arctanh(y/scale),lambda p:scale*np.tanh(p))
np.set_printoptions(precision=4,suppress=True)
print('bilinear coefficients',coef)
print('largest residuals: q, observed, fitted, error')
ii=np.argsort(abs(r))[-12:];print(np.c_[q,y,y-r,r][ii])
print('quadratic q coefficients',np.linalg.lstsq(np.c_[base,monomials(q,2)],y,rcond=None)[0])
