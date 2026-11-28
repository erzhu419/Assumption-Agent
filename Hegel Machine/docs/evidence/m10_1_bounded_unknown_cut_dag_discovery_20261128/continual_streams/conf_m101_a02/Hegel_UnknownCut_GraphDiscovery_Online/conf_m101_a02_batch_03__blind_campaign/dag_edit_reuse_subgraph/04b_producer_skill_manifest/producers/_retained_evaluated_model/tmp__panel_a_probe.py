import itertools as it
import pickle
import numpy as np

with open('/holdout_task/public_batch.pkl','rb') as f:
    s=pickle.load(f)[0]['study']
u,v=(s['panel_a'][k] for k in ('u','v'))
b=s['panel_b']; db=np.column_stack((np.ones(len(b['u'])),b['u']))
print('Panel B affine residual RMS:', np.mean((b['v']-db@np.linalg.lstsq(db,b['v'],rcond=None)[0])**2)**.5)
print('A ranges',u.min(0),u.max(0),'correlation',np.corrcoef(u.T))
print('Measured train/test coordinate maxima',*[np.max(np.abs(s[k]['x']),axis=0) for k in ('train','test')])

def fit(d,y,kind='shared'):
    w=np.ones((len(d),1 if kind=='shared' else y.shape[1]))
    for _ in range(35):
        if kind=='shared':
            sw=w[:,0]**.5
            c=np.linalg.lstsq(d*sw[:,None],y*sw[:,None],rcond=None)[0]
        else:
            c=np.column_stack([np.linalg.lstsq(d*w[:,j,None]**.5,y[:,j]*w[:,j]**.5,rcond=None)[0] for j in range(y.shape[1])])
        e=y-d@c
        if kind=='shared':
            r=np.linalg.norm(e,axis=1,keepdims=True)
            scale=np.median(r,axis=0)
        else:
            r=np.abs(e)
            scale=np.median(np.abs(e-np.median(e,axis=0)),axis=0)*1.4826
        w=1/(1+(r/np.maximum(1.8*scale,1e-6))**2)
    return c

p={k:list(it.combinations_with_replacement(range(4),k)) for k in range(1,4)}
bases={'affine':p[1], 'diagonal_cubic':p[1]+[(j,j,j) for j in range(4)], 'all_odd':p[1]+p[3], 'all_degree3':p[1]+p[2]+p[3]}
for name,idx in bases.items():
    d=np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for i in idx])
    c=fit(d,v)
    residual=np.linalg.norm(v-d@c,axis=1)
    pred=[]
    for j in range(len(d)):
        keep=np.arange(len(d))!=j
        pred.append(d[j]@fit(d[keep],v[keep]))
    err=np.linalg.norm(v-pred,axis=1)
    print(name,'fit row residual quantiles',np.quantile(residual,[.25,.5,.9,.95,1]),'largest rows',np.argsort(residual)[-6:],'LOO row quantiles',np.quantile(err,[.25,.5,.85,.9,.95,1]))
    print([(str(i),np.round(row,4).tolist()) for i,row in zip([()]+idx,c) if name!='all_degree3' or max(abs(row))>.03])
print('Independent affine residual medians:',[float(np.median(np.abs(v[:,j]-np.column_stack((np.ones(len(u)),u[:,j]))@fit(np.column_stack((np.ones(len(u)),u[:,j])),v[:,j,None])[:,0]))) for j in range(4)])
