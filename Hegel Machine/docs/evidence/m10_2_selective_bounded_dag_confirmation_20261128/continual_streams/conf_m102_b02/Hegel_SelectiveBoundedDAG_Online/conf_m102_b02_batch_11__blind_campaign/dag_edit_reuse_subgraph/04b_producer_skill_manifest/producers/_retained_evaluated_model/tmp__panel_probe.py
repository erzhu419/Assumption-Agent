import pickle, numpy as np, runpy
from itertools import combinations_with_replacement as cwr
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'node_00.py')['run_node']({'__input__':s})
x=a['corrected'][:len(s['train']['x'])]; b=a['operator']; f=lambda x: np.c_[np.ones(len(x)), x]
y=np.c_[s['train']['y'],s['train']['traj']]
pairs=((0,1),(0,3),(1,2),(2,3))
base=[(i,) for i in range(4)]+list(pairs)+[c for c in cwr(range(4),3) if any(i in c and j in c for i,j in pairs)]
full=lambda k:[c for j in range(1,k+1) for c in cwr(range(4),j)]
def calc(z,terms):
 d=np.column_stack([np.ones(len(z))]+[z[:,c].prod(1) for c in terms]); co=np.linalg.lstsq(d[:52],y[:52],rcond=None)[0]; e=d[52:]@co-y[52:]; et=d@np.linalg.lstsq(d,y,rcond=None)[0]-y
 return np.round([abs(e[:,0]).mean(),abs(e[:,1:]).mean(),abs(et[:,0]).mean(),abs(et[:,1:]).mean()],5)
z=f(x)@b; c=s['panel_c']; g=[np.linalg.lstsq(f(c['u']),c[k],rcond=None)[0] for k in ('v1','v2')]
for k,v in dict(x=x,B=z,Bt=x@b[1:].T+b[0],Bi=(x-b[0])@np.linalg.inv(b[1:]),Bzero=x@b[1:],C=f(x)@g[0],BC=f(z)@g[0],CB=f(f(x)@g[0])@b,BC2=f(z)@g[1]).items():
 print(k, 'base',calc(v,base),'quad',calc(v,full(2)),'cubic',calc(v,full(3)),'quartic',calc(v,full(4)))
print('calibration'); print(np.linalg.lstsq(f(s['train']['x']),x,rcond=None)[0]); print('B',b)
