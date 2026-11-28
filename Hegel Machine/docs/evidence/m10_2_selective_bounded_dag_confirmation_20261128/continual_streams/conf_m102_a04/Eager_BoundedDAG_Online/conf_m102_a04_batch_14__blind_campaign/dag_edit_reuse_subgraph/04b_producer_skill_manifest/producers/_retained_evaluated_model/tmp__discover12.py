import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def one(u):return np.c_[np.ones(len(u)),u]
def ls(f,v):return np.linalg.lstsq(f,v,rcond=1e-10)[0]
a=s['panel_a']; sel=np.linalg.norm(a['v'],axis=1)<5; u=a['u'][sel];v=a['v'][sel]
b=s['panel_b']; B=ls(one(b['u']),b['v']);print('B',B)
def xy(u):return np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])
def study(u,v,name):
 print('\n',name);n=len(u);w=u*u
 def score(f, coeff=False):
  c=ls(f,v); r=f@c-v;h=np.sum(np.linalg.svd(f,full_matrices=False)[0][:,:np.linalg.matrix_rank(f)]**2,axis=1);return np.round([np.std(r),np.std(r/np.maximum(1-h[:,None],1e-9))],6) if not coeff else c
 transforms={'abs':lambda q:np.abs(u),'sq':lambda q:w,'cube':lambda q:u*w,'soft':lambda q:u/(1+q*abs(u)), 'tanh':lambda q:np.tanh(q*u),'cos':lambda q:np.cos(q*u),'sin':lambda q:np.sin(q*u),'odd_rat':lambda q:u/(1+q*w),'even_rat':lambda q:w/(1+q*w),'plusminus':lambda q:np.c_[np.column_stack([np.sin(q*(u[:,i]+u[:,j])) for i,j in combinations(range(4),2)]),np.column_stack([np.sin(q*(u[:,i]-u[:,j])) for i,j in combinations(range(4),2)])]}
 for k,g in transforms.items():
  fixed=[]
  for base in [one(u),np.c_[one(u),xy(u)]]:
   options=[]
   for q in [.22,.45,.85,1.25,1.6,2.15,2.6,3.14,4]:
    f=np.c_[base,g(q)];options.append((score(f),q))
   q=min(options,key=lambda p:p[0][1])[1];fixed.append((q,score(np.c_[base,g(q)])))
  print(k,fixed)
 for f in [one(u),np.c_[one(u),u*u,xy(u),u**3],np.c_[one(u),u**3,u*u]]:
  print('coefs',len(f.T),score(f),score(f,True).T)
 print('affine residual singular values',np.linalg.svd(v-one(u)@ls(one(u),v),compute_uv=False))
 for w in [u@B[1:],(u-B[0])@np.linalg.inv(B[1:])]:
  for op in [('soft',lambda x:x/(1+.85*abs(x))),('sin',lambda x:np.sin(2.15*x)),('rat',lambda x:x*x/(1+.25*x*x))]:
   print('B',op[0],score(np.c_[one(u),op[1](w)]))
study(u,v,'A clean');c=s['panel_c'];study(c['u'],c['v1'],'C1');study(c['u'],c['v2'],'C2')
