import os, pickle, importlib.util, numpy as np
from itertools import product
from scipy.linalg import qr
from scipy.optimize import least_squares

def load(p):
 s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
n=load('/holdout_task/node_00.py'); f=load('/holdout_task/_frozen_nodes/node_01.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; z=n._invert(x,n._fit_sensor(s['panel_a']));
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def rat(a,k):return a/(1+k*a*a)
def sq(a):return np.c_[aff(a),a*a]
B=s['panel_b']; C=s['panel_c']; b,wb=n._fit_coordinates(B); c,wc=n._fit_coordinates(dict(u=C['u'],v=C['v1'])); wd=ls(sq(C['u']),C['v2']);
M={'B':lambda a:n._design(a,b)@wb,'C':lambda a:n._design(a,c)@wc,'D':lambda a:sq(a)@wd}
for name,sc,W in [('B',b,wb),('C',c,wc),('D',None,wd)]:
 L=W[1:5]; R=W[5:9]; ratio=(L*R).sum(1)/(L*L).sum(1); print(name,'ratio',ratio,'nonparallel',np.linalg.norm(R-ratio[:,None]*L));
 def d(a,k,r,name=name):return aff(a+a*a*r if name=='D' else a+r*rat(a,k))
 U,V=(B['u'],B['v']) if name=='B' else (C['u'],C['v1' if name=='C' else 'v2']);
 def rhs(p):
  H=d(U,np.exp(p[0]),p[1]);return (H@ls(H,V)-V).ravel()
 p=least_squares(rhs,[0.,float(np.median(ratio))],bounds=(-5,5),max_nfev=180).x; k,r=np.exp(p[0]),p[1]; W=ls(d(U,k,r),V);print('shared',name,k,r,np.std(rhs(p)),W)
 M[name.lower()]=lambda a,k=k,r=r,W=W,d=d:d(a,k,r)@W
 M[name+'0']=lambda a,W=W,d=d,k=k,r=r:d(a,k,r)[:,1:]
 M[name+'1']=lambda a,W=W:aff(a)@W
T=np.c_[y,t]; folds=[(np.arange(52),np.arange(52,72)),(np.arange(18,72),np.arange(18)),(np.r_[0:18,36:72],np.arange(18,36))];
def scr(a):
 if not np.isfinite(a).all() or np.amax(abs(a))>1e8:return [1e5]*3
 h=aff(a); return [np.mean(abs(h[j]@ls(h[i],t[i])-t[j])) for i,j in folds]
A={'z':z,'x':x}; ents=[]
def add(name,a):
 e=scr(a); ents.append((max(e),np.mean(e),name,a)); A[name]=a
for k,a in list(A.items()):
 for g,m in M.items(): add(k+g,m(a))
for k,a in list(A.items()):
 if len(k)<2:continue
 for g,m in M.items():add(k+':'+g,m(a))
for k,a in list(A.items()):
 if ':' not in k:continue
 for g in 'BCD':add(k+':'+g,M[g](a))
for k,a in list(A.items()):
 if k.count(':')>1:continue
 v,w=M['C'](a),M['D'](a)
 for op,q in [('sum',v+w),('dif',v-w),('prod',v*w),('gate',v*(1+w)),('gate2',w*(1+v)),('sq',v*v+w),('rat',v/(1+w*w)),('blend',a+v+w)]:add(k+'_'+op,q)
for e in sorted(ents,key=lambda a:a[:2])[:22]:
 a=e[-1]; sy=[]
 for i,j in folds:sy.append(round(np.mean(abs(f._scalar_readout(a[i],a[j],y[i])-y[j])),5))
 print('top',e[:3],sy)
# Profile combinations in a calibrated feature dictionary, independently inside each fold.
base=['z','x','zB','zC','zD','zB:C','zB:D','zC:B','zD:B','zB0','zC0','zD0']; names=[]; blocks=[]
def block(name,a):
 if a.ndim==1:a=a[:,None]
 names.extend([name+'_'+str(j) for j in range(a.shape[1])]);blocks.append(a)
for key in base:
 a=A[key];block(key,a);block(key+'2',a*a);block(key+'3',a*a*a)
 for k in (.3,1,3,8):block(key+'rat'+str(k),rat(a,k));block(key+'even'+str(k),1/(1+k*a*a))
 for k in (1,2,4,7):block(key+'sin'+str(k),np.sin(k*a));block(key+'cos'+str(k),np.cos(k*a))
from itertools import combinations
for key in base:
 a=A[key];block(key+'inter',np.column_stack([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]))
H=np.column_stack(blocks); sig=H.std(0); H=H[:,sig>1e-7]; names=np.asarray(names)[sig>1e-7];
for i,j in folds:
 h=(H-H[i].mean(0))/H[i].std(0).clip(1e-7); tar=T-T[i].mean(0); print('ridge',[(l,np.mean(abs(h[j]@h[i].T@np.linalg.solve(h[i]@h[i].T+l*np.eye(len(i)),tar[i])+T[i].mean(0)-T[j]),0)[[0,-1]].round(4)) for l in (.01,.1,1,10,100)]);
 for mode,target in [('end',t[:,-1]),('scalar',y)]:
  cols=[]; r=target[i]-target[i].mean()
  for _ in range(15):
   R=h[i].copy()
   if cols:R-=R[:,cols]@ls(R[:,cols],R)
   k=np.argmax(abs(R.T@r)/np.maximum(np.linalg.norm(R,axis=0),.01));cols.append(k); g=aff(h[:,cols]);w=ls(g[i],target[i]);r=target[i]-g[i]@w
   if len(cols) in (3,6,10,15):print(mode,len(cols),round(np.mean(abs(g[j]@w-target[j])),4),names[cols].tolist())
