import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['panel_b'];u=p['u'];v=p['v']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def pairs(x):return np.stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)],axis=1)
flags={
 'cos':lambda x,g:aff(np.cos(g*x)), 'sin+cos':lambda x,g:aff(np.c_[np.sin(g*x),np.cos(g*x)]), 'lin+cos':lambda x,g:aff(np.c_[x,np.cos(g*x)]), 'lin+sin+cos':lambda x,g:aff(np.c_[x,np.sin(g*x),np.cos(g*x)]), 'quad+cos':lambda x,g:aff(np.c_[x,pairs(x),np.cos(g*x)]), 'sin+cos+h2':lambda x,g:aff(np.c_[np.sin(g*x),np.cos(g*x),np.sin(2*g*x),np.cos(2*g*x)]), 'quad':lambda x,g:aff(np.c_[x,pairs(x)])}
def score(a,y):
 q,r=np.linalg.qr(a,mode='reduced');e=a@ls(a,y)-y;return np.mean(abs(e)),np.mean(abs(e/np.maximum(1-np.sum(q*q,axis=1),1e-9)[:,None]))
for name,f in flags.items():
 for mode in [0,1]:
  gs=np.linspace(.5,10,191);i=np.argmin([score(f(u,g),v)[mode] for g in gs]);r=minimize_scalar(lambda g:score(f(u,g),v)[mode],bounds=(gs[max(i-1,0)],gs[min(i+1,len(gs)-1)]),method='bounded');print(name,mode,round(r.x,5),score(f(u,r.x),v),'at2.15',score(f(u,2.15),v))
# single channel library to see if cross-frequency terms are needed
for a in [aff(u),aff(np.c_[u,pairs(u)]),flags['sin+cos'](u,2.15),flags['lin+sin+cos'](u,2.15)]:
 print('coeffs',ls(a,v).round(3))
