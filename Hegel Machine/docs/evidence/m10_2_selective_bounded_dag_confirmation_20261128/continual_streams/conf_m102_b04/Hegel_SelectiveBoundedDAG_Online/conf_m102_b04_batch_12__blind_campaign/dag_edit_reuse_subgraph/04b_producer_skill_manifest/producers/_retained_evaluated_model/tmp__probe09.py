import numpy as np,pickle,runpy,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); q=np.cbrt(z['cubic_train']); n=len(q)
def features(u,kind,w=1.25):
 i,j=np.triu_indices(u.shape[1],1); pair=u[:,i]*u[:,j]
 return np.column_stack([np.ones(len(u)), *[{'l':u,'s':np.sin(w*u),'c':np.cos(w*u)-1,'2':u*u,'3':u**3,'p':pair,'t':np.tanh(w*u),'r':u/(1+w*u*u)}[x] for x in kind]])
def prof(u,v,kind,w=1.25):
 a=features(u,kind,w); c=np.linalg.lstsq(a,v,rcond=None)[0]; r=a@c-v; h=(np.linalg.qr(a,mode='reduced')[0]**2).sum(1); return np.mean(r*r), np.mean(abs(r/np.maximum(1-h,1e-10)[:,None])), c
if __name__=='__main__':
 for name,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u=s[name]['u']; v=s[name][key]; print(name,key)
  for kind in ['l','s','ls','lsc','lsp','l2','l3','l23','l2p','l23p','lt','lr','lcp','l2s','l2sp']:
   ws=np.linspace(.2,4,39); w=min(ws,key=lambda w:prof(u,v,kind,w)[0]); opt=minimize_scalar(lambda w:prof(u,v,kind,w)[0],bounds=(max(.05,w-.11),w+.11),method='bounded'); e,loo,c=prof(u,v,kind,opt.x); print(kind,round(opt.x,5),f'{e**.5:.5g}',f'{loo:.5g}')
  if key=='v2':
   for kind in ['l','s','ls','lsp','l2','l23','l2p','l23p','lt','lr','lsc']:
    a=s[name]['v1']; opt=minimize_scalar(lambda w:prof(a,v,kind,w)[0],bounds=(.1,4),method='bounded'); e,loo,c=prof(a,v,kind,opt.x); print('v1->v2',kind,round(opt.x,5),f'{e**.5:.5g}',f'{loo:.5g}')
