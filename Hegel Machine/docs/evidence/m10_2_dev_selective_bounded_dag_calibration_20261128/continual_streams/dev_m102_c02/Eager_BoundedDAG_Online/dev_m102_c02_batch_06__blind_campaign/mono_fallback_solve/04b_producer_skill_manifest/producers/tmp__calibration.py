import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; keep=np.max(abs(v-u),1)<2; u,v=u[keep],v[keep]; L=lambda z,y:np.linalg.lstsq(z,y,rcond=None)[0]
for j in range(4):
 a,b,c,d=np.roll(u,-j,axis=1).T; y=v[:,j]; base=np.array([a*0+1,a,b,a*b]).T; results=[]
 for den,q in [('a',a),('b',b),('c',c),('d',d),('a+b',a+b),('a-b',a-b),('b+c',b+c),('b-c',b-c),('ab',a*b),('norm',np.linalg.norm(u,axis=1)),('normab',np.hypot(a,b))]:
  for bits in itertools.product([False,True],repeat=4):
   if not any(bits):continue
   def calc(k,ret=False):
    z=base/np.where(bits,1+k*q[:,None]**2,1); w=L(z,y); r=y-z@w
    if ret:return np.mean(abs(r/(1-np.sum(np.linalg.qr(z)[0]**2,1)))),w
    return np.linalg.norm(r)
   k=minimize_scalar(calc,bounds=(0,1),method='bounded').x; err,w=calc(k,True);results.append((err,den,bits,k,w))
 print(j, *sorted(results,key=lambda a:a[0])[:7],sep='\n')
