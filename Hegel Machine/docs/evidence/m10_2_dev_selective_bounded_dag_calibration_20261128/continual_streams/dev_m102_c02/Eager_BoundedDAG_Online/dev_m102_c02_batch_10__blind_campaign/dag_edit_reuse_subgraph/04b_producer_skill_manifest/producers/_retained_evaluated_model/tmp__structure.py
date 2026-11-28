import pickle,numpy as np
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for name,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x,y=s[name]['u'],s[name][k]; keep=np.max(abs(y),axis=1)<3; x,y=x[keep],y[keep]; n=len(x); print(name,k,n)
 def test(fs):
  d=np.column_stack([np.ones(n),*fs]); h=np.linalg.pinv(d,rcond=1e-10); w=h@y; e=y-d@w; loo=e/np.maximum(1-np.sum(d*h.T,axis=1)[:,None],1e-9); return (np.mean(loo**2)**.5,np.mean(e**2)**.5,w)
 f=[('x',x),('x2',x*x),('x3',x**3)];
 for a in [.75,1,1.25,1.5,2,2.15,2.5,3,3.75,4.3]:
  f.extend([(f'sin{a}',np.sin(a*x)),(f'cos{a}',np.cos(a*x)),(f'tanh{a}',np.tanh(a*x)),(f'rat{a}',x/(1+a*x*x))])
 out=[]
 for a,b in [((n0,z0),('',None)) for n0,z0 in f]+list(combinations(f,2)):
  if a[0]!='x' and b[0]!='x' and name=='panel_a':continue
  loo,fit,w=test([z for _,z in [a,b] if z is not None]);out.append((loo,fit,a[0]+' '+b[0],w))
 for lo,fi,nm,w in sorted(out,key=lambda a:a[0])[:7]:print(nm,round(lo,7),round(fi,7));
 print('best coefficients',sorted(out,key=lambda a:a[0])[0][3].round(5));
 if name=='panel_a':
  for col in range(4):
   dx=np.column_stack([np.ones(n),x[:,col],x[:,(col+2)%4],x[:,(col+2)%4]**3]); print('A reduced',col,np.linalg.lstsq(dx,y[:,col],rcond=None)[0].round(4),np.std(dx@np.linalg.lstsq(dx,y[:,col],rcond=None)[0]-y[:,col]));
