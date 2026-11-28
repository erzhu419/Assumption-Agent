import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
rm=lambda e:np.mean(e**2)**.5
aff=lambda x:np.c_[np.ones(len(x)),x]
p=s['panel_a']; a=aff(p['u']); v=p['v']; keep=np.linalg.norm(a@ls(a,v)-v,axis=1)<2.5; cal=ls(a[keep],v[keep]); print('cal',cal.round(5))
p=s['panel_b']; B=ls(aff(p['u']),p['v']); print('B',B.round(5))
p=s['panel_c']; u=p['u']; pairs=list(combinations(range(4),2)); powers=lambda z,d:np.column_stack([np.prod(z[:,p],axis=1) for p in combinations_with_replacement(range(4),d)])
cyc=lambda z:z*np.roll(z,-1,axis=1)
pair=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in pairs]);
funcs={'sin':lambda z,r:np.sin(r*z), 'cos': lambda z,r:np.cos(r*z),'tanh':lambda z,r:np.tanh(r*z),'rat':lambda z,r:z/(1+(r*z)**2),'soft':lambda z,r:z/(1+r*abs(z)), 'gauss':lambda z,r:np.exp(-(r*z)**2)}
extras={'cube':lambda z:z**3,'sq':lambda z:z**2,'cyc':cyc,'pair':pair,'poly2':lambda z:powers(z,2),'cyc3':lambda z:cyc(z)*z,'cyc3b':lambda z:cyc(z)*np.roll(z,-1,axis=1)}
for k in ['v1','v2']:
 v=p[k]; rows=[]
 for move in ['raw','B','invB']:
  transform={'raw':lambda z:z,'B':lambda z:aff(z)@B,'invB':lambda z:(z-B[0])@np.linalg.inv(B[1:])}[move]; w=transform(u)
  for aa in [(),('cube',),('sq',),('cyc',),('pair',),('poly2',),('cube','cyc'),('sq','cyc'),('cyc','cyc3'),('cyc','cyc3b')]:
   base=np.column_stack([aff(w)]+[extras[key](w) for key in aa]);
   for name in ['none']+list(funcs):
    def des(r):return base if name=='none' else np.c_[base,funcs[name](w,r)]
    def res(r):
     a=des(r); return a@ls(a,v)-v
    r=np.array([1.]) if name=='none' else min([least_squares(lambda x:res(x).ravel(),[r],bounds=(.08,6),max_nfev=35) for r in [.5,2.15,4.3]],key=lambda x:rm(x.fun)).x
    a=des(r); e=res(r); h=(a*np.linalg.pinv(a).T).sum(axis=1); score=rm(e/np.maximum(.001,1-h[:,None])); rows.append((score,rm(e),move,aa,name,round(r[0],4)))
 print(k,'best'); print(*sorted(rows)[:18],sep='\n'); print('raw no add',*[q for q in rows if q[2:4]==('raw',())],sep='\n')
 a=np.c_[aff(u),powers(u,2),u**3]; print('poly coef',ls(a,v).round(3)); print('poly powers',list(combinations_with_replacement(range(4),2)))
