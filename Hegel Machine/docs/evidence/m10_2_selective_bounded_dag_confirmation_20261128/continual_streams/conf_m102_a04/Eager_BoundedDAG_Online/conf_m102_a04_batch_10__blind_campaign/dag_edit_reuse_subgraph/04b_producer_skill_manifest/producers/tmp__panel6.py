from hegel_probe import *
from itertools import combinations
for key in ['panel_a','panel_c']:
 p0=p[key]; u=p0['u']; v=p0.get('v',p0.get('v1')); ok=np.ones(len(u),bool)
 if key=='panel_a':
  for i in range(5):
   F=a.one(u); r=np.linalg.norm(v-F@a.lin(F[ok],v[ok]),axis=1); ok=r<2
 u,v=u[ok],v[ok]; F=a.one(u); R=v-F@a.lin(F,v)
 fs={'sq':u**2,'cube':u**3,'quart':u**4,'abs':abs(u),'oddabs':u*abs(u),'sin':np.sin(2*u),'tanh':np.tanh(2*u),'soft':u/(1+abs(u)),'pair':np.c_[[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]].T,'exp':np.exp(u),'rat_sq':u**2/(1+.25*u**2)}
 print('\n',key)
 for k,w in fs.items():
  G=np.c_[F,w]; print(k,np.sqrt(np.mean((v-G@a.lin(G,v))**2,axis=0)),np.round(a.lin(G,v)[5:],3))
 for ks in [('sq','cube'),('sq','pair'),('sq','cube','pair'),('sq','tanh'),('sq','soft'),('sq','rat_sq'),('cube','pair'),('pair','tanh')]:
  G=np.column_stack([F]+[fs[k] for k in ks]); print(ks,np.sqrt(np.mean((v-G@a.lin(G,v))**2)))
 if key=='panel_a':
  for j in range(4):
   print('channel',j,'best correlated residual columns')
   cc=[]
   for k,w in fs.items():
    w=w-F@a.lin(F,w); corr=(w.T@R[:,j])/np.maximum(np.linalg.norm(w,axis=0)*np.linalg.norm(R[:,j]),1e-12)
    cc+=sorted([(round(abs(c),3),k,i,round(c,3)) for i,c in enumerate(corr)],reverse=True)
   print(sorted(cc,reverse=True)[:12])
