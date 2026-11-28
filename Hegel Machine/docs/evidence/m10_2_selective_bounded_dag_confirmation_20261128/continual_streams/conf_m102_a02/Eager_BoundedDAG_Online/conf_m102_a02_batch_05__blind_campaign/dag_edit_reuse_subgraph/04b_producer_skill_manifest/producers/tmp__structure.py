exec(open('/tmp/probe.py').read().split("print('cal',m)")[0]); z=(x-m[:,1])/m[:,0]
from itertools import combinations as cc
from scipy.linalg import qr
from scipy.optimize import least_squares
A=lambda u:np.c_[np.ones(len(u)),u]
def bases(u):
 pair=np.column_stack([u[:,i]*u[:,j] for i,j in cc(range(4),2)])
 triple=np.column_stack([np.prod(u[:,ix],1) for ix in cc(range(4),3)])
 return {'sq':u**2,'pairs':pair,'triple':triple,'cube':u**3,'mixed3':np.column_stack([u[:,i]**2*u[:,j] for i in range(4) for j in range(4) if i!=j]),'quart':u**4,'fifth':u**5,'sin3':np.sin(3*u),'sin5':np.sin(5*u),'cos3':np.cos(3*u),'tanh3':np.tanh(3*u),'ratio':u/(1+u*u),'sign':np.sign(u),'abs':abs(u),'sinpair':np.sin(3*pair),'sintriple':np.sin(3*triple)}
def show(u,V):
 B=bases(u)
 def meas(k):
  F=np.column_stack([A(u)]+[B[j] for j in k]); H=F@np.linalg.pinv(F); E=(V-H@V)/(1-np.diag(H))[:,None]; return np.sqrt(np.mean(E**2)),np.sqrt(np.mean((V-H@V)**2))
 print('single blocks',sorted((tuple(np.round(meas([k]),5)),k) for k in B))
 for n in (2,3): print('multi',n,sorted((tuple(np.round(meas(k),5)),k) for k in cc(B,n) if 5+sum(B[j].shape[1] for j in k)<len(u)-3)[:7])
for j,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]: print(j,k);show(s[j]['u'],s[j][k]);
for j in [0,8]: print('output',j);show(z,Y[:,j:j+1]);
for k,V in [('b',s['panel_b']['v']),('c1',s['panel_c']['v1'])]:
 u=s['panel_b' if k=='b' else 'panel_c']['u']; F=np.c_[A(u),u**2,u**3]; c=np.linalg.lstsq(F,V,rcond=None)[0]; print(k,'pure power fit',np.sqrt(np.mean((F@c-V)**2)),c.T)
print('times',s.get('eval_times'))
