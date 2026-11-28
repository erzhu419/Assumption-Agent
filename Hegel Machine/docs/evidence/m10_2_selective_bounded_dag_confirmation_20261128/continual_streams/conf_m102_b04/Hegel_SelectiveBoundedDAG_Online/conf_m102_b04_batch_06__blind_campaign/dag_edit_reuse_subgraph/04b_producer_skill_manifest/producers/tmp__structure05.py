exec(open('/tmp/probe05.py').read().split("a=s['panel_a']")[0])
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def lib(u,a=.165,deg=3):
 names=['1'];cols=[np.ones(len(u))]; d=u.shape[1]
 for k in range(1,deg+1):
  for j in combinations_with_replacement(range(d),k): names.append(''.join(map(str,j)));cols.append(np.prod(u[:,j],axis=1))
 return names,np.column_stack(cols)
def sparse(D,y,k):
 sel=[]
 for _ in range(k):
  r=y-D[:,sel]@fit(D[:,sel],y) if sel else y
  residual=D-D[:,sel]@fit(D[:,sel],D) if sel else D
  c=abs(residual.T@r)/np.maximum(np.linalg.norm(residual,axis=0),1e-12);c[sel]=-1
  sel.append(int(np.argmax(c)))
 return sel,fit(D[:,sel],y)
for p in 'ab':
 a=s['panel_'+p];u,v=a['u'],a['v'];
 if p=='a':
  keep=(abs(v-u).max(1)<3);u,v=u[keep],v[keep]
 print('PANEL',p,'range',u.min(0),u.max(0), 'n',len(u));n,D=lib(u)
 for d in [1,2,3]:
  _,Q=lib(u,deg=d); f=fit(Q,v); print('poly',d,np.sqrt(np.mean((Q@f-v)**2)),np.sqrt(np.mean((Q[::5]@fit(Q[np.arange(len(u))%5!=0],v[np.arange(len(u))%5!=0])-v[::5])**2)))
 for j in range(4):
  ix,b=sparse(D,v[:,j],10);print(j,list(zip(np.array(n)[ix],b.round(4))),np.std(D[:,ix]@b-v[:,j]))
 print('u, v',np.c_[u,v][:6])
 print('singular',np.linalg.svd(v-v.mean(0),compute_uv=False))

x,y,z=[s['train'][k] for k in ('x','y','traj')]; q=s['test']['x']; print('study x range',x.min(0),x.max(0));
for p in 'abc':
 a=s['panel_'+p]
 for k,v in a.items():
  if v.shape[1]==4: print('dist',p,k, np.quantile(cdist(x,v).min(1),[0,.05,.25,.5,1]))
# Probe raw and calibrated coordinates with exact observed nonlinearities. 
a=s['panel_a'];keep=abs(a['v']-a['u']).max(1)<3; u,v=a['u'][keep],a['v'][keep]; W=fit(aff(u),v); X=np.vstack([x,q]); variants={'raw':X, 'Adiag':(X-W[0])/W[1:].diagonal(),'Ainv':(X-W[0])@np.linalg.inv(W[1:])}
for d in (2,3):
 def F(w):return lib(w,deg=d)[1]
 co=fit(F(u),v)
 variants['Ainv'+str(d)]=np.array([least_squares(lambda a:(F(a[None])@co-r).ravel(),a, max_nfev=45).x for r,a in zip(X,variants['Ainv'])]);variants['Adirect'+str(d)]=F(X)@co
b=s['panel_b']; c=s['panel_c']; ac=2.1143; alpha=.1653
P=lambda x:aff(np.arctan(ac*x)); Q=lambda x:np.c_[aff(x),x*x/(1+alpha*x*x)]
wc=fit(P(c['u']),c['v1']);wd=fit(Q(c['u']),c['v2']);
for k,w in list(variants.items()):
 for d in (1,2,3):variants[k+'/B'+str(d)]=poly(w,d)@fit(poly(b['u'],d),b['v'])
Y=np.c_[y,z];ii=np.arange(52); jj=np.arange(52,72); res=[]
for key,w in variants.items():
 for name,v in [('id',w),('atan',np.arctan(ac*w)),('C1',P(w)@wc),('C2',Q(w)@wd),('amp',np.c_[P(w),Q(w)@wd])]:
  for d in (1,2,3):
   D=poly(v,d) if v.shape[1]<=4 else aff(v); A=D[ii];B=D[jj];
   for l in (.00001,.002,.025,.2):
    coef=np.linalg.solve(A.T@A+l*np.eye(A.shape[1]),A.T@Y[ii]); e=abs(B@coef-Y[jj]).mean(0);res.append((e[0],e[1:].mean(),key,name,d,l))
for i in [0,1]:print('BEST',i,*sorted(res,key=lambda x:x[i])[:15],sep='\n')
