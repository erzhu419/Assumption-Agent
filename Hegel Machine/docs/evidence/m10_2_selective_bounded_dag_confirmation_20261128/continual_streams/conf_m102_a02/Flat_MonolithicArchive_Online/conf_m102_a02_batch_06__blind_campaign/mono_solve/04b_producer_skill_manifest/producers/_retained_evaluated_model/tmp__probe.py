exec(open('/tmp/discover.py').read().split('forms=')[0])
from sklearn.linear_model import OrthogonalMatchingPursuit
u,v=s['panel_b']['u'],s['panel_b']['v']; print('B ranges',u.min(0),u.max(0)); print('B linear',ls(aff(u),v));
fs={}
for p in [.22,.5,.75,1,1.25,1.5,2,2.5,3]:
 for n,g in [('exp',lambda z,p:z*np.exp(-p*z*z)),('expa',lambda z,p:z*np.exp(-p*abs(z))),('rat',lambda z,p:z/(1+p*z*z)),('rt',lambda z,p:z/np.sqrt(1+p*z*z)),('tan',lambda z,p:np.tanh(p*z)),('sin',lambda z,p:np.sin(p*z))]:fs[n+str(p)]=lambda z,g=g,p=p:g(z,p)
fs['odd7']=lambda z:np.c_[z**3,z**5,z**7]; fs['odd9']=lambda z:np.c_[z**3,z**5,z**7,z**9];
ops=[]
for name,f in fs.items():
 for tag,g in [('aff',aff),('cyc',lambda z:np.c_[aff(z),z*np.roll(z,-1,axis=1)]),('cube',lambda z:np.c_[aff(z),z**3]),('xy',lambda z:np.c_[aff(z),z[:,ia]*z[:,ja]])]:
  def gg(z,f=f,g=g):return np.c_[g(z),f(z)]
  A=gg(u); P=np.linalg.pinv(A); w=P@v; err=A@w-v; hat=np.einsum('ij,ji->i',A,P); loo=np.mean(abs(err)/np.maximum(1-hat[:,None],1e-8)); ops.append((loo,name,tag,gg,w));
for l,n,k,g,w in sorted(ops,key=lambda a:a[0])[:10]: print('B op',l,n,k,np.round(w,3) if k=='aff' and n in ('exp1','exp1.25') else '')
b=min(ops,key=lambda a:a[0]); h=b[3](x)@b[4];
p=s['panel_c']; U=p['u']; cfun=lambda q:np.c_[aff(q),q*np.exp(-1.0223*abs(q)**.9585)]; dfun=lambda q:np.c_[aff(q),q*q,np.exp(-.7109*q*q)]; C=lambda q:cfun(q)@ls(cfun(U),p['v1']); D=lambda q:dfun(q)@ls(dfun(U),p['v2']);
Y=np.c_[s['train']['y'],s['train']['traj']]; H=quad(h); E=Y-H@ls(H[:52],Y[:52]); print('h range',h.min(0),h.max(0)); print('quadratic weights y/end',np.round(ls(H[:52],Y[:52])[:,[0,-1]],3)); print('errors B',np.abs(E[52:]).mean(0));
R=[]
for n,z in [('h',h),('x',x),('C(x)',C(x)),('D(x)',D(x)),('C(h)',C(h)),('D(h)',D(h)),('B(C(x))',b[3](C(x))@b[4]),('B(D(x))',b[3](D(x))@b[4]),('x/h',np.c_[x,h]),('C/D',np.c_[C(x),D(x)])]:
 for m,F in [('aff',aff(z)),('quad',np.c_[aff(z),*(z[:,k]*z[:,l] for k,l in it.combinations_with_replacement(range(z.shape[1]),2))]),('sep3',np.c_[aff(z),z*z,z**3])]:
  for add in [0,1]:
   A=np.c_[H,F[:,1:]] if add else F
   ss=np.maximum(np.linalg.norm(A[:52],axis=0),1e-9); A=A/ss
   for alpha in [.00001,.001,.01]:
    o=A@np.linalg.solve(A[:52].T@A[:52]+alpha*np.eye(A.shape[1]),A[:52].T@Y[:52]); e=abs(o-Y);R.append((e[52:,0].mean(),e[52:,1:].mean(),e[:52,0].mean(),e[:52,1:].mean(),n,m,add,alpha))
for j in (0,1):print('extension',j,sorted(R,key=lambda r:r[j])[:7]);
for n,g in fs.items():
 if n in ['exp1','exp1.25','rat1.25','tan1.25','odd7']:
  A=np.c_[aff(u),g(u)];w=ls(A,v); print('B',n,np.round(w,3),'res first',np.round((A@w-v)[:5],4))
print('ranges Y',Y.min(0),Y.max(0));print('summary',s['train']['summary'])
