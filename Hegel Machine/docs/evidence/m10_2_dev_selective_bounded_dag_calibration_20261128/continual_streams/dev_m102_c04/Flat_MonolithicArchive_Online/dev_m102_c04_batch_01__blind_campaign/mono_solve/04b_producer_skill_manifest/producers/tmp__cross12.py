from dag import *
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']);print('WQ',wq,'WH',wh)
Y=np.c_[y,t]; D=paths(x,1); n=52; out=[]
def tt(nm,f):
 f=np.c_[np.ones(len(x)),f]; f=f/np.sqrt(np.mean(f[:n]**2,axis=0)).clip(1e-10); a=f[:n]; u,sv,vt=np.linalg.svd(a,full_matrices=False); uy=u.T@Y[:n];
 for l in [.008,.18,1.5]:
  d=sv**2/(sv**2+l); r=(Y[:n]-u@(d[:,None]*uy))/(1-(u*u)@d)[:,None]; pr=f[n:]@vt.T@((d/sv.clip(1e-10))[:,None]*uy); h=abs(pr-Y[n:]); out.append((np.mean(abs(r[:,0])),np.mean(abs(r[:,1:])),h[:,0].mean(),h[:,1:].mean(),nm,l))
def ints(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for j,z in D.items():
 if len(j)>1 and j[-1] not in 'Bb':continue
 for name,p,h in [('C',Q(z),H(z)),('CB',Q(z),H(B(z))),('BC',Q(B(z)),H(z)),('Cb',Q(z),H(Bi(z))),('bC',Q(Bi(z)),H(z)),('split',z,Q(z)),('splitH',z,H(z)),('pp',Q(z),Q(B(z))),('hh',H(z),H(B(z)))]:
  for k in [0,.22,1.25,-.22]:
   if k>=0: r=p/(1+k*h*h); v=h/(1+k*p*p)
   else: r=p/(1-k*abs(h)); v=h/(1-k*abs(p))
   for label,f in [('dot',np.c_[z,r,v,np.sum(r*v,1)]),('diag',np.c_[z,r,v,r*v]),('full',np.c_[z,r,v,ints(r,v)]),('zp',np.c_[z,r,v,ints(z,r)]),('zpzv',np.c_[z,r,v,ints(z,r),ints(z,v)]),('quad',poly(np.c_[r,v],2)[:,1:]),('quadplus',np.c_[z,poly(np.c_[r,v],2)[:,1:]])]:tt((j,name,k,label),f)
   for g in [0,.22,1.,-1.]:
    if g==0:continue
    m=r+g*v
    for label,a in [('chainQ',Q(m)),('chainH',H(m)),('chainBHQ',Q(H(B(m)))),('tanh',np.tanh(1.25*m)),('sin',np.sin(2.4*m)),('square',m*m)]:tt((j,name,k,g,label),np.c_[z,r,v,a,a*r,a*v])
for i in range(4):print('rank',i,*sorted(out,key=lambda q:q[i])[:8],sep='\n');
