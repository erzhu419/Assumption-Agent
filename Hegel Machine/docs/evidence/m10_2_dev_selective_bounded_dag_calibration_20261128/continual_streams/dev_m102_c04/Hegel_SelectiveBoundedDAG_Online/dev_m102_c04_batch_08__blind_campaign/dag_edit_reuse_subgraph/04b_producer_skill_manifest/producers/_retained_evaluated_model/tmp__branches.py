exec(open('/tmp/spatial.py').read().split('ranks=')[0])
def pp(z):return np.c_[z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]]
def rr(z):return np.cos(k*z)
wp=L(A(pp(u)),C['v1']); wr=L(A(rr(u)),C['v2']); n=52; T=np.c_[y,t]; rank=[]
trans={key:z for key,z in trans.items() if len(key.split())<=2 or key in ('B- A- B+','B- A+ B+','B+ A- B-','B+ A+ B-')}
for pn,z in trans.items():
 for rn,q in trans.items():
  p=A(pp(z))@wp;r=A(rr(q))@wr
  for fn,F in [('ind',np.c_[p,r]),('dot',np.c_[p,r,np.sum(p*r,axis=1)]),('mix',np.c_[p,r,p*r]),('cross',np.c_[p,r,(p[:,:,None]*r[:,None,:]).reshape(-1,16)]),('latent',np.c_[pp(z),rr(q),pp(z)*rr(q)]),('rat',np.c_[p,r,p/(1+r*r),r/(1+p*p)]),('trig',np.c_[p,r,np.sin(p),np.sin(r),np.cos(p),np.cos(r)])]:
   for zn,raw in [('none',x[:,:0]),('raw',x)]:
    D=A(np.c_[raw,F]);w0=L(D[:n],T[:n]);e=abs(D[n:]@w0-T[n:]); et=abs(D[:n]@w0-T[:n]); rank.append((e[:,0].mean(),e[:,1:].mean(),pn,rn,fn,zn,et[:,0].mean(),et[:,1:].mean()))
for j in range(2):print('branch rank',j,*sorted(rank,key=lambda r:r[j])[:10],sep='\n')
