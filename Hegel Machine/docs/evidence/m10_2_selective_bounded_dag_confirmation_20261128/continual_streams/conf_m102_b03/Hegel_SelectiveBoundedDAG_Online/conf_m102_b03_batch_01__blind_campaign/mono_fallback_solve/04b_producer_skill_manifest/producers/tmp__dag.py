exec(open('/tmp/explore.py').read().split('a=s[')[0])
a=s['panel_a']; p=s['panel_b']; c=s['panel_c']; u=a['u'];v=a['v']; f=ones(u);g=np.linalg.norm(f@ls(f,v)-v,axis=1)<3; A=ls(f[g],v[g]); B=ls(ones(p['u']),p['v']); I=[0,0,1,2];J=[1,3,2,3]
def q(z):return ones(z[:,I]*z[:,J])
def co(z):return ones(np.cos(2.4*z))
C=ls(q(c['u']),c['v1']); D=ls(co(c['u']),c['v2']);
def Q(z):return q(z)@C
def R(z):return co(z)@D
def L(z):return ones(z)@B
x=s['train']['x']; z=(x-A[0])/np.diag(A[1:]); T=s['eval_times']; H=-np.expm1(-T[:,None]*[.37,1.78,5.42]); ts=s['train']['traj']; y=s['train']['y']; Y=np.c_[y,ts]; S=np.linalg.svd(ts,full_matrices=False)[2][:3]; YY=np.c_[y,ts@S.T];
ops={'L':L,'Q':Q,'R':R,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'soft':lambda w:w/(1+abs(w)),'clip':lambda w:np.clip(w,-1,1)}
F={'z':z,'x':x};
for n in ['z','x']:
 for o,c1 in ops.items():F[n+':'+o]=c1(F[n]);
for n in list(F):
 if ':' not in n:continue
 for o,c1 in ops.items():F[n+':'+o]=c1(F[n]);
for n in list(F):
 if n.count(':')!=2:continue
 for o,c1 in ops.items():
  if o in ['L','Q','R'] or n.split(':')[-1] in ['L','Q','R']:F[n+':'+o]=c1(F[n]);
U={}; inds=np.triu_indices(4,1)
def tr(w,order):
 if order==0:return w
 if order==1:return np.c_[w,w*w]
 if order==2:return np.c_[w,w[:,inds[0]]*w[:,inds[1]]]
 return np.c_[w,w[:,np.triu_indices(4)[0]]*w[:,np.triu_indices(4)[1]]]
def score(w,l=0):
 w=ones(w); a=w[:52];b=w[52:]; beta=ls(a,Y[:52]) if not l else np.linalg.solve(a.T@a+l*np.eye(a.shape[1]),a.T@Y[:52]); r=abs(w@beta-Y); return r[52:,0].mean(),r[52:,1:].mean(),r[:52,0].mean(),r[:52,1:].mean()
def show(T,k=10):
 for j in [0,1,2,3]:print('sort',j,*sorted(T,key=lambda row:row[j])[:k],sep='\n')
print('B/C matrices',B,C,D,sep='\n'); print('train outcome stats',Y.mean(0),Y.std(0));
print('single block');show([(*score(tr(v,i)),n,i) for n,v in F.items() for i in range(4)],6)
G={n:v for n,v in F.items() if n.startswith('z') and n.count(':')<3 and (n in ['z','z:L','z:Q','z:R'] or n.split(':')[1] in ['Q','R','L'])}; ans=[]
for a,(n,v) in enumerate(G.items()):
 for m,w in list(G.items())[a+1:]:
  h=v[:,:,None]*w[:,None,:];
  for i,V in enumerate([np.c_[v,w],np.c_[v,w,np.sum(v*w,1)],np.c_[v,w,v*w],np.c_[v,w,h.reshape(len(z),-1)],np.c_[tr(v,3),tr(w,3)],np.c_[tr(v,3),tr(w,3),v*w]]):ans.append((*score(V),n,m,i))
print('double block');show(ans,9)
for n in ['z','z:L','z:Q','z:R','z:L:Q','z:L:R','z:Q:L','z:R:L','z:Q:R','z:R:Q']:
 w=F[n]; print(n,'corr',np.corrcoef(np.c_[YY,w].T)[:4,4:],sep='\n')
