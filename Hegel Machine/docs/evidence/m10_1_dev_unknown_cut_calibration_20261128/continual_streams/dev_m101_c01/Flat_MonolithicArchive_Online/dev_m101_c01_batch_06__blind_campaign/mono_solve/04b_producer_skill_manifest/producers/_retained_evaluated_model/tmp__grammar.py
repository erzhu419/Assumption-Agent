exec(open('/tmp/cgraph.py').read().split('for fun,V,start,lim in')[0])
print('times',t,'telemetry',s['telemetry_vec']); print('B',B,'RMS',np.std(s['panel_b']['v']-poly(s['panel_b']['u'],1)@B)); print('A',A)
r1,L1,_=prof(c1,v,[2.2]*4+[0]*4,([.4]*4+[-1]*4,[5]*4+[1]*4),slice(None));
def c2(U,r): return np.c_[poly(U,1),np.cos(U*r)]
r2,L2,_=prof(c2,w,[1.3]*4,([.2]*4,[4]*4),slice(None)); print('C2 rms',np.std(c2(u,r2)@L2-w),r2)
rng=np.random.default_rng(160); folds=[np.arange(56,72)]+np.array_split(rng.permutation(n),5)+np.array_split(rng.permutation(n),5)
mode=np.linalg.svd(z,full_matrices=False)[2][0]; out=np.c_[y,z@mode]; eye=np.eye(n); P=poly(a,2); E=out-P@ls(P,out); print('quadratic residual correlation',np.corrcoef(E.T))
def score(K, Y=out):
 errors=[]
 for iv in folds:
  it=np.setdiff1d(np.arange(n),iv); h=K[np.ix_(iv,it)]@cho_solve(cho_factor(K[np.ix_(it,it)]),Y[it]); errors.append(np.mean(abs(h-Y[iv]),axis=0))
 e=np.array(errors); return e.mean(0)+.35*e.max(0),np.round(np.array([e[0],e.mean(0),e.max(0)]),5)
def gram(Q):return Q@Q.T
base=gram(P); ans=[]
def test(nm,Q,base=base):
 G=gram(Q); G*=n/np.trace(G)
 for scale in [.05,.5,5,50]:
  for nug in [.00015,.001,.006]:
   S,E=score(base+scale*G+nug*eye);ans.append((S,E,nm,scale,nug))
coords={'x':x,'a':a,'b':b,'binv':(a-B[0])@np.linalg.inv(B[1:])}
for nm,X in coords.items():
 C=c1(X,r1)@L1; D=c2(X,r2)@L2
 for name,W in [('C1',C),('C2',D),('Csum',C+D),('Cprod',C*D),('Cjoin',np.c_[C,D]),('tanh',np.tanh(1.5*X)),('sin',np.sin(1.3*X)),('cos',np.cos(1.3*X)),('soft',X/(1+.5*X*X))]:
  test((nm,name,'quadratic'),poly(W,2)); test((nm,name,'cross'),np.c_[W,(W[:,:,None]*X[:,None,:]).reshape(n,-1)]); test((nm,name,'mixed'),poly(np.c_[X,W],2));
 for name,U in [('linear',X),('pair',np.column_stack([X[:,i]+sign*X[:,j] for i in range(4) for j in range(i+1,4) for sign in [-1,1]]))]:
  for form,V in [('rat',1/(1+.5*U*U)),('tanh',np.tanh(U)),('sin',np.sin(U))]:
   test((nm,name,form), (V[:,:,None]*poly(X,2)[:,None,:]).reshape(n,-1))
 test((nm,'poly3'),poly(X,3)); test((nm,'poly4'),poly(X,4));
 for length in [1.2,2,3]:
  d=cdist(X,X,'sqeuclidean')/length**2; rr=np.sqrt(5*d); G=(1+rr+rr**2/3)*np.exp(-rr); S,E=score(base+G+.0004*eye);ans.append((S,E,(nm,'MAT',length),1,.0004))
for j in range(2):
 print('OUTPUT',j); seen=set()
 for S,E,nm,weight,nug in sorted(ans,key=lambda item:item[0][j]):
  if nm in seen:continue
  seen.add(nm); print(nm,weight,nug,E[:,j]);
  if len(seen)>=10:break
