exec(open('/tmp/explore.py').read().split('for key')[0]); from itertools import product
A=s['panel_a']; keep=np.linalg.norm(A['u']-A['v'],axis=1)<2; z=(x-np.array([np.polyfit(A['u'][keep,j],A['v'][keep,j],1)[1] for j in range(4)]))/np.array([np.polyfit(A['u'][keep,j],A['v'][keep,j],1)[0] for j in range(4)]); B=s['panel_b']; C=s['panel_c']; b=np.linalg.lstsq(pol(B['u'],1),B['v'],rcond=None)[0]; p=C['u']; pp=pol(p,2); mats=[]
for j in ('v1','v2'):
 v=C[j]; w=np.linalg.lstsq(pp,v,rcond=None)[0]; print(j,'quad err',rms(v-pp@w),'weights\n',w); mats.append(w)
 for mode in ('tanh','sin','sat','atan'):
  fn={'tanh':np.tanh,'sin':np.sin,'sat':lambda r:r/(1+abs(r)),'atan':np.arctan}[mode]; P=pol(p,1); out=[]
  for i in range(4):
   targ=v[:,i]; a=np.linalg.lstsq(P,targ,rcond=None)[0]; sol=least_squares(lambda a: a[5]*fn(P@a[:5])+a[6]-targ,np.r_[a,1.6,0.],max_nfev=300); out.append(rms(sol.fun))
  print(mode,out)
q=-np.expm1(-t[:,None]*np.array([.413,1.811])); Y=np.c_[y,T@np.linalg.pinv(q.T)];
Z={'raw':x,'z':z,'b':pol(z,1)@b,'bi':(z-b[0])@np.linalg.inv(b[1:]),'c1':pol(z,2)@mats[0],'c2':pol(z,2)@mats[1]};
for k1 in list(Z):
 r=Z[k1]; Z['c1('+k1+')']=pol(r,2)@mats[0]; Z['c2('+k1+')']=pol(r,2)@mats[1]; Z['b('+k1+')']=pol(r,1)@b
rows=[]
for k,f0 in Z.items():
 for d in (2,3,4):
  f=pol(f0,d); f/=np.maximum(rms(f[:n]),1e-7); K=f[:n]@f[:n].T; Kt=f[n:]@f[:n].T
  for lam in (.001,.03,.3):
   pred=Kt@np.linalg.solve(K+lam*np.eye(n),Y[:n]); err=np.r_[mae(pred-Y[n:]),mae(((pred-Y[n:])[:,1:]@q.T).ravel())];rows.append((k,d,lam,err))
for i in range(4):
 print('sort',i);print(*sorted(rows,key=lambda a:a[-1][i])[:9],sep='\n')
