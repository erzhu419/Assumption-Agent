exec(open('/tmp/grammar.py').read().split('f=PolynomialFeatures(5)')[0])
p=s['panel_c']; u=p['u']; U=np.c_[np.ones(len(u)),u]; Y=p['v1']; V=p['v2'];
for k,target in [('v1',Y),('v2',V)]:
 results=[]
 for typ in ['diag2','diag3','diag4','cross','diag2cross','diag3cross','trig','tanh','mixtrig','u_times_tanh']:
  for a in ([.3,.6,1,1.5,2] if 'trig' in typ or 'tanh' in typ else [1.]):
   def feat(q):
    c=np.c_[np.ones(len(q)),q]; cross=np.column_stack([q[:,i]*q[:,j] for i in range(4) for j in range(i+1,4)]); diag2=np.c_[c,q*q]; diag3=np.c_[diag2,q**3]; h={'diag2':diag2,'diag3':diag3,'diag4':np.c_[diag3,q**4],'cross':np.c_[c,cross],'diag2cross':np.c_[diag2,cross],'diag3cross':np.c_[diag3,cross],'trig':np.c_[diag2,np.sin(a*q)-a*q],'tanh':np.c_[diag2,np.tanh(a*q)-a*q],'mixtrig':np.c_[c,np.sin(a*q)-a*q,cross], 'u_times_tanh':np.c_[c,q*np.tanh(a*q)]}[typ]; return h
   F=feat(u); W=np.linalg.lstsq(F,target,rcond=None)[0]; H=np.sum(F*np.linalg.pinv(F).T,axis=1); e=(F@W-target)/(1-H[:,None]); results.append((np.sqrt(np.mean(e*e)),typ,a,W,feat(xf)@W))
 results.sort(key=lambda r:r[0]); print(k,[(round(e,6),t,a) for e,t,a,*_ in results[:8]])
 if k=='v2': V2=results[0][-1]
 else: V1=results[0][-1]
# v1 = affine plus transformed affine/radial/coordinate term, profiled over linear coefficients.
for typ in ['tanh','sin','tanh_cross','sin_cross','square_ridge','radial']:
 def one(q):
  if typ.startswith('tanh'): return np.tanh(q)
  if typ.startswith('sin'): return np.sin(q)
  if typ=='square_ridge':return q*q
  return q/(1+q*q)
 C=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]); D=np.column_stack([xf[:,i]*xf[:,j] for i in range(4) for j in range(i+1,4)]); F=np.c_[U,C] if typ.endswith('cross') else U; G=np.c_[np.ones(len(xf)),xf,D] if typ.endswith('cross') else np.c_[np.ones(len(xf)),xf]; ip=np.linalg.pinv(F); preds=[]; loo=[]
 for targ in Y.T:
  w=np.linalg.lstsq(U,targ,rcond=None)[0]; best=None
  for a in [.4,.9,1.5]:
   for sign in [-1.,1.]:
    def r(q):
     rem=targ-q[0]*one(U@q[1:]); return rem-F@(ip@rem)
    fit=least_squares(r,np.r_[sign,a*w],bounds=(-5,5),max_nfev=200)
    if best is None or fit.cost<best.cost: best=fit
  q=best.x; preds.append(q[0]*one(np.c_[np.ones(len(xf)),xf]@q[1:])+G@(ip@(targ-q[0]*one(U@q[1:])))); J=np.c_[F,best.jac]; h=np.sum(J*np.linalg.pinv(J).T,axis=1); loo.append((np.mean(best.fun**2)**.5,np.mean((best.fun/np.maximum(1-h,.03))**2)**.5))
 print('ridge',typ,np.round(loo,5));
 if typ=='tanh_cross':V1r=np.array(preds).T
P=lambda X,d:PolynomialFeatures(d).fit_transform(X)
F0=P(x,3); F0[:,15:]*=.5
F1=P(xc/np.sqrt(1+.09*xc**2),3); F2=P(np.tanh(.3*xc)/.3,3)
lista=[('raw',F0),('C_a',F1),('C_b',F2)]
for nm,zz in [('v1_nonlin',V1),('v1_ridge',V1r),('v2_nonlin',V2)]:
 for d in [1,2,3]:
  lista.append((nm+str(d),P(zz,d)));lista.append(('mix'+nm+str(d),np.c_[F1,P(zz,d)]))
lista.extend([('blend',np.c_[F0*.4,F1,F2]),('dual',np.c_[F1,V1r,V2, (V1r[:,:,None]*V2[:,None,:]).reshape(len(x),-1)]),('mult',np.c_[F1,V2,(xc[:,:,None]*V2[:,None,:]).reshape(len(x),-1)])])
ranks=[]
for label,F in lista:
 for a in [.003,.012,.035]:
  T=scores(F,a); ranks.append((T[2].sum(),label,a,T))
for _,label,a,T in sorted(ranks,key=lambda r:r[0])[:18]: print(label,a,T.round(5).tolist())
