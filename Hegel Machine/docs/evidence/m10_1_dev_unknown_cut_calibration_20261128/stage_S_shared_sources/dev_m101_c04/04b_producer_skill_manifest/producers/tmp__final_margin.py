exec(open('/tmp/check_candidate.py').read().split('ys={')[0])
splits += [(f'extrap{i}',np.argsort(z[:,i])[-8:]) for i in (0,1)]
splits += [('leverage',np.argsort(np.sum((S@np.linalg.pinv(S.T@S))*S,axis=1))[-12:])]
cal=m._calibration(s); B=cal[2].copy(); U,sig,V=np.linalg.svd(B[1:]); B[1:]=sig.mean()*(U@V)
zp=m._coordinates(x,(*cal[:2],B)); P=np.c_[m._affine(zp),m._cross(zp)]
sc={k:[] for k in ('shared','scaled_orthogonal','half_intercept','no_intercept')}
tr={k:[] for k in ('exp2','blend_25','blend_50','affine')}
for name,j in splits:
 i=np.setdiff1d(np.arange(n),j)
 y0=S[j]@m._ls(S[i],y[i]); yz=S[j,1:]@m._ls(S[i,1:],y[i]); yp=P[j]@m._ls(P[i],y[i]);
 for key,pred in zip(sc,(y0,yp,(y0+yz)/2,yz)): sc[key].append(abs(pred-y[j]).mean())
 ste=D[j]@m._relaxation(D[i],T[i],t); sta=D[j]@m._ls(D[i],T[i]);
 for key,pred in zip(tr,(ste,.75*ste+.25*sta,(ste+sta)/2,sta)): tr[key].append(abs(pred-T[j]).mean(0))
for key,vals in {**sc,**tr}.items():
 a=np.asarray(vals); idx=np.unravel_index(a.argmax(),a.shape); print(key,'mean',a.mean(),'worst',a[idx],splits[idx[0]][0],idx[1:],'upper tails and leverage',a[-3:].reshape(3,-1).mean(1),'terminal worst',a[:,-1].max() if a.ndim==2 else '-')
print('max scalar effect of scaled-orthogonal panel on public test',np.max(abs(np.c_[m._affine(test),m._cross(test)]@m._ls(S,y)-np.c_[m._affine(m._coordinates(s['test']['x'],(*cal[:2],B))),m._cross(m._coordinates(s['test']['x'],(*cal[:2],B)))]@m._ls(P,y))))
