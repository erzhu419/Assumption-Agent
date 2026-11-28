exec(open('/tmp/check_batch.py').read().split('for name,a,b in splits:')[0])
z,f,g,q=m._spatial(s,x)
fg=(f[:,:,None]*g[:,None,:]).reshape(n,16)
base=np.c_[z,q]
times=s['eval_times']; print('times',times)
for name,a,b in (splits[0],splits[5],splits[-1]):
 h=m._time_fit(times,t[a]); amps=m._fit(h,t[a].T).T
 D=m._aff(base,fg)
 c=m._fit(D[a],amps)
 print(name,'rate',-np.log1p(-h[0])/times[0],'condition',np.linalg.cond(D[a])); print('base',np.round(c[:15].T,3));print('fg slow/fast',np.round(c[15:].T.reshape(2,4,4),3))
 print('temporal projection MAE',abs(t[a]-amps@h.T).mean(0))
sets={'full':np.c_[base,fg],'diag+cyclic':np.c_[base,f*g,f*np.roll(g,-1,axis=1),f*np.roll(g,1,axis=1)],'diag+summed shifts':np.c_[base,f*g,*[(f*np.roll(g,i,axis=1)).sum(axis=1) for i in (1,2,3)]],'diag+f':np.c_[base,f*g,f],'diag+g':np.c_[base,f*g,g]}
for key,D in sets.items():
 vals=[]
 for _,a,b in splits:
  h=m._time_fit(times,t[a]); amps=m._fit(h,t[a].T).T
  p=m._ridge(m._aff(D[np.r_[a,b]]),amps,len(a),.01)[len(a):]@h.T
  e=abs(p-t[b]); vals.append([e.mean(),e.mean(0).max()])
 print(key,np.round(vals,5).tolist(),'max',np.max(vals,axis=0))
