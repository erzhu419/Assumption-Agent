from calprobe import *

p=s['panel_b'];u,v=p['u'],p['v'];z=calibration(s['panel_a'])[2](x)
print('times, telemetry',s['eval_times'],s['telemetry_vec'])
fs={
'exp': (lambda a,k:a*np.exp(-k*a*a),.99),
'sin': (lambda a,k:np.sin(k*a),2.82),
'rat': (lambda a,k:a/(1+k*a*a),.74),
 'tanh':(lambda a,k:np.tanh(k*a),1.37),
}
for kind,(f,k0) in fs.items():
 for split in ('untied','shared','by_input'):
  def design(a,q):
   k=np.exp(q[0]); h=f(a,k)
   return np.c_[aff(a),h] if split=='untied' else aff(a+q[1:]*h)
  def err(q):
   d=design(u,q);return (d@fit(d,v)-v).ravel()
  q0=np.r_[np.log(k0),[] if split=='untied' else np.full(1 if split=='shared' else 4,-.4 if kind=='sin' else -1.)]
  sol=least_squares(err,q0,bounds=(np.r_[-3,np.full(len(q0)-1,-5.)],np.r_[2,np.full(len(q0)-1,5.)]),max_nfev=300)
  d=design(u,sol.x);b=design(z,sol.x)@fit(d,v); a,g,h,j=b.T; D=aff(b)
  L=np.c_[D,a*h+g*j]; vy=Y[:,0]; tr=Y[:,1:]; e=vy-L@fit(L,vy)
  print(kind,split,np.r_[np.exp(sol.x[0]),sol.x[1:]],'panel rms',np.std(sol.fun),'CV y,state',np.mean(abs(vy[52:]-L[52:]@fit(L[:52],vy[:52]))),np.mean(abs(tr[52:]-D[52:]@fit(D[:52],tr[:52]))),'readout rms',np.std(e),'label coef',fit(L,vy))
  if kind=='exp' and split=='shared':
   t=s['eval_times'];
   def basis(q):return -np.expm1(-np.exp(q)[:,None]*t)
   def time_err(q):
    G=basis(q);return (tr[:52]@np.linalg.pinv(G)@G-tr[:52]).ravel()
   rates=least_squares(time_err,np.log([.4,1.8]),bounds=(-4,3)).x; G=basis(rates)
   print('rates',np.exp(rates),'time residual rms',np.std(time_err(rates)),'readout rms',np.std(tr-D@fit(D,tr)),'end errors',np.mean(abs(tr[52:,-1]-D[52:]@fit(D[:52],tr[:52]@np.linalg.pinv(G))@G[:,-1])))
