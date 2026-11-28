import pickle,numpy as np,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
for k in ['panel_a','panel_b','panel_c']:
 print(k,{a:np.shape(v) for a,v in s[k].items()})
u=s['panel_c']['u']; print('panel range',u.min(0),u.max(0)); print('train range',s['train']['x'].min(0),s['train']['x'].max(0)); print('telemetry',s['telemetry_vec'])
base=np.c_[np.ones(len(u)),u]; pairs=list(it.combinations(range(4),2)); forms=[('prod',lambda a,b:a*b),('sum',lambda a,b:a+b),('diff',lambda a,b:a-b)]; funcs=[('id',lambda w:w),('sq',lambda w:w*w),('cube',lambda w:w**3),('tanh',np.tanh),('soft',lambda w:w/(1+abs(w))),('sin',np.sin),('cos',np.cos),('rat',lambda w:w/(1+w*w))]; cols=[];names=[]
for nm,fn in forms:
 for ij in pairs:
  w=fn(u[:,ij[0]],u[:,ij[1]])
  for g,f in funcs:
   for scale in ([1.] if g in ['id','sq','cube'] else [.3,.5,.85,1.25,1.7,2.15,2.4,3.]): cols.append(f(scale*w));names.append((nm,ij,g,scale))
A=np.column_stack(cols); A-=base@np.linalg.lstsq(base,A,rcond=None)[0]; A/=np.maximum(np.linalg.norm(A,axis=0),1e-12)
for k in [1,2]:
 v=s['panel_c']['v'+str(k)]; R=v-base@np.linalg.lstsq(base,v,rcond=None)[0]; rm=[]; J=[]
 for j in range(8):
  jj=np.argmax(np.linalg.norm(A.T@R,axis=1)); J.append(jj); F=np.c_[base,A[:,J]]; R=v-F@np.linalg.lstsq(F,v,rcond=None)[0]; rm.append((names[jj],np.mean(R**2)**.5))
 print(k,'pursuit',rm)
