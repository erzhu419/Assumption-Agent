exec(open('/tmp/separable.py').read().split('mods=[]')[0]); t0=np.arange(n); tv=np.arange(n,len(z)); soft=fit_sep(p,C['v1'],('soft',),1,np.arange(24)); print('soft fit',rms(soft[0]),soft[2],soft[3]);
D=np.cos(2.4*z); V=aff(D)@np.linalg.lstsq(aff(np.cos(2.4*p)),C['v2'],rcond=None)[0]; U=phi(z,('soft',),soft[3],1)@soft[2]; BZ=aff(z)@b
base=np.c_[aff(z),D]; W=np.linalg.lstsq(base[:n],Y[:n],rcond=None)[0]; print('basic weights',W); R=Y-base@W; print('basic train',rms(R[:n]),'test',mae(R[n:]), 'largest',np.argsort(abs(R[n:,0]))[-5:], 'time',t, 'telemetry',s['telemetry_vec']);
fs={};
def add(k,F): fs[k]=np.reshape(F,(len(z),-1))
for key,F in [('z',z),('soft',z/(1+.83*abs(z))),('U',U),('D',D),('V',V),('B',BZ)]:
 add(key,F)
 for i in range(4):
  for fn in ['sin','cos','tanh','exp','rat','soft']:
   for f in [.5,1,1.5,2,2.4,3]:add(f'{key}:{fn}{f}:{i}',func[fn](f*F[:,i]))
 for i,j in combinations(range(4),2):
  u,v=F[:,i],F[:,j]; add(f'{key}:{i}{j}',u*v);add(f'{key}:rat{i}{j}',u*v/(1+.5*(u*u+v*v)))
  for sgn in [-1,1]:
   for fn in ['sin','cos','tanh','rat']:
    for f in [.7,1.4,2.4]:add(f'{key}:{fn}{f}:{i},{sgn*(j+1)}',func[fn](f*(u+sgn*v)))
for k1,k2 in product(['z','D','V','U','B','soft'],repeat=2):
 A1,A2=fs[k1],fs[k2];
 for i,j in product(range(4),repeat=2):add(f'{k1}{i}*{k2}{j}',A1[:,i]*A2[:,j])
for d in [2,3,4]:add('poly'+str(d),pol(z,d)[:,pol(z,d-1).shape[1]:])
for key in ['z','D','V','U','B','soft']:
 F=fs[key];add('quad'+key,pol(F,2)[:,5:]);add('cube'+key,pol(F,3)[:,15:]);
for k1,k2 in [('z','D'),('z','V'),('z','U')]:add('mixed'+k2,np.einsum('ni,nj->nij',fs[k1],fs[k2]).reshape(len(z),16))
def stats(pred):
 e=Y[n:]-pred[n:]; return np.r_[mae(e),np.mean(abs(e[:,1:]@q.T))]
print('group shifts')
for k in ['soft','U','pol y2'.replace(' ',''),'poly3','poly4','quadD','quadV','quadU','mixedD','mixedV','mixedU']:
 F=np.c_[base,fs[k]]; W=np.linalg.lstsq(F[:n],Y[:n],rcond=1e-7)[0];print(k,rms((F@W-Y)[:n]),stats(F@W))
names=[k for k,F in fs.items() if F.shape[1]==1]; F=np.column_stack([fs[k] for k in names]);
for base0 in [base,np.c_[base,fs['soft']]]:
 for out in range(3):
  A=base0.copy(); yy=Y[:,out]; results=[]; inds=[]
  for i in range(12):
   AF=A[:n]; AR=F[:n]-AF@np.linalg.lstsq(AF,F[:n],rcond=1e-7)[0]; rr=yy[:n]-AF@np.linalg.lstsq(AF,yy[:n],rcond=1e-7)[0]; score=np.abs(AR.T@rr)/np.maximum(np.linalg.norm(AR,axis=0),1e-7); j=score.argmax(); inds.append(j); A=np.c_[A,F[:,j]]; e=yy-A@np.linalg.lstsq(A[:n],yy[:n],rcond=1e-7)[0]; results.append((names[j],round(float(rms(e[:n])),5),round(float(mae(e[n:])),5)))
  print('OMP',base0.shape[1],out,results)
