exec(open('/tmp/check_batch04.py').read().split('out=errors')[0])
import json
for rec in json.load(open(s._ARCHIVE)):
    a,x,y=(np.asarray(rec[k]) for k in ('lab_id','cmd_stress','true_stress'))
    fit=np.array([s._line(x[a==c],y[a==c]) for c in range(4)])
    print('archive sweep deviation from current [gain,bias]',np.round(fit-cal,6).tolist())
print('archive prior deviation',np.round(s._prior()-cal,6).tolist())
for c in range(4):
    ids=np.flatnonzero(ck['lab_id']==c); x=ck['cmd_stress'][ids]; ends=ids[[x.argmin(),x.argmax()]]
    for n in (1,2,3):
        use=ends[:n] if n<=2 else np.r_[ends,ids[np.argmin(abs(x))]]
        worst=np.zeros(3)
        for i in use:
            altered={k:v.copy() for k,v in ck.items()}; altered['true_stress'][i]+=.20
            sparse=subset(altered,(ck['lab_id']!=c)|np.isin(np.arange(len(ck['lab_id'])),use))
            delta=s._calibrate(sparse)[c]-cal[c]
            worst=np.maximum(worst,np.r_[abs(delta),max(abs(delta[0]*x+delta[1]))])
        print('lab / checks / worst gain, bias, sweep-range stress error with one EMI spike',c,n,worst.round(6))
# Test the two plastic basis columns independently on the measured stress histories.
x=s._align(tr,d['test']); model=s._Design(tr,x[:len(lab)],cal,d['past_dt'],d['eval_times'],d['yield_stress']); tau=np.array([.55,3.8]); a=model.matrix(tau); hist=np.any(abs(model.past)>d['yield_stress'],axis=1); hold=abs(model.hold)>d['yield_stress']; t=d['eval_times']; work=(np.maximum(abs(model.past)-d['yield_stress'],0.)**2*d['past_dt']).sum(axis=1)*model.hold; over=np.maximum(abs(model.hold)-d['yield_stress'],0.)**2*np.sign(model.hold)
np.testing.assert_allclose(a[:,:-1,7],over[:,None]*t); np.testing.assert_allclose(a[:,-1,7],over)
np.testing.assert_allclose(a[:,:-1,8],work[:,None]*(1-np.exp(-t/tau[0]))); np.testing.assert_allclose(a[:,-1,8],work/tau[0])
print('measured training coupon regimes [both sub-yield, history only, hold only, both]',[int((~hist&~hold).sum()),int((hist&~hold).sum()),int((~hist&hold).sum()),int((hist&hold).sum())]); print('plastic basis curve and initial derivative identities passed')
