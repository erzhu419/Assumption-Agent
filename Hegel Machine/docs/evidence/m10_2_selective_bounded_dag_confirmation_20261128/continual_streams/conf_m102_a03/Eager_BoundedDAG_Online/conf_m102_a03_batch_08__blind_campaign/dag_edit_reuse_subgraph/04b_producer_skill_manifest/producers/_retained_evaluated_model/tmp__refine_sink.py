from pathlib import Path
p=Path('/tmp/node01_verified_traj.py')
s=p.read_text().replace('labels, q_train, q_test):','labels, q_train, q_test, mode=0):')
needle='    A = np.column_stack(blocks)'
s=s.replace(needle,'''    if mode in (1, 3, 4):
        d = u.shape[1]
        for j, (_, w, f) in enumerate(b):
            g = 40 + 15*j
            add(np.c_[u**2*f, u*f**2, f**3],g)
            add(np.c_[*[u[:,i,None]*w for i in range(d)]],g+1)
            add(np.c_[*[u[:,i,None]**2*w for i in range(d)]],g+2)
            add(np.c_[*[f[:,i,None]**2*w for i in range(d)]],g+3)
            add(np.c_[*[u[:,i,None]*w*w for i in range(d)]],g+4)
            for k,q in enumerate((u,w,f)):
                pair = np.column_stack([q[:,i]+sg*q[:,l] for i,l in itertools.combinations(range(d),2) for sg in (1,-1)])
                add(np.c_[pair/(1+pair**2), pair/(1+4*pair**2), np.sin(3*pair)],g+5+k)
                add(np.c_[pair**2/(1+pair**2), np.cos(3*pair)],g+8+k)
            add(np.c_[*[np.prod(w[:,ids],axis=1) for ids in itertools.combinations(range(d),3)]],g+11)
    A = np.column_stack(blocks)''')
s=s.replace('    def evidence(t):','''    E = Y[:, :1] if mode in (2,3) else Y
    def evidence(t):''').replace('P=cho_solve(L,Y,check_finite=False)\n        Z=cho_solve(L,eye,check_finite=False)-P@P.T/Y.shape[1]','P=cho_solve(L,E,check_finite=False)\n        Z=cho_solve(L,eye,check_finite=False)-P@P.T/E.shape[1]').replace('.5*np.sum(Y*P)/Y.shape[1]','.5*np.sum(E*P)/E.shape[1]')
s=s.replace('    for f in (.2,1.,4.): kernel_candidate(GK,noise*f)','''    for f in ((.2,1.,4.) if not mode else (.07,.25,1.,3.,10.)): kernel_candidate(GK,noise*f)''')
s=s.replace('    for iters in (5,12,24):\n','''    configs = [(it,.008) for it in (5,12,24)] if not mode else [(5,.008),(12,.008),(24,.008),(16,.002),(16,.025)]
    for iters, ridge in configs:
''').replace('G+.008*eye+5.','G+ridge*eye+5.')
s=s.replace('        e=np.mean((C-target[:,None])**2,axis=0)','''        e=np.mean((C-target[:,None])**2,axis=0)
        if mode in (1,3,4):
            e=np.maximum(np.mean(np.abs(C-target[:,None]),axis=0)**2,1e-12)''')
s=s.replace('lambda w: np.mean((C@w-target)**2)+.08*e.min()*np.sum((w-base)**2)','lambda w: (np.mean(np.sqrt((C@w-target)**2+.0001)) if mode in (1,3,4) else np.mean((C@w-target)**2))+.08*e.min()*np.sum((w-base)**2)')
s=s.replace('def run_node(inputs: dict) -> dict:', 'def _predict(inputs, modes_to_use):').replace("    pred=_readout(q,b,c,np.c_[train['y'],(traj-avg)@modes.T],q[:n],q[n:])\n    yh=pred[:,0]; sp=avg+pred[:,1:]@modes\n    return {'y_hat': yh,'state_pred':sp,'diagnostics_vec':np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()])}",'''    amps=(traj-avg)@modes.T
    target=np.asarray(train['y'])
    Y=np.c_[target,amps]
    base=_readout(q,b,c,Y,q[:n],q[n:])
    ys=[base[:,0]]
    for mode in modes_to_use:
        if mode==4:
            # Trajectory amplitudes are auxiliary training observations. Profile
            # out their shared linear component, then predict the scalar residual.
            factor=np.linalg.lstsq(amps,target-target.mean(),rcond=None)[0]
            pred=_readout(q,b,c,np.c_[target-amps@factor,amps],q[:n],q[n:],mode)
            ys.append(pred[:,0]+base[:,1:]@factor)
        else:
            pred=_readout(q,b,c,Y,q[:n],q[n:],mode)
            ys.append(pred[:,0])
    return np.column_stack(ys),avg+base[:,1:]@modes''')
assert 'return np.column_stack(ys)' in s and 'P@P.T/E.shape[1]' in s
Path('/tmp/node01_refinement.py').write_text(s)
