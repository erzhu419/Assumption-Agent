import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_w03_batch_08_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for case in pickle.load(open(root+'public_beta_w03_batch_08.pkl','rb')):
    s, gt = case['study'], case['gt']
    ix, iy = s['sensor_ij'].T
    bank = s['row_bank_id']
    def design(u):
        return u[:,:,ix,iy].transpose(0,2,1).reshape(-1,u.shape[1])
    def profile(v):
        return m._profile(s['Z_nuisance'],v,s['nuisance_ridge'])
    def rel(a,b):
        return np.linalg.norm(a-b)/np.linalg.norm(b)
    u = m.responses(s)
    S = design(u)
    V = s['V_reaction_2d']
    additive = V.mean(0) + V.mean(1)[:,None] - V.mean()
    alternatives = {'coupled_1.45':s, 'additive_reaction':{**s,'V_reaction_2d':additive}, 'unscaled_diffusion':{**s,**{f'kappa_{a}':s[f'kappa_{a}']/1.45 for a in 'xy'}}}
    for name,t in alternatives.items():
        U = m.responses(t)
        print(name, 'profiled design rel L2', rel(profile(design(U)),gt['S_tilde']), 'reference-weight terminal rel L2', rel(np.einsum('m,mij->ij',gt['w_opt'],U[-1]),gt['terminal_field_A']))
    y = m._calibrate(s,S)
    archive = m._ARCHIVE
    try:
        m._ARCHIVE = [[],[],[],[]]
        free = m._calibrate(s,S)
    finally:
        m._ARCHIVE = archive
    for name,obs in [('archive',y),('no_archive',free)]:
        p = profile(obs[:,None])[:,0]
        print(name, 'profiled observation rel L2', rel(p,gt['r_y']), 'within-bank RMSE', [float(np.sqrt(np.mean((p[bank==b]-gt['r_y'][bank==b])**2))) for b in range(4)])
    out = m.solve_beta_w03_batch_08(s)
    print('exact artifact final output errors', rel(out['weights'],gt['w_opt']), rel(out['terminal_field'],gt['terminal_field_A']))
