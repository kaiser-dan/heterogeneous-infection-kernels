import pytest
import numpy as np
import networkx as nx
from pathlib import Path
from cmdstanpy import CmdStanModel
from src.dynamics import heterogeneous_contagion_process, get_activations

@pytest.fixture
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture
def simple_model_path(project_root):
    return project_root / 'src' / 'models' / 'model_simple_contagion.stan'

@pytest.fixture
def complex_model_path(project_root):
    return project_root / 'src' / 'models' / 'model_complex_contagion.stan'

def test_simple_contagion_recovery(simple_model_path):
    # 1. Setup graph
    V = 100
    G = nx.barabasi_albert_graph(V, 4, seed=42)
    A = nx.to_numpy_array(G)
    kmax = int(np.max(A.sum(axis=1)))
    
    # 2. Ground truth kernel (Simple Contagion)
    rho_true = 0.2
    c_true = np.zeros((V, kmax))
    for nu in range(1, kmax + 1):
        c_true[:, nu-1] = 1.0 - (1.0 - rho_true)**nu
    
    # 3. Simulate process
    gamma = 0.0
    x0 = np.zeros(V)
    np.random.seed(42)
    x0[np.random.choice(V, 10, replace=False)] = 1
    
    states = heterogeneous_contagion_process(A, gamma, c_true, x0, tmax=50, dt=1, random_seed=42)
    M, N = get_activations(states.T, A, kmax)
    
    # 4. Fit Stan model
    model = CmdStanModel(stan_file=str(simple_model_path))
    
    stan_data = {
        'V': V, 'kmax': kmax, 'K': 1, 'M': M.tolist(), 'N': N.tolist(),
        'DEVIATION_SCALE': 0.1, 'LOGIT_P_LOCATION': 0.0, 'LOGIT_P_SCALE': 1.0
    }
    
    fit = model.sample(data=stan_data, iter_sampling=500, iter_warmup=500, chains=1, show_progress=False)
    
    # 5. Verify recovery
    mu_samples = fit.stan_variable('mu')
    mu_inferred = np.mean(mu_samples, axis=0)[0]
    gt_kernel = 1.0 - (1.0 - rho_true)**np.arange(1, kmax + 1)
    
    np.testing.assert_allclose(mu_inferred, gt_kernel, atol=0.15)

def test_complex_contagion_recovery(complex_model_path):
    # 1. Setup graph
    V = 200
    G = nx.barabasi_albert_graph(V, 5, seed=42)
    A = nx.to_numpy_array(G)
    kmax = int(np.max(A.sum(axis=1)))
    
    # 2. Ground truth kernel (Complex Contagion - Sigmoid)
    rho_l, rho_h, w, k0 = 0.05, 0.4, 1.5, 6.0
    
    c_true = np.zeros((V, kmax))
    for nu in range(1, kmax + 1):
        c_true[:, nu-1] = rho_l + (rho_h - rho_l) / (1.0 + np.exp(-w * (nu - k0)))
        
    # 3. Simulate process
    gamma = 0.0
    x0 = np.zeros(V)
    np.random.seed(42)
    x0[np.random.choice(V, 20, replace=False)] = 1
    
    states = heterogeneous_contagion_process(A, gamma, c_true, x0, tmax=100, dt=1, random_seed=42)
    M, N = get_activations(states.T, A, kmax)
    
    # 4. Fit Stan model
    model = CmdStanModel(stan_file=str(complex_model_path))
    
    stan_data = {
        'V': V, 'kmax': kmax, 'K': 1, 'M': M.tolist(), 'N': N.tolist(),
        'DEVIATION_SCALE': 0.1, 'LOGIT_P_LOCATION': 0.0, 'LOGIT_P_SCALE': 1.0
    }
    
    fit = model.sample(data=stan_data, iter_sampling=500, iter_warmup=500, chains=1, show_progress=False)
    
    # 5. Verify recovery
    mu_samples = fit.stan_variable('mu')
    mu_inferred = np.mean(mu_samples, axis=0)[0]
    
    gt_kernel = rho_l + (rho_h - rho_l) / (1.0 + np.exp(-w * (np.arange(1, kmax + 1) - k0)))
    
    # Inferred kernel should catch the sigmoid transition
    # Probability at k=1 should be low, at k=kmax should be high
    assert mu_inferred[0] < 0.2
    assert mu_inferred[-1] > 0.3
    
    # Verify monotonicity
    assert np.all(np.diff(mu_inferred) >= -1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
