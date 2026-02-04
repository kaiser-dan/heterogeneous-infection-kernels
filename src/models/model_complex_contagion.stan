data {
  int<lower=1> kmax;
  array[kmax] real<lower=0> M;
  array[kmax] real<lower=0> N;
}

parameters {
  real<lower=0, upper=1> rho_l;
  real<lower=0, upper=1> rho_h;
  real<lower=0> w;
  real<lower=0> k0;
}

transformed parameters {
  vector[kmax] c;
  for (nu in 1:kmax) {
    // Sigmoid: rho_l + (rho_h - rho_l) / (1 + exp(-w*(nu - k0)))
    c[nu] = rho_l + (rho_h - rho_l) / (1.0 + exp(-w * (nu - k0)));
  }
}

model {
  // Priors
  rho_l ~ beta(1, 100);
  rho_h ~ beta(1, 100);
  w ~ exponential(0.1);
  k0 ~ normal(kmax/2.0, kmax/2.0);
  
  // Likelihood
  for (nu in 1:kmax) {
    if (M[nu] > 0) target += M[nu] * log(fmax(c[nu], 1e-12));
    if (N[nu] > 0) target += N[nu] * log(fmax(1.0 - c[nu], 1e-12));
  }
}
