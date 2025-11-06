/**
 * GLM Family Definitions
 *
 * Each family defines:
 * - link functions (canonical and alternatives)
 * - variance function (relationship between mean and variance)
 * - deviance function (goodness of fit)
 * - initialization and valid ranges
 */

// ============================================================================
// Link Functions
// ============================================================================

/**
 * Identity link: η = μ
 */
export class IdentityLink {
  name = 'identity';

  linkfun(mu) {
    return mu;
  }

  linkinv(eta) {
    return eta;
  }

  mu_eta(eta) {
    return Array(eta.length).fill(1);
  }

  valideta(eta) {
    return true; // always valid
  }

  validmu(mu) {
    return mu.every(m => isFinite(m));
  }
}

/**
 * Log link: η = log(μ)
 */
export class LogLink {
  name = 'log';

  linkfun(mu) {
    return mu.map(m => Math.log(Math.max(m, 1e-10)));
  }

  linkinv(eta) {
    return eta.map(e => Math.exp(Math.min(e, 700))); // prevent overflow
  }

  mu_eta(eta) {
    return eta.map(e => Math.exp(Math.min(e, 700)));
  }

  valideta(eta) {
    return eta.every(e => isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m > 0 && isFinite(m));
  }
}

/**
 * Logit link: η = log(μ/(1-μ))
 */
export class LogitLink {
  name = 'logit';

  linkfun(mu) {
    return mu.map(m => {
      const p = Math.max(1e-10, Math.min(1 - 1e-10, m));
      return Math.log(p / (1 - p));
    });
  }

  linkinv(eta) {
    return eta.map(e => {
      const exp_e = Math.exp(Math.min(e, 700));
      return exp_e / (1 + exp_e);
    });
  }

  mu_eta(eta) {
    return eta.map(e => {
      const exp_e = Math.exp(Math.min(e, 700));
      return exp_e / Math.pow(1 + exp_e, 2);
    });
  }

  valideta(eta) {
    return eta.every(e => isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m > 0 && m < 1 && isFinite(m));
  }
}

/**
 * Probit link: η = Φ^{-1}(μ)
 */
export class ProbitLink {
  name = 'probit';

  qnorm(p) {
    // Approximation of inverse normal CDF (Beasley-Springer-Moro algorithm)
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;

    const a0 = 2.50662823884;
    const a1 = -18.61500062529;
    const a2 = 41.39119773534;
    const a3 = -25.44106049637;
    const b1 = -8.47351093090;
    const b2 = 23.08336743743;
    const b3 = -21.06224101826;
    const b4 = 3.13082909833;
    const c0 = -2.78718931138;
    const c1 = -2.29796479134;
    const c2 = 4.85014127135;
    const c3 = 2.32121276858;
    const d1 = 3.54388924762;
    const d2 = 1.63706781897;

    const q = p - 0.5;

    if (Math.abs(q) <= 0.42) {
      const r = q * q;
      return q * (a0 + r * (a1 + r * (a2 + r * a3))) /
             (1 + r * (b1 + r * (b2 + r * (b3 + r * b4))));
    } else {
      let r = p;
      if (q > 0) r = 1 - p;
      r = Math.sqrt(-Math.log(r));
      const val = (c0 + r * (c1 + r * (c2 + r * c3))) /
                  (1 + r * (d1 + r * d2));
      return q < 0 ? -val : val;
    }
  }

  dnorm(x) {
    // Standard normal PDF
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  }

  pnorm(x) {
    // Standard normal CDF (approximation)
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }

  linkfun(mu) {
    return mu.map(m => {
      const p = Math.max(1e-10, Math.min(1 - 1e-10, m));
      return this.qnorm(p);
    });
  }

  linkinv(eta) {
    return eta.map(e => this.pnorm(e));
  }

  mu_eta(eta) {
    return eta.map(e => this.dnorm(e));
  }

  valideta(eta) {
    return eta.every(e => isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m > 0 && m < 1 && isFinite(m));
  }
}

/**
 * Inverse link: η = 1/μ
 */
export class InverseLink {
  name = 'inverse';

  linkfun(mu) {
    return mu.map(m => 1 / Math.max(m, 1e-10));
  }

  linkinv(eta) {
    return eta.map(e => 1 / Math.max(Math.abs(e), 1e-10));
  }

  mu_eta(eta) {
    return eta.map(e => -1 / Math.max(e * e, 1e-10));
  }

  valideta(eta) {
    return eta.every(e => e !== 0 && isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m > 0 && isFinite(m));
  }
}

/**
 * Inverse squared link: η = 1/μ²
 */
export class InverseSquaredLink {
  name = 'inverse_squared';

  linkfun(mu) {
    return mu.map(m => 1 / Math.max(m * m, 1e-10));
  }

  linkinv(eta) {
    return eta.map(e => 1 / Math.sqrt(Math.max(e, 1e-10)));
  }

  mu_eta(eta) {
    return eta.map(e => -0.5 / Math.pow(Math.max(e, 1e-10), 1.5));
  }

  valideta(eta) {
    return eta.every(e => e > 0 && isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m > 0 && isFinite(m));
  }
}

/**
 * Square root link: η = √μ
 */
export class SqrtLink {
  name = 'sqrt';

  linkfun(mu) {
    return mu.map(m => Math.sqrt(Math.max(m, 0)));
  }

  linkinv(eta) {
    return eta.map(e => e * e);
  }

  mu_eta(eta) {
    return eta.map(e => 2 * e);
  }

  valideta(eta) {
    return eta.every(e => e >= 0 && isFinite(e));
  }

  validmu(mu) {
    return mu.every(m => m >= 0 && isFinite(m));
  }
}

// ============================================================================
// Family Classes
// ============================================================================

/**
 * Gaussian family (normal distribution)
 * Canonical link: identity
 * Variance: constant (φ)
 */
export class Gaussian {
  constructor(link = 'identity') {
    this.family = 'gaussian';
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      identity: new IdentityLink(),
      log: new LogLink(),
      inverse: new InverseLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for Gaussian family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(() => 1); // constant variance
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const d = y[i] - mu[i];
      dev += w[i] * d * d;
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    const mu = [...y]; // start with observed values
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

/**
 * Binomial family (binary or proportion)
 * Canonical link: logit
 * Variance: μ(1-μ)
 */
export class Binomial {
  constructor(link = 'logit') {
    this.family = 'binomial';
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      logit: new LogitLink(),
      probit: new ProbitLink(),
      log: new LogLink(),
      identity: new IdentityLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for Binomial family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(m => {
      const p = Math.max(1e-10, Math.min(1 - 1e-10, m));
      return p * (1 - p);
    });
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = Math.max(1e-10, Math.min(1 - 1e-10, y[i]));
      const mui = Math.max(1e-10, Math.min(1 - 1e-10, mu[i]));
      dev += 2 * w[i] * (yi * Math.log(yi / mui) + (1 - yi) * Math.log((1 - yi) / (1 - mui)));
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    // Initialize mu as (y + 0.5) / (n + 1) to avoid 0 and 1
    const mu = y.map(yi => (yi + 0.5) / 2);
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

/**
 * Poisson family (count data)
 * Canonical link: log
 * Variance: μ
 */
export class Poisson {
  constructor(link = 'log') {
    this.family = 'poisson';
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      log: new LogLink(),
      identity: new IdentityLink(),
      sqrt: new SqrtLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for Poisson family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(m => Math.max(m, 1e-10));
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = y[i];
      const mui = Math.max(mu[i], 1e-10);
      if (yi > 0) {
        dev += 2 * w[i] * (yi * Math.log(yi / mui) - (yi - mui));
      } else {
        dev += 2 * w[i] * mui;
      }
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    // Initialize mu as y + 0.1 to avoid log(0)
    const mu = y.map(yi => Math.max(yi + 0.1, 0.1));
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

/**
 * Gamma family (positive continuous data)
 * Canonical link: inverse
 * Variance: μ²
 */
export class Gamma {
  constructor(link = 'inverse') {
    this.family = 'gamma';
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      inverse: new InverseLink(),
      log: new LogLink(),
      identity: new IdentityLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for Gamma family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(m => {
      const mui = Math.max(m, 1e-10);
      return mui * mui;
    });
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = Math.max(y[i], 1e-10);
      const mui = Math.max(mu[i], 1e-10);
      dev += 2 * w[i] * ((yi - mui) / mui - Math.log(yi / mui));
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    // Initialize mu as max(y, 0.1) to ensure positive values
    const mu = y.map(yi => Math.max(yi, 0.1));
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

/**
 * Inverse Gaussian family (skewed positive data)
 * Canonical link: inverse squared
 * Variance: μ³
 */
export class InverseGaussian {
  constructor(link = 'inverse_squared') {
    this.family = 'inverse_gaussian';
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      inverse_squared: new InverseSquaredLink(),
      inverse: new InverseLink(),
      log: new LogLink(),
      identity: new IdentityLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for InverseGaussian family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(m => {
      const mui = Math.max(m, 1e-10);
      return mui * mui * mui;
    });
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = Math.max(y[i], 1e-10);
      const mui = Math.max(mu[i], 1e-10);
      const d = (yi - mui) / (yi * mui);
      dev += w[i] * d * d / yi;
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    const mu = y.map(yi => Math.max(yi, 0.1));
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

/**
 * Negative Binomial family (overdispersed count data)
 * Canonical link: log
 * Variance: μ + μ²/θ (where θ is dispersion parameter)
 */
export class NegativeBinomial {
  constructor(link = 'log', theta = 1) {
    this.family = 'negative_binomial';
    this.theta = theta; // dispersion parameter
    this.link = this._getLink(link);
  }

  _getLink(linkName) {
    const links = {
      log: new LogLink(),
      identity: new IdentityLink(),
      sqrt: new SqrtLink()
    };

    if (!links[linkName]) {
      throw new Error(`Invalid link '${linkName}' for NegativeBinomial family. Valid links: ${Object.keys(links).join(', ')}`);
    }

    return links[linkName];
  }

  variance(mu) {
    return mu.map(m => {
      const mui = Math.max(m, 1e-10);
      return mui + (mui * mui) / this.theta;
    });
  }

  deviance(y, mu, weights = null) {
    const w = weights || Array(y.length).fill(1);
    let dev = 0;
    for (let i = 0; i < y.length; i++) {
      const yi = y[i];
      const mui = Math.max(mu[i], 1e-10);
      const theta = this.theta;

      if (yi > 0) {
        dev += 2 * w[i] * (yi * Math.log(yi / mui) - (yi + theta) * Math.log((yi + theta) / (mui + theta)));
      } else {
        dev += 2 * w[i] * theta * Math.log(theta / (mui + theta));
      }
    }
    return dev;
  }

  initialize(y, weights = null) {
    const w = weights || Array(y.length).fill(1);
    const mu = y.map(yi => Math.max(yi + 0.1, 0.1));
    const eta = this.link.linkfun(mu);
    return { mu, eta };
  }

  validmu(mu) {
    return this.link.validmu(mu);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a family object from string or config
 */
export function createFamily(config) {
  if (typeof config === 'string') {
    config = { family: config };
  }

  const { family, link, theta } = config;

  switch (family.toLowerCase()) {
    case 'gaussian':
    case 'normal':
      return new Gaussian(link);
    case 'binomial':
      return new Binomial(link);
    case 'poisson':
      return new Poisson(link);
    case 'gamma':
      return new Gamma(link);
    case 'inverse_gaussian':
    case 'inverse.gaussian':
      return new InverseGaussian(link);
    case 'negative_binomial':
    case 'negativebinomial':
    case 'nb':
      return new NegativeBinomial(link, theta);
    default:
      throw new Error(`Unknown family: ${family}. Valid families: gaussian, binomial, poisson, gamma, inverse_gaussian, negative_binomial`);
  }
}

/**
 * Get canonical link for a family
 */
export function getCanonicalLink(family) {
  const canonical = {
    gaussian: 'identity',
    binomial: 'logit',
    poisson: 'log',
    gamma: 'inverse',
    inverse_gaussian: 'inverse_squared',
    negative_binomial: 'log'
  };

  return canonical[family.toLowerCase()] || 'identity';
}
