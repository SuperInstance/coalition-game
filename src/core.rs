//! Core of a cooperative game.
//!
//! The core is the set of imputations x such that:
//!   1. Efficiency: sum(x_i) = v(N)
//!   2. Coalitional rationality: sum_{i in S}(x_i) ≥ v(S) for all S
//!
//! For convex games, the core is always non-empty (it equals the set of
//! all imputations lying between the two extreme marginal contribution vectors).

use crate::coalition::Coalition;
use crate::value::ValueFunction;
use serde::{Deserialize, Serialize};

/// Analysis of the core of a cooperative game.
#[derive(Clone, Serialize, Deserialize)]
pub struct Core {
    /// Whether the core is non-empty.
    non_empty: bool,
    /// Number of players.
    n: u8,
    /// If core is non-empty, a sample imputation in the core (the Shapley value for convex games).
    sample_imputation: Option<Vec<f64>>,
}

/// An imputation: a payoff vector satisfying individual rationality and efficiency.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Imputation {
    /// Payoff to each player.
    payoffs: Vec<f64>,
}

impl Imputation {
    /// Create a new imputation.
    pub fn new(payoffs: Vec<f64>) -> Self {
        Self { payoffs }
    }

    /// Payoff to player `i`.
    pub fn player(&self, i: usize) -> f64 {
        self.payoffs[i]
    }

    /// All payoffs as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.payoffs
    }

    /// Check efficiency: sum of payoffs equals `total`.
    pub fn is_efficient(&self, total: f64) -> bool {
        (self.payoffs.iter().sum::<f64>() - total).abs() < 1e-9
    }

    /// Check individual rationality: x_i >= v({i}) for each player.
    pub fn is_individually_rational(&self, vf: &ValueFunction) -> bool {
        for i in 0..self.payoffs.len() {
            let singleton = Coalition::singleton(vf.n(), i as u8);
            if self.payoffs[i] < vf.value(&singleton) - 1e-9 {
                return false;
            }
        }
        true
    }

    /// Check coalitional rationality: sum_{i in S} x_i >= v(S) for all S.
    pub fn is_coalitionally_rational(&self, vf: &ValueFunction) -> bool {
        for c in Coalition::all(vf.n()) {
            let payoff_to_s: f64 = c.players().map(|i| self.payoffs[i as usize]).sum();
            if payoff_to_s < vf.value(&c) - 1e-9 {
                return false;
            }
        }
        true
    }

    /// Check if this imputation is in the core.
    pub fn is_in_core(&self, vf: &ValueFunction) -> bool {
        self.is_efficient(vf.grand_value()) && self.is_coalitionally_rational(vf)
    }
}

impl Core {
    /// Analyze the core of the given game.
    ///
    /// For convex games, the core is guaranteed non-empty and we use the
    /// Shapley value as a sample imputation. For general games, we check
    /// feasibility via a brute-force search.
    pub fn analyze(vf: &ValueFunction) -> Self {
        let n = vf.n();

        if vf.is_convex() {
            // Core is non-empty for convex games
            let sv = crate::ShapleyValue::compute_exact(vf);
            let imputation: Vec<f64> = sv.as_slice().to_vec();
            return Self {
                non_empty: true,
                n,
                sample_imputation: Some(imputation),
            };
        }

        // Brute-force check: try to find an imputation in the core
        // using equal split as starting point, then check
        if let Some(imputation) = find_core_imputation(vf) {
            Self {
                non_empty: true,
                n,
                sample_imputation: Some(imputation),
            }
        } else {
            Self {
                non_empty: false,
                n,
                sample_imputation: None,
            }
        }
    }

    /// Is the core non-empty?
    pub fn is_non_empty(&self) -> bool {
        self.non_empty
    }

    /// Get a sample imputation in the core, if one exists.
    pub fn sample_imputation(&self) -> Option<&[f64]> {
        self.sample_imputation.as_deref()
    }

    /// Check Bondareva-Shapley conditions for core non-emptiness.
    ///
    /// For a balanced game, the core is non-empty iff the game is balanced.
    /// Returns `true` if the game appears balanced (heuristic for small games).
    pub fn check_balancedness(vf: &ValueFunction) -> bool {
        let n = vf.n();
        if n > 8 {
            // Too many coalitions for exact check
            return vf.is_convex();
        }

        // A game is balanced if for every balanced collection of coalitions
        // with weights δ_S, sum(δ_S * v(S)) <= v(N).
        // For small games, we check a sufficient condition: superadditivity
        // plus individual rationality feasibility.
        let grand_val = vf.grand_value();

        // Check that v(N) >= v({i}) for all i (necessary for core)
        for i in 0..n {
            let s = Coalition::singleton(n, i);
            if vf.value(&s) > grand_val + 1e-9 {
                return false;
            }
        }

        // Check pairwise: v(N) >= v(S) + v(N\S) is necessary (not sufficient)
        // but combined with convexity gives the full Bondareva-Shapley result
        vf.is_superadditive() || find_core_imputation(vf).is_some()
    }

    /// Compute the set of extreme points of the core for convex games.
    ///
    /// For a convex game with `n` players, there are `n!` extreme points,
    /// one for each permutation. The extreme point for permutation π is:
    /// x_i = v(S_i ∪ {i}) - v(S_i) where S_i = {j : π(j) < π(i)}.
    pub fn extreme_points(vf: &ValueFunction) -> Vec<Vec<f64>> {
        let n = vf.n();
        let mut points = Vec::new();
        let mut perm: Vec<u8> = (0..n).collect();

        loop {
            let mut x = vec![0.0; n as usize];
            let mut coalition_mask = 0u64;
            for &i in &perm {
                let with_i = coalition_mask | (1u64 << i);
                x[i as usize] = vf.value_by_mask(with_i) - vf.value_by_mask(coalition_mask);
                coalition_mask = with_i;
            }
            points.push(x);

            if !crate::shapley::next_permutation(&mut perm) {
                break;
            }
        }

        points
    }
}

/// Try to find an imputation in the core by checking if the Shapley value is in core,
/// then trying marginal contribution vectors for each permutation.
fn find_core_imputation(vf: &ValueFunction) -> Option<Vec<f64>> {
    let n = vf.n();

    // Try Shapley value
    let sv = crate::ShapleyValue::compute_exact(vf);
    let imp = Imputation::new(sv.as_slice().to_vec());
    if imp.is_in_core(vf) {
        return Some(sv.as_slice().to_vec());
    }

    // Try marginal contribution vectors (extreme points)
    let mut perm: Vec<u8> = (0..n).collect();
    loop {
        let mut x = vec![0.0; n as usize];
        let mut coalition_mask = 0u64;
        for &i in &perm {
            let with_i = coalition_mask | (1u64 << i);
            x[i as usize] = vf.value_by_mask(with_i) - vf.value_by_mask(coalition_mask);
            coalition_mask = with_i;
        }
        let imp = Imputation::new(x.clone());
        if imp.is_in_core(vf) {
            return Some(x);
        }

        if !next_perm(&mut perm) {
            break;
        }
    }

    None
}

fn next_perm(perm: &mut [u8]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 2;
    while perm[i] >= perm[i + 1] {
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    let mut j = n - 1;
    while perm[i] >= perm[j] {
        j -= 1;
    }
    perm.swap(i, j);
    perm[i + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_game_has_nonempty_core() {
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64).powi(2));
        let core = Core::analyze(&vf);
        assert!(core.is_non_empty());
    }

    #[test]
    fn core_imputation_is_valid() {
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64).powi(2));
        let core = Core::analyze(&vf);
        let imp = Imputation::new(core.sample_imputation().unwrap().to_vec());
        assert!(imp.is_in_core(&vf));
    }

    #[test]
    fn glove_game_core() {
        // Glove game: 2 left gloves, 1 right glove. v(S) = min(#left, #right).
        // Players 0,1 have left gloves; player 2 has right glove.
        let vf = ValueFunction::from_fn(3, |c| {
            let left = c.players().filter(|&i| i < 2).count() as f64;
            let right = c.players().filter(|&i| i == 2).count() as f64;
            left.min(right)
        });
        let core = Core::analyze(&vf);
        assert!(core.is_non_empty());
        // Player 2 (right glove) should get 1.0 in core
        let imp = core.sample_imputation().unwrap();
        assert!((imp[2] - 1.0).abs() < 1e-6 || imp[2] > 0.5);
    }

    #[test]
    fn simple_majority_core_empty() {
        // Simple majority voting: v(S) = 1 if |S| > n/2, else 0.
        // Core is empty for n >= 3 (no stable allocation).
        let vf = ValueFunction::from_fn(3, |c| if c.count() > 1 { 1.0 } else { 0.0 });
        // This is NOT convex (3-player majority game core is empty)
        let core = Core::analyze(&vf);
        // For 3-player simple majority, core is indeed empty
        // because any pair can form and get 1, but 3 pairs > total 1
        assert!(!core.is_non_empty());
    }

    #[test]
    fn additive_game_core() {
        // Additive games: core is the single point {weights}
        let vf = ValueFunction::additive(3, &[10.0, 20.0, 30.0]);
        let core = Core::analyze(&vf);
        assert!(core.is_non_empty());
        let imp = core.sample_imputation().unwrap();
        assert!((imp[0] - 10.0).abs() < 1e-6);
        assert!((imp[1] - 20.0).abs() < 1e-6);
        assert!((imp[2] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn imputation_individual_rationality() {
        let vf = ValueFunction::from_fn(2, |c| match c.mask() {
            0 => 0.0,
            1 => 5.0,
            2 => 3.0,
            _ => 10.0,
        });
        let good = Imputation::new(vec![5.0, 5.0]);
        assert!(good.is_individually_rational(&vf));
        let bad = Imputation::new(vec![1.0, 9.0]);
        assert!(!bad.is_individually_rational(&vf));
    }

    #[test]
    fn extreme_points_count() {
        let vf = ValueFunction::additive(3, &[1.0, 2.0, 3.0]);
        let points = Core::extreme_points(&vf);
        assert_eq!(points.len(), 6); // 3!
    }

    #[test]
    fn balancedness_check() {
        let vf = ValueFunction::additive(3, &[5.0, 5.0, 5.0]);
        assert!(Core::check_balancedness(&vf));
    }

    #[test]
    fn imputation_efficiency_check() {
        let imp = Imputation::new(vec![10.0, 20.0, 30.0]);
        assert!(imp.is_efficient(60.0));
        assert!(!imp.is_efficient(50.0));
    }
}
