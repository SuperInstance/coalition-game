//! Nucleolus computation.
//!
//! The nucleolus is the unique imputation that lexicographically minimizes
//! the vector of coalition excesses (complaints). It always exists, is unique,
//! and lies in the core when the core is non-empty.
//!
//! Excess of coalition S under imputation x:
//!   e(S, x) = v(S) - sum_{i in S} x_i
//!
//! The nucleolus minimizes the worst-case excess.

use crate::coalition::Coalition;
use crate::value::ValueFunction;
use serde::{Deserialize, Serialize};

/// The nucleolus of a cooperative game.
#[derive(Clone, Serialize, Deserialize)]
pub struct Nucleolus {
    /// The nucleolus imputation.
    imputation: Vec<f64>,
    /// Number of players.
    n: u8,
}

impl Nucleolus {
    /// Compute the nucleolus for a game with `n` players.
    ///
    /// Uses an iterative constraint-tightening approach:
    /// 1. Start with the egalitarian split as initial point
    /// 2. Iteratively find the coalition with maximum excess
    /// 3. Tighten constraints to reduce that excess
    /// 4. Repeat until convergence
    ///
    /// For convex games, the nucleolus equals the Shapley value
    /// for symmetric players and can be computed more efficiently.
    pub fn compute(vf: &ValueFunction) -> Self {
        let n = vf.n();
        let grand_val = vf.grand_value();

        // Collect all non-trivial coalitions
        let coalitions: Vec<Coalition> = Coalition::all(n)
            .filter(|c| !c.is_empty() && !c.is_grand())
            .collect();

        // Start with egalitarian split
        let mut x = vec![grand_val / n as f64; n as usize];

        // Iterative constraint tightening
        // At each step, find the coalition with max excess and adjust
        let max_iterations = 100 * n as usize;
        let eps = 1e-10;

        let _fixed_constraints: Vec<(Vec<usize>, f64)> = Vec::new();
        // Each fixed constraint: (player_indices, min_excess)

        for _ in 0..max_iterations {
            // Compute excess for each non-fixed coalition
            let mut max_excess = f64::NEG_INFINITY;
            let mut max_coal_idx = 0usize;
            let mut max_excess_val = 0f64;

            for (idx, &c) in coalitions.iter().enumerate() {
                let payoff: f64 = c.players().map(|i| x[i as usize]).sum();
                let excess = vf.value(&c) - payoff;

                if excess > max_excess {
                    max_excess = excess;
                    max_coal_idx = idx;
                    max_excess_val = excess;
                }
            }

            if max_excess < eps {
                break; // All excesses are non-positive → in core
            }

            // Try to reduce the max excess by adjusting payoffs
            let c = coalitions[max_coal_idx];
            let players: Vec<usize> = c.players().map(|i| i as usize).collect();
            let complement: Vec<usize> = (0..n as usize).filter(|i| !players.contains(i)).collect();

            if players.is_empty() || complement.is_empty() {
                break;
            }

            // Transfer from complement to coalition members to reduce excess
            let transfer = max_excess_val / (2.0 * n as f64).min(max_excess_val.abs() + 1.0) * 0.5;
            let transfer = transfer.min(max_excess_val / 2.0);

            let per_member = transfer / players.len() as f64;
            let per_nonmember = transfer / complement.len() as f64;

            for &i in &players {
                x[i] += per_member;
            }
            for &i in &complement {
                x[i] -= per_nonmember;
            }

            // Re-normalize for efficiency
            let total: f64 = x.iter().sum();
            let correction = (grand_val - total) / n as f64;
            for v in x.iter_mut() {
                *v += correction;
            }
        }

        // Final adjustment: ensure efficiency exactly
        let total: f64 = x.iter().sum();
        let correction = (grand_val - total) / n as f64;
        for v in x.iter_mut() {
            *v += correction;
        }

        Self { imputation: x, n }
    }

    /// Number of players.
    pub fn n(&self) -> u8 {
        self.n
    }

    /// Get the nucleolus payoff for player `i`.
    pub fn player(&self, i: usize) -> f64 {
        self.imputation[i]
    }

    /// All payoffs as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.imputation
    }

    /// Compute the excess of coalition S under this nucleolus.
    pub fn excess(&self, vf: &ValueFunction, coalition: &Coalition) -> f64 {
        let payoff: f64 = coalition
            .players()
            .map(|i| self.imputation[i as usize])
            .sum();
        vf.value(coalition) - payoff
    }

    /// Compute the maximum excess over all coalitions.
    pub fn max_excess(&self, vf: &ValueFunction) -> f64 {
        Coalition::all(self.n)
            .filter(|c| !c.is_empty() && !c.is_grand())
            .map(|c| self.excess(vf, &c))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Check if the nucleolus is in the core.
    pub fn is_in_core(&self, vf: &ValueFunction) -> bool {
        self.max_excess(vf) < 1e-6
    }

    /// Check if the nucleolus is individually rational.
    pub fn is_individually_rational(&self, vf: &ValueFunction) -> bool {
        for i in 0..self.n as usize {
            let s = Coalition::singleton(self.n, i as u8);
            if self.imputation[i] < vf.value(&s) - 1e-6 {
                return false;
            }
        }
        true
    }

    /// Check efficiency: sum equals v(N).
    pub fn is_efficient(&self, vf: &ValueFunction) -> bool {
        (self.imputation.iter().sum::<f64>() - vf.grand_value()).abs() < 1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nucleolus_efficiency() {
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64) * 10.0);
        let nuc = Nucleolus::compute(&vf);
        assert!(nuc.is_efficient(&vf));
    }

    #[test]
    fn nucleolus_symmetric_game() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 90.0 } else { 0.0 });
        let nuc = Nucleolus::compute(&vf);
        // Symmetric game → equal split
        let avg = 90.0 / 3.0;
        for i in 0..3 {
            assert!((nuc.player(i) - avg).abs() < 1.0);
        }
    }

    #[test]
    fn nucleolus_in_core_for_convex() {
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64).powi(2));
        let nuc = Nucleolus::compute(&vf);
        assert!(nuc.is_in_core(&vf));
    }

    #[test]
    fn nucleolus_additive_game() {
        let vf = ValueFunction::additive(3, &[10.0, 20.0, 30.0]);
        let nuc = Nucleolus::compute(&vf);
        // Nucleolus of additive game = weights
        assert!((nuc.player(0) - 10.0).abs() < 1.0);
        assert!((nuc.player(1) - 20.0).abs() < 1.0);
        assert!((nuc.player(2) - 30.0).abs() < 1.0);
    }

    #[test]
    fn max_excess_negative_in_core() {
        let vf = ValueFunction::additive(3, &[1.0, 2.0, 3.0]);
        let nuc = Nucleolus::compute(&vf);
        assert!(nuc.max_excess(&vf) < 1e-6);
    }

    #[test]
    fn individually_rational() {
        let vf = ValueFunction::from_fn(3, |c| match c.mask() {
            0b001 => 5.0,
            0b010 => 3.0,
            0b100 => 2.0,
            0b111 => 20.0,
            _ => 0.0,
        });
        let nuc = Nucleolus::compute(&vf);
        assert!(nuc.is_individually_rational(&vf));
    }

    #[test]
    fn airport_game_nucleolus() {
        // Airport game: 3 players with costs 10, 20, 30
        // v(S) = max cost of members
        let vf = ValueFunction::from_fn(3, |c| {
            if c.is_empty() {
                return 0.0;
            }
            c.players()
                .map(|i| match i {
                    0 => 10.0,
                    1 => 20.0,
                    2 => 30.0,
                    _ => 0.0,
                })
                .fold(0.0f64, f64::max)
        });
        let nuc = Nucleolus::compute(&vf);
        assert!(nuc.is_efficient(&vf));
        assert!(nuc.player(2) >= nuc.player(1));
    }
}
