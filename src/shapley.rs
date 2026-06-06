//! Shapley value computation.
//!
//! The Shapley value is the unique payoff division satisfying four axioms:
//! efficiency, symmetry, dummy player, and additivity. Computed either exactly
//! (all permutations) or by Monte Carlo sampling.

use crate::coalition::Coalition;
use crate::value::ValueFunction;
use serde::{Deserialize, Serialize};

/// Shapley value computation for cooperative games.
#[derive(Clone, Serialize, Deserialize)]
pub struct ShapleyValue {
    /// Computed values, one per player.
    values: Vec<f64>,
    /// Number of players.
    n: u8,
}

impl ShapleyValue {
    /// Compute the exact Shapley value by enumerating all permutations.
    ///
    /// Complexity: O(n! · n). Feasible up to ~10 players.
    pub fn compute_exact(vf: &ValueFunction) -> Self {
        let n = vf.n();
        let mut values = vec![0.0; n as usize];
        let mut perm: Vec<u8> = (0..n).collect();
        let mut count = 0u64;

        loop {
            // Compute marginal contribution for this permutation
            let mut coalition_mask = 0u64;
            for &i in &perm {
                let with_i = coalition_mask | (1u64 << i);
                let marginal = vf.value_by_mask(with_i) - vf.value_by_mask(coalition_mask);
                values[i as usize] += marginal;
                coalition_mask = with_i;
            }
            count += 1;

            if !next_permutation(&mut perm) {
                break;
            }
        }

        for v in values.iter_mut() {
            *v /= count as f64;
        }

        Self { values, n }
    }

    /// Estimate Shapley values via Monte Carlo permutation sampling.
    ///
    /// Use for large games where exact computation is infeasible.
    /// `seed` enables reproducibility.
    pub fn compute_sampled(vf: &ValueFunction, samples: u32, seed: u64) -> Self {
        let n = vf.n();
        let mut values = vec![0.0; n as usize];
        let mut rng = SimpleRng::new(seed);

        for _ in 0..samples {
            let perm = random_permutation(n, &mut rng);
            let mut coalition_mask = 0u64;
            for &i in &perm {
                let with_i = coalition_mask | (1u64 << i);
                let marginal = vf.value_by_mask(with_i) - vf.value_by_mask(coalition_mask);
                values[i as usize] += marginal;
                coalition_mask = with_i;
            }
        }

        for v in values.iter_mut() {
            *v /= samples as f64;
        }

        Self { values, n }
    }

    /// Number of players.
    pub fn n(&self) -> u8 {
        self.n
    }

    /// Get the Shapley value for player `i`.
    pub fn player(&self, i: u8) -> f64 {
        self.values[i as usize]
    }

    /// All Shapley values as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Check the efficiency axiom: sum of values equals v(N).
    pub fn check_efficiency(&self, vf: &ValueFunction) -> bool {
        (self.values.iter().sum::<f64>() - vf.grand_value()).abs() < 1e-6
    }

    /// Check the symmetry axiom: players with identical contributions get equal value.
    pub fn check_symmetry(&self, vf: &ValueFunction) -> bool {
        let n = self.n;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut symmetric = true;
                for c in Coalition::all(n) {
                    if c.contains(i) == c.contains(j) {
                        continue;
                    }
                    let mut with_i = c;
                    with_i.insert(i);
                    with_i.remove(j);
                    let mut with_j = c;
                    with_j.insert(j);
                    with_j.remove(i);
                    if (vf.value(&with_i) - vf.value(&with_j)).abs() > 1e-9 {
                        symmetric = false;
                        break;
                    }
                }
                if symmetric && (self.values[i as usize] - self.values[j as usize]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Check the dummy player axiom: a player whose marginal contribution
    /// is always zero receives Shapley value zero.
    pub fn check_dummy(&self, vf: &ValueFunction) -> bool {
        let n = self.n;
        'outer: for i in 0..n {
            for c in Coalition::all(n) {
                if c.contains(i) {
                    continue;
                }
                let mut with_i = c;
                with_i.insert(i);
                if (vf.value(&with_i) - vf.value(&c)).abs() > 1e-9 {
                    continue 'outer;
                }
            }
            // i is a dummy player
            if self.values[i as usize].abs() > 1e-6 {
                return false;
            }
        }
        true
    }
}

/// Simple LCG-based RNG for reproducible sampling.
#[derive(Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Knuth LCG
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn random_permutation(n: u8, rng: &mut SimpleRng) -> Vec<u8> {
    let mut perm: Vec<u8> = (0..n).collect();
    for i in 0..n as usize {
        let j = i + rng.next_usize(n as usize - i);
        perm.swap(i, j);
    }
    perm
}

pub(super) fn next_permutation(perm: &mut [u8]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    // Find largest i such that perm[i] < perm[i+1]
    let mut i = n - 2;
    while perm[i] >= perm[i + 1] {
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    // Find largest j such that perm[i] < perm[j]
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
    fn symmetric_players_equal_value() {
        // All players symmetric in grand-coalition-only game
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 100.0 } else { 0.0 });
        let sv = ShapleyValue::compute_exact(&vf);
        assert!((sv.player(0) - sv.player(1)).abs() < 1e-9);
        assert!((sv.player(1) - sv.player(2)).abs() < 1e-9);
    }

    #[test]
    fn efficiency_axiom() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 60.0 } else { 0.0 });
        let sv = ShapleyValue::compute_exact(&vf);
        assert!(sv.check_efficiency(&vf));
        assert!((sv.values.iter().sum::<f64>() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn shapley_additive_game() {
        // For additive game v(S) = sum(w_i), Shapley = weights
        let weights = vec![3.0, 5.0, 2.0];
        let vf = ValueFunction::additive(3, &weights);
        let sv = ShapleyValue::compute_exact(&vf);
        for i in 0..3 {
            assert!((sv.player(i as u8) - weights[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn dummy_player() {
        // Player 2 is a dummy: adds nothing
        let vf = ValueFunction::from_fn(3, |c| {
            let mask = c.mask();
            if mask & 0b11 == 0b11 {
                10.0
            } else if mask & 0b01 == 0b01 {
                3.0
            } else if mask & 0b10 == 0b10 {
                5.0
            } else {
                0.0
            }
        });
        let sv = ShapleyValue::compute_exact(&vf);
        assert!(sv.player(2).abs() < 1e-9);
        assert!(sv.check_dummy(&vf));
    }

    #[test]
    fn symmetry_check_passes() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 90.0 } else { 0.0 });
        let sv = ShapleyValue::compute_exact(&vf);
        assert!(sv.check_symmetry(&vf));
    }

    #[test]
    fn sampled_converges_to_exact() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 60.0 } else { 0.0 });
        let exact = ShapleyValue::compute_exact(&vf);
        let sampled = ShapleyValue::compute_sampled(&vf, 10000, 42);
        for i in 0..3 {
            assert!((exact.player(i) - sampled.player(i)).abs() < 1.0);
        }
    }

    #[test]
    fn unanimity_game_shapley() {
        // Unanimity game on {0,1}: both get 0.5, player 2 gets 0
        let target = Coalition::from_indices(3, &[0, 1]);
        let vf = ValueFunction::unanimity(3, &target);
        let sv = ShapleyValue::compute_exact(&vf);
        assert!((sv.player(0) - 0.5).abs() < 1e-9);
        assert!((sv.player(1) - 0.5).abs() < 1e-9);
        assert!(sv.player(2).abs() < 1e-9);
    }

    #[test]
    fn two_player_game() {
        let vf = ValueFunction::from_fn(2, |c| match c.mask() {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            _ => 4.0,
        });
        let sv = ShapleyValue::compute_exact(&vf);
        // P1: (v({1})-v({}))/2 + (v({0,1})-v({0}))/2 = 0.5 + 1.5 = 2.0
        // P0: (v({0})-v({}))/2 + (v({0,1})-v({1}))/2 = 0.5 + 1.0 = 1.5
        assert!((sv.player(0) - 1.5).abs() < 1e-9);
        assert!((sv.player(1) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn weighted_voting_game() {
        // 4-player weighted voting: weights [3,2,1,1], quota 4
        let vf = ValueFunction::from_fn(4, |c| {
            let w: f64 = c
                .players()
                .map(|i| match i {
                    0 => 3.0,
                    1 => 2.0,
                    2 => 1.0,
                    3 => 1.0,
                    _ => 0.0,
                })
                .sum();
            if w >= 4.0 { 1.0 } else { 0.0 }
        });
        let sv = ShapleyValue::compute_exact(&vf);
        assert!(sv.player(0) > sv.player(1));
        assert!(sv.check_efficiency(&vf));
    }
}
