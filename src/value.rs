//! Characteristic function v(S) mapping coalitions to real values.
//!
//! Includes classification into superadditive, subadditive, and convex games,
//! plus marginal contribution computation.

use crate::coalition::Coalition;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A characteristic function v(S) for a cooperative game with `n` players.
///
/// Values are stored in a hash map keyed by coalition bitmask for O(1) lookup.
#[derive(Clone, Serialize, Deserialize)]
pub struct ValueFunction {
    /// Number of players.
    n: u8,
    /// Map from coalition bitmask → value.
    values: HashMap<u64, f64>,
}

impl ValueFunction {
    /// Create a new value function by evaluating a closure over all coalitions.
    ///
    /// The closure receives each coalition and returns its value.
    pub fn from_fn<F>(n: u8, f: F) -> Self
    where
        F: Fn(Coalition) -> f64,
    {
        let mut values = HashMap::new();
        for c in Coalition::all(n) {
            values.insert(c.mask(), f(c));
        }
        Self { n, values }
    }

    /// Create a value function from an explicit map of bitmask → value.
    ///
    /// The empty coalition is set to 0.0 if not present.
    pub fn from_map(n: u8, values: HashMap<u64, f64>) -> Self {
        let mut values = values;
        values.entry(0).or_insert(0.0);
        Self { n, values }
    }

    /// Number of players.
    pub fn n(&self) -> u8 {
        self.n
    }

    /// Evaluate v(S) for a coalition.
    pub fn value(&self, coalition: &Coalition) -> f64 {
        debug_assert_eq!(coalition.num_players(), self.n);
        *self.values.get(&coalition.mask()).unwrap_or(&0.0)
    }

    /// Evaluate v(S) by bitmask directly.
    pub fn value_by_mask(&self, mask: u64) -> f64 {
        *self.values.get(&mask).unwrap_or(&0.0)
    }

    /// Marginal contribution of player `i` to coalition `S`.
    ///
    /// Returns v(S ∪ {i}) − v(S \ {i}).
    pub fn marginal_contribution(&self, i: u8, coalition: &Coalition) -> f64 {
        debug_assert!(i < self.n);
        let with_i = {
            let mut c = *coalition;
            c.insert(i);
            c
        };
        let without_i = {
            let mut c = *coalition;
            c.remove(i);
            c
        };
        self.value(&with_i) - self.value(&without_i)
    }

    /// The value of the grand coalition v(N).
    pub fn grand_value(&self) -> f64 {
        self.value(&Coalition::grand(self.n))
    }

    /// Check if the game is superadditive: for all disjoint S, T,
    /// v(S ∪ T) ≥ v(S) + v(T).
    pub fn is_superadditive(&self) -> bool {
        for s in Coalition::all(self.n) {
            for t in Coalition::all(self.n) {
                if (s.mask() & t.mask()) == 0 {
                    // disjoint
                    let union = s | t;
                    if self.value(&union) < self.value(&s) + self.value(&t) - 1e-9 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if the game is subadditive: for all disjoint S, T,
    /// v(S ∪ T) ≤ v(S) + v(T).
    pub fn is_subadditive(&self) -> bool {
        for s in Coalition::all(self.n) {
            for t in Coalition::all(self.n) {
                if (s.mask() & t.mask()) == 0 {
                    let union = s | t;
                    if self.value(&union) > self.value(&s) + self.value(&t) + 1e-9 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if the game is convex: for all S, T,
    /// v(S ∪ T) + v(S ∩ T) ≥ v(S) + v(T).
    ///
    /// Equivalent to: marginal contributions are non-decreasing
    /// (player contributes more to larger coalitions).
    pub fn is_convex(&self) -> bool {
        for s in Coalition::all(self.n) {
            for t in Coalition::all(self.n) {
                let union = s | t;
                let inter = s & t;
                if self.value(&union) + self.value(&inter) < self.value(&s) + self.value(&t) - 1e-9
                {
                    return false;
                }
            }
        }
        true
    }

    /// Get all coalition masks and their values.
    pub fn entries(&self) -> impl Iterator<Item = (u64, f64)> + use<'_> {
        self.values.iter().map(|(&k, &v)| (k, v))
    }

    /// A simple additive game: v(S) = sum of w_i for i in S.
    pub fn additive(n: u8, weights: &[f64]) -> Self {
        assert_eq!(weights.len(), n as usize);
        Self::from_fn(n, |c| c.players().map(|i| weights[i as usize]).sum())
    }

    /// A unanimity game: v(S) = 1 if S ⊇ T, else 0.
    pub fn unanimity(n: u8, target: &Coalition) -> Self {
        let target_mask = target.mask();
        Self::from_fn(n, |c| {
            if (c.mask() & target_mask) == target_mask {
                1.0
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_game_values() {
        let vf = ValueFunction::from_fn(3, |c| c.count() as f64 * 10.0);
        assert_eq!(vf.value(&Coalition::empty(3)), 0.0);
        assert_eq!(vf.value(&Coalition::singleton(3, 0)), 10.0);
        assert_eq!(vf.grand_value(), 30.0);
    }

    #[test]
    fn marginal_contribution() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 100.0 } else { 0.0 });
        let s = Coalition::from_indices(3, &[0, 1]);
        // v({0,1,2}) - v({0,1}) = 100 - 0 = 100
        assert!((vf.marginal_contribution(2, &s) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn superadditive_game() {
        // Grand coalition only has value — classic superadditive
        let vf = ValueFunction::from_fn(2, |c| if c.is_grand() { 10.0 } else { 0.0 });
        assert!(vf.is_superadditive());
        assert!(!vf.is_subadditive());
    }

    #[test]
    fn additive_is_both() {
        let vf = ValueFunction::additive(3, &[1.0, 2.0, 3.0]);
        assert!(vf.is_superadditive());
        assert!(vf.is_subadditive());
        // Additive is convex
        assert!(vf.is_convex());
    }

    #[test]
    fn convex_game_detection() {
        // v(S) = |S|^2 — convex since marginal contributions increase
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64).powi(2));
        assert!(vf.is_convex());
    }

    #[test]
    fn unanimity_game() {
        let target = Coalition::from_indices(3, &[0, 1]);
        let vf = ValueFunction::unanimity(3, &target);
        assert_eq!(vf.value(&target), 1.0);
        assert_eq!(vf.value(&Coalition::grand(3)), 1.0);
        assert_eq!(vf.value(&Coalition::singleton(3, 0)), 0.0);
    }

    #[test]
    fn from_map_preserves_explicit_values() {
        let mut map = std::collections::HashMap::new();
        map.insert(0b111, 100.0);
        map.insert(0b001, 10.0);
        map.insert(0b010, 20.0);
        map.insert(0b100, 30.0);
        let vf = ValueFunction::from_map(3, map);
        assert_eq!(vf.grand_value(), 100.0);
        assert_eq!(vf.value_by_mask(0), 0.0); // empty defaults to 0
    }

    #[test]
    fn voting_game_simple() {
        // Quorum: need >= 2 of 3 players
        let vf = ValueFunction::from_fn(3, |c| if c.count() >= 2 { 1.0 } else { 0.0 });
        assert!(!vf.is_convex());
        assert!(vf.is_superadditive());
    }
}
