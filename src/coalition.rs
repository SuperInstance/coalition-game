//! Coalition representation — a subset of players from a fixed universe.
//!
//! Coalitions are stored as bitmasks for efficient set operations. The lattice
//! of all coalitions over `n` players has `2^n` elements.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitAnd, BitOr, BitOrAssign, Sub};

/// A coalition of players represented as a bitmask.
///
/// Player `i` is a member when bit `i` is set. The empty set is the coalition
/// with mask `0`; the grand coalition of `n` players has mask `(1 << n) - 1`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Coalition {
    /// Bitmask of member players.
    mask: u64,
    /// Number of players in the universe (determines valid bit range).
    num_players: u8,
}

impl Coalition {
    /// The empty coalition for a universe of `n` players.
    pub fn empty(n: u8) -> Self {
        Self {
            mask: 0,
            num_players: n,
        }
    }

    /// The grand coalition containing all `n` players.
    pub fn grand(n: u8) -> Self {
        Self {
            mask: (1u64 << n) - 1,
            num_players: n,
        }
    }

    /// A singleton coalition containing only player `i`.
    ///
    /// # Panics
    /// Panics if `i >= n`.
    pub fn singleton(n: u8, i: u8) -> Self {
        assert!(i < n, "player index out of bounds");
        Self {
            mask: 1u64 << i,
            num_players: n,
        }
    }

    /// Build from a bitmask and player count.
    pub fn from_mask(n: u8, mask: u64) -> Self {
        Self {
            mask,
            num_players: n,
        }
    }

    /// Build from a slice of player indices.
    pub fn from_indices(n: u8, players: &[u8]) -> Self {
        let mut mask = 0u64;
        for &p in players {
            assert!(p < n, "player index out of bounds");
            mask |= 1u64 << p;
        }
        Self {
            mask,
            num_players: n,
        }
    }

    /// Number of players in the universe.
    pub fn num_players(&self) -> u8 {
        self.num_players
    }

    /// Raw bitmask.
    pub fn mask(&self) -> u64 {
        self.mask
    }

    /// Number of players in this coalition.
    pub fn count(&self) -> u32 {
        self.mask.count_ones()
    }

    /// Is the coalition empty?
    pub fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// Is this the grand coalition?
    pub fn is_grand(&self) -> bool {
        self.mask == (1u64 << self.num_players) - 1
    }

    /// Does this coalition contain player `i`?
    pub fn contains(&self, i: u8) -> bool {
        assert!(i < self.num_players, "player index out of bounds");
        (self.mask & (1u64 << i)) != 0
    }

    /// Add a player to the coalition.
    pub fn insert(&mut self, i: u8) {
        assert!(i < self.num_players, "player index out of bounds");
        self.mask |= 1u64 << i;
    }

    /// Remove a player from the coalition.
    pub fn remove(&mut self, i: u8) {
        assert!(i < self.num_players, "player index out of bounds");
        self.mask &= !(1u64 << i);
    }

    /// Is `self` a subset of `other`?
    pub fn is_subset_of(&self, other: &Coalition) -> bool {
        debug_assert_eq!(self.num_players, other.num_players);
        (self.mask & other.mask) == self.mask
    }

    /// Is `self` a strict (proper) subset of `other`?
    pub fn is_strict_subset_of(&self, other: &Coalition) -> bool {
        self.mask != other.mask && self.is_subset_of(other)
    }

    /// Is `self` a superset of `other`?
    pub fn is_superset_of(&self, other: &Coalition) -> bool {
        other.is_subset_of(self)
    }

    /// Iterator over member player indices.
    pub fn players(&self) -> impl Iterator<Item = u8> + use<'_> {
        let n = self.num_players;
        (0..n).filter(move |&i| (self.mask & (1u64 << i)) != 0)
    }

    /// Iterate over all coalitions for a universe of `n` players.
    /// Yields `2^n` coalitions from empty to grand.
    pub fn all(n: u8) -> impl Iterator<Item = Coalition> + use<> {
        let total = if n == 0 { 1u64 } else { 1u64 << n };
        (0..total).map(move |mask| Coalition {
            mask,
            num_players: n,
        })
    }

    /// Iterate over all coalitions of exactly `k` players.
    pub fn of_size(n: u8, k: u8) -> impl Iterator<Item = Coalition> + use<> {
        Coalition::all(n).filter(move |c| c.count() == k as u32)
    }
}

impl BitOr for Coalition {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        debug_assert_eq!(self.num_players, rhs.num_players);
        Self {
            mask: self.mask | rhs.mask,
            num_players: self.num_players,
        }
    }
}

impl BitOrAssign for Coalition {
    fn bitor_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.num_players, rhs.num_players);
        self.mask |= rhs.mask;
    }
}

impl BitAnd for Coalition {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        debug_assert_eq!(self.num_players, rhs.num_players);
        Self {
            mask: self.mask & rhs.mask,
            num_players: self.num_players,
        }
    }
}

impl Sub for Coalition {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        debug_assert_eq!(self.num_players, rhs.num_players);
        Self {
            mask: self.mask & !rhs.mask,
            num_players: self.num_players,
        }
    }
}

impl fmt::Debug for Coalition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let players: Vec<u8> = self.players().collect();
        write!(
            f,
            "Coalition{{{}}}",
            players
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_grand() {
        let e = Coalition::empty(3);
        let g = Coalition::grand(3);
        assert!(e.is_empty());
        assert!(g.is_grand());
        assert_eq!(e.count(), 0);
        assert_eq!(g.count(), 3);
    }

    #[test]
    fn singleton() {
        let s = Coalition::singleton(4, 2);
        assert_eq!(s.count(), 1);
        assert!(s.contains(2));
        assert!(!s.contains(0));
    }

    #[test]
    fn from_indices() {
        let c = Coalition::from_indices(4, &[0, 2, 3]);
        assert_eq!(c.count(), 3);
        assert!(c.contains(0));
        assert!(!c.contains(1));
        assert!(c.contains(2));
        assert!(c.contains(3));
    }

    #[test]
    fn set_operations() {
        let a = Coalition::from_indices(3, &[0, 1]);
        let b = Coalition::from_indices(3, &[1, 2]);
        let union = a | b;
        assert_eq!(union.count(), 3);
        assert!(union.is_grand());

        let inter = a & b;
        assert_eq!(inter.count(), 1);
        assert!(inter.contains(1));

        let diff = a - b;
        assert_eq!(diff.count(), 1);
        assert!(diff.contains(0));
    }

    #[test]
    fn subset_superset() {
        let a = Coalition::from_indices(3, &[0]);
        let b = Coalition::from_indices(3, &[0, 1]);
        assert!(a.is_subset_of(&b));
        assert!(b.is_superset_of(&a));
        assert!(!a.is_superset_of(&b));
        assert!(a.is_strict_subset_of(&b));
    }

    #[test]
    fn all_coalitions_count() {
        let all: Vec<_> = Coalition::all(3).collect();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn of_size_count() {
        let pairs: Vec<_> = Coalition::of_size(4, 2).collect();
        assert_eq!(pairs.len(), 6); // C(4,2)
    }

    #[test]
    #[should_panic]
    fn singleton_out_of_bounds() {
        Coalition::singleton(3, 3);
    }

    #[test]
    fn players_iterator() {
        let c = Coalition::from_indices(5, &[1, 3, 4]);
        let players: Vec<u8> = c.players().collect();
        assert_eq!(players, vec![1, 3, 4]);
    }

    #[test]
    fn insert_and_remove() {
        let mut c = Coalition::empty(4);
        c.insert(1);
        c.insert(3);
        assert_eq!(c.count(), 2);
        assert!(c.contains(1));
        assert!(c.contains(3));
        c.remove(1);
        assert_eq!(c.count(), 1);
        assert!(!c.contains(1));
    }

    #[test]
    fn bitmask_roundtrip() {
        let c = Coalition::from_mask(5, 0b10110);
        assert_eq!(c.mask(), 0b10110);
        let c2 = Coalition::from_mask(c.num_players(), c.mask());
        assert_eq!(c, c2);
    }
}
