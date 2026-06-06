//! Stability analysis for coalition structures.
//!
//! A coalition structure is a partition of the player set into disjoint
//! coalitions. Stability concepts determine whether any player or group
//! has an incentive to deviate.
//!
//! Concepts implemented:
//! - **Nash-stable**: no player wants to unilaterally move to another coalition
//! - **Individually stable**: no player can move and be better off without
//!   making someone in the receiving coalition worse off
//! - **Core-stable**: no coalition can deviate and all be better off

use crate::coalition::Coalition;
use crate::value::ValueFunction;
use serde::{Deserialize, Serialize};

/// A coalition structure: a partition of players into disjoint coalitions.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CoalitionStructure {
    /// Groups of player indices (each group is a coalition).
    groups: Vec<Vec<u8>>,
    /// Number of players in the universe.
    n: u8,
}

impl CoalitionStructure {
    /// Create a coalition structure from groups.
    ///
    /// Each group is a vec of player indices. Groups must partition {0, ..., n-1}.
    pub fn new(n: u8, groups: Vec<Vec<u8>>) -> Self {
        // Validate: all players appear exactly once
        let mut seen = vec![false; n as usize];
        for g in &groups {
            for &p in g {
                assert!(p < n, "player index out of bounds");
                assert!(!seen[p as usize], "player {} appears in multiple groups", p);
                seen[p as usize] = true;
            }
        }
        for (i, s) in seen.iter().enumerate() {
            assert!(*s, "player {} missing from structure", i);
        }
        Self { groups, n }
    }

    /// The grand coalition structure (everyone together).
    pub fn grand(n: u8) -> Self {
        Self {
            groups: vec![(0..n).collect()],
            n,
        }
    }

    /// The singleton structure (everyone alone).
    pub fn singleton(n: u8) -> Self {
        Self {
            groups: (0..n).map(|i| vec![i]).collect(),
            n,
        }
    }

    /// Number of players.
    pub fn n(&self) -> u8 {
        self.n
    }

    /// Number of groups (coalitions in the structure).
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Access the groups.
    pub fn groups(&self) -> &[Vec<u8>] {
        &self.groups
    }

    /// Find which group index contains player `i`.
    pub fn group_of(&self, player: u8) -> usize {
        for (idx, g) in self.groups.iter().enumerate() {
            if g.contains(&player) {
                return idx;
            }
        }
        panic!("player {} not found in structure", player);
    }

    /// Compute the value of group `idx` using the characteristic function.
    pub fn group_value(&self, idx: usize, vf: &ValueFunction) -> f64 {
        let c = Coalition::from_indices(self.n, &self.groups[idx]);
        vf.value(&c)
    }

    /// Total value of the coalition structure.
    pub fn total_value(&self, vf: &ValueFunction) -> f64 {
        self.groups
            .iter()
            .enumerate()
            .map(|(i, _)| self.group_value(i, vf))
            .sum()
    }

    /// Derive a per-player payoff using Shapley value within each group.
    pub fn player_payoffs(&self, vf: &ValueFunction) -> Vec<f64> {
        let mut payoffs = vec![0.0; self.n as usize];
        for g in &self.groups {
            if g.len() == 1 {
                let c = Coalition::singleton(self.n, g[0]);
                payoffs[g[0] as usize] = vf.value(&c);
            } else {
                let c = Coalition::from_indices(self.n, g);
                // Use equal split within each group
                let val = vf.value(&c);
                let share = val / g.len() as f64;
                for &p in g {
                    payoffs[p as usize] = share;
                }
            }
        }
        payoffs
    }

    /// Create a new structure where player has moved from one group to another.
    fn with_move(&self, player: u8, from_group: usize, to_group: usize) -> CoalitionStructure {
        let mut new_groups = self.groups.clone();
        new_groups[from_group].retain(|&p| p != player);
        new_groups[to_group].push(player);
        // Remove empty groups
        new_groups.retain(|g| !g.is_empty());
        CoalitionStructure {
            groups: new_groups,
            n: self.n,
        }
    }

    /// Create a structure where player forms a new singleton.
    fn with_leave(&self, player: u8, from_group: usize) -> CoalitionStructure {
        let mut new_groups = self.groups.clone();
        new_groups[from_group].retain(|&p| p != player);
        new_groups.push(vec![player]);
        CoalitionStructure {
            groups: new_groups,
            n: self.n,
        }
    }
}

/// Results of stability analysis.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StabilityReport {
    /// Is the structure Nash-stable?
    pub nash_stable: bool,
    /// Is the structure individually stable?
    pub individually_stable: bool,
    /// Is the structure core-stable?
    pub core_stable: bool,
    /// Players who want to deviate (Nash).
    pub nash_unstable_players: Vec<u8>,
    /// Blocking coalitions (for core stability).
    pub blocking_coalitions: Vec<Vec<u8>>,
}

/// Stability analysis for coalition structures.
#[derive(Clone, Serialize, Deserialize)]
pub struct StabilityAnalysis {
    n: u8,
}

impl StabilityAnalysis {
    /// Create a new stability analyzer for `n` players.
    pub fn new(n: u8) -> Self {
        Self { n }
    }

    /// Check Nash-stability: no player wants to unilaterally change groups.
    ///
    /// A structure is Nash-stable if for every player `i`, moving to any
    /// other existing group or forming a singleton does not improve `i`'s payoff.
    pub fn is_nash_stable(
        &self,
        structure: &CoalitionStructure,
        vf: &ValueFunction,
    ) -> (bool, Vec<u8>) {
        let payoffs = structure.player_payoffs(vf);
        let mut unstable = Vec::new();

        for player in 0..self.n {
            let current_payoff = payoffs[player as usize];
            let current_group = structure.group_of(player);

            // Check: move to another existing group
            for target_group in 0..structure.num_groups() {
                if target_group == current_group {
                    continue;
                }
                let new_struct = structure.with_move(player, current_group, target_group);
                let new_payoffs = new_struct.player_payoffs(vf);
                if new_payoffs[player as usize] > current_payoff + 1e-9 {
                    unstable.push(player);
                    break;
                }
            }

            if unstable.last() == Some(&player) {
                continue;
            }

            // Check: form own singleton
            if structure.groups()[current_group].len() > 1 {
                let new_struct = structure.with_leave(player, current_group);
                let new_payoffs = new_struct.player_payoffs(vf);
                if new_payoffs[player as usize] > current_payoff + 1e-9 {
                    unstable.push(player);
                }
            }
        }

        (unstable.is_empty(), unstable)
    }

    /// Check individual stability: no player can move to another group
    /// and be better off without making an existing member of that group worse off.
    pub fn is_individually_stable(
        &self,
        structure: &CoalitionStructure,
        vf: &ValueFunction,
    ) -> bool {
        let payoffs = structure.player_payoffs(vf);

        for player in 0..self.n {
            let current_payoff = payoffs[player as usize];
            let current_group = structure.group_of(player);

            for target_group in 0..structure.num_groups() {
                if target_group == current_group {
                    continue;
                }

                // Would player be better off?
                let new_struct = structure.with_move(player, current_group, target_group);
                let new_payoffs = new_struct.player_payoffs(vf);

                if new_payoffs[player as usize] <= current_payoff + 1e-9 {
                    continue;
                }

                // Would any existing member of target_group be worse off?
                let existing_members = &structure.groups()[target_group];
                let any_worse = existing_members
                    .iter()
                    .any(|&m| new_payoffs[m as usize] < payoffs[m as usize] - 1e-9);

                if !any_worse {
                    return false;
                }
            }
        }
        true
    }

    /// Check core stability: no coalition can deviate and all be better off.
    ///
    /// A structure is core-stable if there is no coalition S such that
    /// every member of S gets strictly more in the deviation.
    pub fn is_core_stable(
        &self,
        structure: &CoalitionStructure,
        vf: &ValueFunction,
    ) -> (bool, Vec<Vec<u8>>) {
        let payoffs = structure.player_payoffs(vf);
        let mut blocking = Vec::new();

        // Check all non-trivial coalitions as potential deviating groups
        for c in Coalition::all(self.n) {
            if c.count() < 2 {
                continue;
            }

            let members: Vec<u8> = c.players().collect();
            let coalition_val = vf.value(&c);

            // If the coalition forms, equal split within
            let per_member = coalition_val / members.len() as f64;

            // Is every member strictly better off?
            let all_better = members
                .iter()
                .all(|&m| per_member > payoffs[m as usize] + 1e-9);

            if all_better {
                blocking.push(members);
            }
        }

        (blocking.is_empty(), blocking)
    }

    /// Run full stability analysis.
    pub fn analyze(&self, structure: &CoalitionStructure, vf: &ValueFunction) -> StabilityReport {
        let (nash, nash_players) = self.is_nash_stable(structure, vf);
        let ind = self.is_individually_stable(structure, vf);
        let (core, blocking) = self.is_core_stable(structure, vf);

        StabilityReport {
            nash_stable: nash,
            individually_stable: ind,
            core_stable: core,
            nash_unstable_players: nash_players,
            blocking_coalitions: blocking,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grand_coalition_nash_stable_for_superadditive() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 90.0 } else { 0.0 });
        let structure = CoalitionStructure::grand(3);
        let analysis = StabilityAnalysis::new(3);
        let (stable, _) = analysis.is_nash_stable(&structure, &vf);
        assert!(stable);
    }

    #[test]
    fn singleton_structure_unstable_when_cooperation_pays() {
        let vf = ValueFunction::from_fn(2, |c| if c.is_grand() { 10.0 } else { 0.0 });
        let structure = CoalitionStructure::singleton(2);
        let analysis = StabilityAnalysis::new(2);
        let (stable, unstable) = analysis.is_nash_stable(&structure, &vf);
        assert!(!stable);
        assert_eq!(unstable.len(), 2);
    }

    #[test]
    fn full_analysis_grand_coalition() {
        let vf = ValueFunction::from_fn(3, |c| (c.count() as f64) * 20.0);
        let structure = CoalitionStructure::grand(3);
        let analysis = StabilityAnalysis::new(3);
        let report = analysis.analyze(&structure, &vf);

        // Grand coalition is stable when it has the most value
        assert!(report.nash_stable);
        assert!(report.individually_stable);
    }

    #[test]
    fn coalition_structure_value() {
        let vf = ValueFunction::from_fn(4, |c| c.count() as f64 * 5.0);
        let structure = CoalitionStructure::new(4, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(structure.num_groups(), 2);
        let total = structure.total_value(&vf);
        assert!((total - 20.0).abs() < 1e-9); // 2*5 + 2*5 = 10 + 10
    }

    #[test]
    fn player_payoffs_equal_split() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 60.0 } else { 0.0 });
        let structure = CoalitionStructure::grand(3);
        let payoffs = structure.player_payoffs(&vf);
        for p in &payoffs {
            assert!((*p - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn core_stability_blocking_coalition() {
        // A pair gets more than their share in the grand coalition
        let vf = ValueFunction::from_fn(3, |c| {
            match c.mask() {
                0b011 => 50.0, // {0,1} is very valuable
                0b111 => 60.0, // grand coalition
                _ => 0.0,
            }
        });
        let structure = CoalitionStructure::grand(3);
        let analysis = StabilityAnalysis::new(3);
        let (stable, blocking) = analysis.is_core_stable(&structure, &vf);
        assert!(!stable);
        assert!(blocking.iter().any(|b| b.len() == 2));
    }

    #[test]
    fn individual_stability_checked() {
        let vf = ValueFunction::from_fn(3, |c| if c.is_grand() { 90.0 } else { 0.0 });
        let structure = CoalitionStructure::grand(3);
        let analysis = StabilityAnalysis::new(3);
        assert!(analysis.is_individually_stable(&structure, &vf));
    }

    #[test]
    #[should_panic]
    fn invalid_structure_missing_player() {
        CoalitionStructure::new(3, vec![vec![0, 1]]);
    }

    #[test]
    #[should_panic]
    fn invalid_structure_duplicate_player() {
        CoalitionStructure::new(3, vec![vec![0, 1], vec![1, 2]]);
    }
}
