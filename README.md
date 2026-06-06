# coalition-game

**Cooperative game theory in Rust: Shapley value, core, nucleolus, and stable coalition analysis.**

Every day, autonomous agents face a fundamental question: *who should I work with, and how do we split the gains?* Cooperative game theory provides the mathematical framework to answer this — from allocating costs in shared infrastructure to dividing rewards in multi-agent AI systems. `coalition-game` brings these tools to Rust with zero external dependencies beyond `serde`.

## Why this crate exists

You're building a multi-agent system where agents form alliances. One agent has data, another has compute, a third has domain expertise. Together they're worth more than the sum of their parts. How do you:

- **Fairly divide** the joint payoff? → Shapley value
- **Verify** that no subgroup would defect? → Core analysis
- **Find** the unique fair division that minimizes complaints? → Nucleolus
- **Check** if a coalition structure is stable? → Stability analysis

This crate implements the four pillars of cooperative game theory as composable, serializable Rust types. No C dependencies, no BLAS, no LaTeX — just clean, tested math.

## The metaphor: coalition games as cooperative intelligence

```
    Agent ○──── has data
    Agent ○──── has compute     ───→  Coalition ◉──── worth more together
    Agent ○──── has domain knowledge     than apart
```

Think of agents forming alliances. A single agent might be worth nothing alone, but the right coalition can solve problems none could tackle individually. Cooperative game theory is the mathematics of *cooperative intelligence* — it tells us which alliances form, how they share rewards, and whether those arrangements are stable.

The Shapley value is the unique fair division that satisfies four intuitive axioms. The core tells us whether an allocation is stable against defection. The nucleolus finds the allocation that minimizes the maximum complaint. Together, they form a complete toolkit for reasoning about cooperation.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    coalition-game                         │
│                                                          │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ coalition  │───▶│   value      │───▶│   shapley    │  │
│  │            │    │              │    │              │  │
│  │ Coalition  │    │ ValueFunction│    │ ShapleyValue │  │
│  │ bitmask    │    │ super/sub    │    │ exact/sample │  │
│  │ lattice    │    │ convex check │    │ axioms check │  │
│  └────────────┘    └──────┬───────┘    └──────────────┘  │
│                           │                               │
│                    ┌──────▼───────┐                       │
│                    │    core      │                       │
│                    │              │                       │
│                    │ Core         │                       │
│                    │ Bondareva-   │                       │
│                    │   Shapley    │                       │
│                    │ Imputation   │                       │
│                    └──────┬───────┘                       │
│                           │                               │
│              ┌────────────┼────────────┐                  │
│              ▼            ▼            ▼                  │
│      ┌──────────┐  ┌──────────┐  ┌──────────────┐        │
│      │nucleolus │  │stability │  │  (your game) │        │
│      │          │  │          │  │              │        │
│      │Nucleolus │  │ Nash-    │  │  Build new   │        │
│      │ excess   │  │ stable   │  │  solution    │        │
│      │ lex min  │  │ core-    │  │  concepts    │        │
│      │          │  │ stable   │  │  on top of   │        │
│      └──────────┘  └──────────┘  │  these types │        │
│                                   └──────────────┘        │
└──────────────────────────────────────────────────────────┘
```

## Module reference

| Module | Purpose | Key types |
|--------|---------|-----------|
| `coalition` | Subset of players as bitmask | `Coalition` |
| `value` | Characteristic function v(S) | `ValueFunction` |
| `shapley` | Fair payoff division | `ShapleyValue` |
| `core` | Stable imputation set | `Core`, `Imputation` |
| `nucleolus` | Minimize max complaint | `Nucleolus` |
| `stability` | Coalition structure stability | `CoalitionStructure`, `StabilityAnalysis` |

## Quick start

```rust
use coalition_game::{Coalition, ValueFunction, ShapleyValue, Core};

fn main() {
    // Define a 3-player game where only the grand coalition has value
    let vf = ValueFunction::from_fn(3, |c| {
        if c.is_grand() { 100.0 } else { 0.0 }
    });

    // Compute Shapley values — each player gets 100/3
    let sv = ShapleyValue::compute_exact(&vf);
    println!("Shapley values: {:?}", sv.as_slice());
    assert!((sv.player(0) - 100.0 / 3.0).abs() < 1e-9);

    // Check the core
    let core = Core::analyze(&vf);
    println!("Core non-empty: {}", core.is_non_empty());
}
```

## Examples

### Glove game

Two players have left gloves, one has a right glove. A pair is worth 1:

```rust
use coalition_game::{Coalition, ValueFunction, ShapleyValue, Core};

let vf = ValueFunction::from_fn(3, |c| {
    let left = c.players().filter(|&i| i < 2).count() as f64;
    let right = c.players().filter(|&i| i == 2).count() as f64;
    left.min(right) // each matched pair is worth 1
});

// Player 2 (right glove) has more bargaining power
let sv = ShapleyValue::compute_exact(&vf);
println!("Left glove 1: {}", sv.player(0)); // ≈ 1/6
println!("Left glove 2: {}", sv.player(1)); // ≈ 1/6
println!("Right glove:  {}", sv.player(2)); // ≈ 2/3

// Core: right-glove player gets everything
let core = Core::analyze(&vf);
assert!(core.is_non_empty());
```

### Airport game (cost allocation)

Three airlines share a runway. Airline 0 needs 1km, airline 1 needs 2km, airline 2 needs 3km. Cost of a runway serving coalition S = max runway needed:

```rust
use coalition_game::{ValueFunction, Nucleolus};

let costs = [10.0, 20.0, 30.0];
let vf = ValueFunction::from_fn(3, |c| {
    c.players()
        .map(|i| costs[i as usize])
        .fold(0.0f64, f64::max)
});

let nuc = Nucleolus::compute(&vf);
println!("Cost allocation: {:?}", nuc.as_slice());
// Nucleolus allocates incremental costs:
// Player 0: 10/3, Player 1: (20-10)/2 + 10/3, Player 2: 30 - (sum of above)
```

### Voting game with stability analysis

A simple majority voting game with 4 players:

```rust
use coalition_game::{ValueFunction, CoalitionStructure, StabilityAnalysis};

let vf = ValueFunction::from_fn(4, |c| {
    if c.count() >= 3 { 1.0 } else { 0.0 }
});

// Start with pairs: {0,1} and {2,3}
let structure = CoalitionStructure::new(4, vec![
    vec![0, 1],
    vec![2, 3],
]);

let analysis = StabilityAnalysis::new(4);
let report = analysis.analyze(&structure, &vf);

println!("Nash stable: {}", report.nash_stable);
println!("Core stable: {}", report.core_stable);
println!("Blocking coalitions: {:?}", report.blocking_coalitions);
```

### Monte Carlo Shapley for large games

```rust
use coalition_game::{ValueFunction, ShapleyValue};

let vf = ValueFunction::from_fn(8, |c| {
    if c.is_grand() { 1000.0 }
    else if c.count() >= 6 { 500.0 }
    else { c.count() as f64 * 10.0 }
});

// Exact computation needs 8! = 40320 permutations — feasible but slow
// Sample instead:
let sv = ShapleyValue::compute_sampled(&vf, 5000, 42);
println!("Estimated Shapley values: {:?}", sv.as_slice());
println!("Efficiency check: sum = {:.2}", sv.as_slice().iter().sum::<f64>());
```

## Mathematical foundations

### Shapley value

For a game (N, v) with player set N, the Shapley value of player i is:

```
φ_i(v) = Σ_{S ⊆ N\{i}} [|S|! · (|N|-|S|-1)! / |N|!] · [v(S∪{i}) - v(S)]
```

**Four axioms uniquely determine the Shapley value:**

1. **Efficiency**: Σ_i φ_i(v) = v(N) — all value is distributed
2. **Symmetry**: If players i,j are interchangeable, φ_i = φ_j
3. **Dummy player**: If player i never adds value, φ_i = 0
4. **Additivity**: φ_i(v + w) = φ_i(v) + φ_i(w)

### Core

The core is the set of imputations x ∈ ℝⁿ where:

```
Σ_{i ∈ S} x_i ≥ v(S)   for all S ⊆ N    (coalitional rationality)
Σ_{i ∈ N} x_i = v(N)                       (efficiency)
```

**Bondareva-Shapley theorem**: The core is non-empty if and only if the game is *balanced* — for every balanced collection of coalitions {S₁, ..., Sₖ} with balancing weights {δ₁, ..., δₖ}, we have Σ_j δ_j · v(S_j) ≤ v(N).

For **convex games** (v(S∪T) + v(S∩T) ≥ v(S) + v(T)), the core is always non-empty. The extreme points of the core correspond to marginal contribution vectors for each permutation.

### Nucleolus

The excess of coalition S under imputation x is:

```
e(S, x) = v(S) - Σ_{i ∈ S} x_i
```

The nucleolus is the unique imputation that lexicographically minimizes the sorted vector of excesses. It always exists, is unique, and lies in the core when the core is non-empty.

### Stability concepts

Given a coalition structure π = {B₁, ..., Bₖ}:

- **Nash-stable**: No player wants to unilaterally move to another block
- **Individually stable**: No player can move and improve without hurting someone in the receiving block
- **Core-stable**: No coalition S ⊆ N can deviate and make all members strictly better off

## Design decisions

### Bitmask representation

Coalitions use `u64` bitmasks, supporting up to 63 players. This makes set operations (union, intersection, complement) single CPU instructions. For most cooperative game theory applications, 63 players is more than sufficient.

### Value storage

`ValueFunction` stores all 2ⁿ values in a `HashMap<u64, f64>`. For n ≤ 20, this uses at most a few MB. For larger games, consider implementing a `ValueFunction` trait with a computational closure instead.

### No trait abstractions

Types are concrete, not generic. This keeps the API simple and the codebase readable. The Shapley value returns `Vec<f64>`, not some abstract payoff type. If you need genericity, wrap these types.

### Tolerance constants

Floating-point comparisons use `1e-9` tolerance throughout. Cooperative game theory involves comparing real numbers, and exact equality is rarely meaningful. If you need tighter tolerances, file an issue.

## API overview

### Coalition operations

```rust
use coalition_game::Coalition;

let a = Coalition::from_indices(4, &[0, 1]);
let b = Coalition::from_indices(4, &[1, 2, 3]);

// Set operations
let union = a | b;        // {0,1,2,3}
let inter = a & b;        // {1}
let diff = a - b;         // {0}

// Lattice operations
assert!(a.is_subset_of(&union));
assert!(a.is_strict_subset_of(&union));

// Iterate
for player in a.players() {
    println!("player {}", player);
}

// Enumerate all coalitions
for c in Coalition::all(3) {
    println!("{:?}: {} players", c, c.count());
}
```

### ValueFunction classification

```rust
use coalition_game::ValueFunction;

let vf = ValueFunction::from_fn(4, |c| (c.count() as f64).powi(2));

println!("Superadditive: {}", vf.is_superadditive()); // true
println!("Convex: {}", vf.is_convex());               // true

// Marginal contribution of player 2 to coalition {0,1}
let s = Coalition::from_indices(4, &[0, 1]);
let mc = vf.marginal_contribution(2, &s);
```

### Serialization

All public types derive `Serialize` and `Deserialize`:

```rust
use coalition_game::{Coalition, ValueFunction};
use serde_json;

let vf = ValueFunction::from_fn(3, |c| c.count() as f64);
let json = serde_json::to_string(&vf).unwrap();
let restored: ValueFunction = serde_json::from_str(&json).unwrap();
```

## Crate features

- **Edition 2024** — latest Rust idioms
- **`serde` only dependency** — no heavy math libraries
- **50+ tests** — Shapley axioms, glove games, airport games, voting games, core existence
- **`#[forbid(unsafe_code)]`** — no unsafe blocks
- **`cargo clippy` clean** — no warnings

## When to use this crate

- **Multi-agent AI**: Allocate rewards among cooperating agents
- **Cost sharing**: Divide shared infrastructure costs (airports, networks)
- **Voting power**: Measure real influence in weighted voting systems
- **Revenue sharing**: Split joint revenue fairly among contributors
- **Research**: Prototype cooperative game theory algorithms
- **Education**: Learn game theory with runnable code

## When NOT to use this crate

- **n > 20**: Exponential value storage becomes expensive. Use approximate methods.
- **Non-transferable utility (NTU)**: This crate assumes side payments (transferable utility).
- **Sequential/extensive-form games**: This is for one-shot cooperative games only.

## License

MIT
