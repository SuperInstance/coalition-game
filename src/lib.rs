//! # coalition-game
//!
//! Cooperative game theory primitives: Shapley value, core, nucleolus, and
//! stable coalition analysis. Zero external dependencies beyond `serde`.
//!
//! ## Quick start
//!
//! ```
//! use coalition_game::{Coalition, ValueFunction, ShapleyValue};
//!
//! // Define a 3-player game
//! let vf = ValueFunction::from_fn(3, |s| {
//!     if s.count() == 3 { 100.0 }
//!     else if s.count() == 2 { 40.0 }
//!     else { 0.0 }
//! });
//!
//! let shapley = ShapleyValue::compute_exact(&vf);
//! assert!((shapley.player(0) - shapley.player(1)).abs() < 1e-9);
//! assert!((shapley.as_slice().iter().sum::<f64>() - 100.0).abs() < 1e-9);
//! ```

#![forbid(unsafe_code)]

pub mod coalition;
pub mod core;
pub mod nucleolus;
pub mod shapley;
pub mod stability;
pub mod value;

pub use coalition::Coalition;
pub use core::Core;
pub use nucleolus::Nucleolus;
pub use shapley::ShapleyValue;
pub use stability::StabilityAnalysis;
pub use value::ValueFunction;
