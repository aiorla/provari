//! The variants provari proves, each a type that names it: the search takes
//! one as a type parameter (`dfpn::Variant`), so that each variant compiles
//! to its own search and none tests at run time which variant it searches.

/// Atomic chess under Lichess's rules (`dfpn::atomic`).
pub enum Atomic {}

/// Antichess under Lichess's rules (`dfpn::antichess`).
#[cfg(feature = "antichess")]
pub enum Antichess {}
