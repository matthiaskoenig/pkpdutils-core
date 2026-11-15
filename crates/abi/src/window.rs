//! The sample labels which key the terminal windows, the eight typed labels of
//! `_encode_window_key` in `pkpdutils.nca.options`.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::nonfinite;

/// A sample label: a scalar or a (nested) tuple, never coerced to a string.
///
/// A date or a datetime stays the ISO 8601 string of the wire, the kernel never computes with a
/// label. The order is total and deterministic: by variant (`Tuple`, `None`, `Datetime`, `Date`,
/// `Bool`, `Str`, `Int`, `Float` in that order), then by value, floats by `f64::total_cmp`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(from = "LabelWire", into = "LabelWire")]
pub enum WindowLabel {
    /// A tuple of labels, nesting allowed.
    Tuple(Vec<WindowLabel>),
    /// The label `None`.
    None,
    /// An ISO 8601 datetime.
    Datetime(String),
    /// An ISO 8601 date.
    Date(String),
    /// A boolean.
    Bool(bool),
    /// A string.
    Str(String),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
}

/// The wire form, `{"type": ..., "value": ...}`; an unknown `type` names the known ones.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
enum LabelWire {
    Tuple(Vec<WindowLabel>),
    None(()),
    Datetime(String),
    Date(String),
    Bool(bool),
    Str(String),
    Int(i64),
    Float(#[serde(with = "nonfinite")] f64),
}

impl From<LabelWire> for WindowLabel {
    fn from(w: LabelWire) -> Self {
        match w {
            LabelWire::Tuple(v) => Self::Tuple(v),
            LabelWire::None(()) => Self::None,
            LabelWire::Datetime(s) => Self::Datetime(s),
            LabelWire::Date(s) => Self::Date(s),
            LabelWire::Bool(b) => Self::Bool(b),
            LabelWire::Str(s) => Self::Str(s),
            LabelWire::Int(i) => Self::Int(i),
            LabelWire::Float(x) => Self::Float(x),
        }
    }
}

impl From<WindowLabel> for LabelWire {
    fn from(l: WindowLabel) -> Self {
        match l {
            WindowLabel::Tuple(v) => Self::Tuple(v),
            WindowLabel::None => Self::None(()),
            WindowLabel::Datetime(s) => Self::Datetime(s),
            WindowLabel::Date(s) => Self::Date(s),
            WindowLabel::Bool(b) => Self::Bool(b),
            WindowLabel::Str(s) => Self::Str(s),
            WindowLabel::Int(i) => Self::Int(i),
            WindowLabel::Float(x) => Self::Float(x),
        }
    }
}

impl WindowLabel {
    fn rank(&self) -> u8 {
        match self {
            Self::Tuple(_) => 0,
            Self::None => 1,
            Self::Datetime(_) => 2,
            Self::Date(_) => 3,
            Self::Bool(_) => 4,
            Self::Str(_) => 5,
            Self::Int(_) => 6,
            Self::Float(_) => 7,
        }
    }
}

impl Ord for WindowLabel {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Tuple(a), Self::Tuple(b)) => a.cmp(b),
            (Self::Datetime(a), Self::Datetime(b))
            | (Self::Date(a), Self::Date(b))
            | (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => a.total_cmp(b),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

impl PartialOrd for WindowLabel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for WindowLabel {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for WindowLabel {}

impl Hash for WindowLabel {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match self {
            Self::Tuple(v) => v.hash(state),
            Self::None => {}
            Self::Datetime(s) | Self::Date(s) | Self::Str(s) => s.hash(state),
            Self::Bool(b) => b.hash(state),
            Self::Int(i) => i.hash(state),
            // total_cmp distinguishes the bit patterns, so the bits are the hash
            Self::Float(x) => x.to_bits().hash(state),
        }
    }
}

impl fmt::Display for WindowLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(v) => {
                f.write_str("(")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if v.len() == 1 {
                    f.write_str(",")?;
                }
                f.write_str(")")
            }
            Self::None => f.write_str("None"),
            Self::Datetime(s) | Self::Date(s) | Self::Str(s) => write!(f, "{s:?}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(x) => write!(f, "{x:?}"),
        }
    }
}

impl From<&str> for WindowLabel {
    fn from(s: &str) -> Self {
        Self::Str(s.to_string())
    }
}

impl From<i64> for WindowLabel {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_order_is_total_and_floats_use_total_cmp() {
        let mut labels = [
            WindowLabel::Float(f64::NAN),
            WindowLabel::Float(-0.0),
            WindowLabel::Float(0.0),
            WindowLabel::Int(2),
            WindowLabel::from("a"),
            WindowLabel::None,
        ];
        labels.sort();
        assert_eq!(labels[0], WindowLabel::None);
        assert_eq!(labels[4], WindowLabel::Float(0.0));
        assert!(matches!(labels[5], WindowLabel::Float(x) if x.is_nan()));
        assert_eq!(WindowLabel::Float(f64::NAN), WindowLabel::Float(f64::NAN));
    }

    #[test]
    fn labels_render_readably() {
        let t = WindowLabel::Tuple(vec!["subject".into(), 3.into()]);
        assert_eq!(t.to_string(), "(\"subject\", 3)");
    }
}
