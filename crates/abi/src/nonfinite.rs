//! A number which may be NaN or infinite, as JSON: a finite number is a number, NaN and the
//! infinities are the strings `"NaN"`, `"inf"` and `"-inf"` (`serde_json` would write `null`,
//! which reads back as no number).
//!
//! Use the module itself as `#[serde(with = "nonfinite")]` for an `f64`, and the submodules
//! [`option`], [`vec`], [`pair`] and [`pair_map`] for the other shapes.
//! An `Option` field also needs `#[serde(default)]` so that a missing key reads as `None`.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Write one number.
pub fn serialize<S: Serializer>(x: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    match *x {
        x if x.is_finite() => serializer.serialize_f64(x),
        x if x.is_nan() => serializer.serialize_str("NaN"),
        x if x > 0.0 => serializer.serialize_str("inf"),
        _ => serializer.serialize_str("-inf"),
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Number {
    Finite(f64),
    Text(String),
}

/// Read one number, a JSON number or one of the three strings.
pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let text = match Number::deserialize(deserializer)? {
        Number::Finite(x) => return Ok(x),
        Number::Text(text) => text,
    };
    match text.as_str() {
        "NaN" => Ok(f64::NAN),
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        other => Err(D::Error::custom(format!(
            "{other:?} is no number (a number, \"NaN\", \"inf\" or \"-inf\")"
        ))),
    }
}

/// One number as a value of the representation, the building block of the container modules.
#[derive(Serialize, Deserialize)]
struct Value(#[serde(with = "self")] f64);

/// A pair of numbers as a two element array.
#[derive(Serialize, Deserialize)]
struct Pair(#[serde(with = "self")] f64, #[serde(with = "self")] f64);

/// `Option<f64>`, `None` is `null`.
pub mod option {
    use super::*;

    /// Write an optional number.
    pub fn serialize<S: Serializer>(x: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        x.map(Value).serialize(serializer)
    }

    /// Read an optional number.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        Ok(Option::<Value>::deserialize(deserializer)?.map(|v| v.0))
    }
}

/// `Vec<f64>`.
pub mod vec {
    use super::*;

    /// Write a vector of numbers.
    pub fn serialize<S: Serializer>(x: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(x.iter().copied().map(Value))
    }

    /// Read a vector of numbers.
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<Value>::deserialize(deserializer)?
            .into_iter()
            .map(|v| v.0)
            .collect())
    }
}

/// `(f64, f64)`.
pub mod pair {
    use super::*;

    /// Write a pair of numbers.
    pub fn serialize<S: Serializer>(x: &(f64, f64), serializer: S) -> Result<S::Ok, S::Error> {
        Pair(x.0, x.1).serialize(serializer)
    }

    /// Read a pair of numbers.
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<(f64, f64), D::Error> {
        let Pair(a, b) = Pair::deserialize(deserializer)?;
        Ok((a, b))
    }
}

/// `BTreeMap<String, (f64, f64)>`.
pub mod pair_map {
    use super::*;
    use std::collections::BTreeMap;

    /// Write a map of pairs.
    pub fn serialize<S: Serializer>(
        x: &BTreeMap<String, (f64, f64)>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_map(x.iter().map(|(k, v)| (k, Pair(v.0, v.1))))
    }

    /// Read a map of pairs.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<String, (f64, f64)>, D::Error> {
        Ok(BTreeMap::<String, Pair>::deserialize(deserializer)?
            .into_iter()
            .map(|(k, Pair(a, b))| (k, (a, b)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        #[serde(with = "super::vec")]
        v: Vec<f64>,
    }

    #[test]
    fn non_finite_values_round_trip_as_strings() {
        let v = vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0];
        let json = serde_json::to_string(&Wrapper { v: v.clone() }).unwrap();
        assert!(json.contains("\"NaN\"") && json.contains("\"inf\"") && json.contains("\"-inf\""));
        let back: Wrapper = serde_json::from_str(&json).unwrap();
        assert!(back.v[1].is_nan());
        assert_eq!(back.v[2], f64::INFINITY);
        assert_eq!(back.v[3], f64::NEG_INFINITY);
        assert!(back.v[4].is_sign_negative());
    }

    #[test]
    fn finite_values_keep_full_precision() {
        let v = vec![0.1 + 0.2, 1e-308, 1.7976931348623157e308];
        let back: Wrapper =
            serde_json::from_str(&serde_json::to_string(&Wrapper { v: v.clone() }).unwrap())
                .unwrap();
        assert_eq!(back.v, v);
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Shapes {
        #[serde(with = "super")]
        x: f64,
        #[serde(with = "super::option", default)]
        o: Option<f64>,
        #[serde(with = "super::pair")]
        p: (f64, f64),
        #[serde(with = "super::pair_map")]
        m: std::collections::BTreeMap<String, (f64, f64)>,
    }

    #[test]
    fn option_pair_and_map_shapes_round_trip() {
        let m = [("a".to_string(), (0.0, f64::INFINITY))]
            .into_iter()
            .collect();
        let s = Shapes {
            x: f64::INFINITY,
            o: None,
            p: (1.0, f64::NEG_INFINITY),
            m,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Shapes>(&json).unwrap(), s);
        let missing: Shapes = serde_json::from_str(r#"{"x":1.0,"p":[1,2],"m":{}}"#).unwrap();
        assert_eq!(missing.o, None);
        assert!(serde_json::from_str::<Shapes>(r#"{"x":"abc","p":[1,2],"m":{}}"#).is_err());
    }
}
