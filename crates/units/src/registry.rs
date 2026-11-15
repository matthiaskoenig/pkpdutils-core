//! The curated unit registry and the parser.
//!
//! Covers mass, volume (`length ** 3`), time, amount of substance, activity
//! (`IU`), pressure and the dimensionless `dimensionless`, `count` and `none`.
//! Temperature is deliberately absent.

use std::fmt;
use std::ops::{Div, Mul};

/// Signed exponents of the base dimensions of the registry.
///
/// Pressure is `mass / (length * time ** 2)` and needs no axis of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Dimension {
    /// Exponent of mass.
    pub mass: i8,
    /// Exponent of length.
    pub length: i8,
    /// Exponent of time.
    pub time: i8,
    /// Exponent of amount of substance.
    pub substance: i8,
    /// Exponent of activity (`IU`).
    pub activity: i8,
}

impl Dimension {
    /// The dimensionless dimension.
    pub const NONE: Dimension = Dimension::new(0, 0, 0, 0, 0);
    /// Mass.
    pub const MASS: Dimension = Dimension::new(1, 0, 0, 0, 0);
    /// Length.
    pub const LENGTH: Dimension = Dimension::new(0, 1, 0, 0, 0);
    /// Time.
    pub const TIME: Dimension = Dimension::new(0, 0, 1, 0, 0);
    /// Amount of substance.
    pub const SUBSTANCE: Dimension = Dimension::new(0, 0, 0, 1, 0);
    /// Activity (`IU`).
    pub const ACTIVITY: Dimension = Dimension::new(0, 0, 0, 0, 1);
    /// Volume, `length ** 3`.
    pub const VOLUME: Dimension = Dimension::new(0, 3, 0, 0, 0);
    /// Pressure, `mass / (length * time ** 2)`.
    pub const PRESSURE: Dimension = Dimension::new(1, -1, -2, 0, 0);

    /// A dimension from its five exponents.
    pub const fn new(mass: i8, length: i8, time: i8, substance: i8, activity: i8) -> Self {
        Dimension {
            mass,
            length,
            time,
            substance,
            activity,
        }
    }

    /// The product of two dimensions.
    pub const fn mul(self, o: Dimension) -> Dimension {
        Dimension::new(
            self.mass + o.mass,
            self.length + o.length,
            self.time + o.time,
            self.substance + o.substance,
            self.activity + o.activity,
        )
    }

    /// The quotient of two dimensions.
    pub const fn div(self, o: Dimension) -> Dimension {
        Dimension::new(
            self.mass - o.mass,
            self.length - o.length,
            self.time - o.time,
            self.substance - o.substance,
            self.activity - o.activity,
        )
    }

    /// The dimension raised to an integer power.
    pub const fn pow(self, n: i8) -> Dimension {
        Dimension::new(
            self.mass * n,
            self.length * n,
            self.time * n,
            self.substance * n,
            self.activity * n,
        )
    }

    /// Whether every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        *self == Dimension::NONE
    }
}

impl Mul for Dimension {
    type Output = Dimension;
    fn mul(self, o: Dimension) -> Dimension {
        Dimension::mul(self, o)
    }
}

impl Div for Dimension {
    type Output = Dimension;
    fn div(self, o: Dimension) -> Dimension {
        Dimension::div(self, o)
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let axes = [
            ("mass", self.mass),
            ("length", self.length),
            ("time", self.time),
            ("substance", self.substance),
            ("activity", self.activity),
        ];
        let parts: Vec<String> = axes
            .iter()
            .filter(|(_, e)| *e != 0)
            .map(|(n, e)| format!("[{n}] ** {e}"))
            .collect();
        if parts.is_empty() {
            f.write_str("dimensionless")
        } else {
            f.write_str(&parts.join(" * "))
        }
    }
}

/// An error of parsing or checking a unit.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum UnitError {
    /// The string is empty.
    #[error("'' is not a unit: use 'dimensionless' for a dimensionless quantity")]
    Empty,
    /// A name which is not a unit of the registry.
    #[error(
        "'{name}' is not a unit of the registry in '{unit}'; the supported dimensions are mass, \
         volume, time, amount of substance, activity (IU), pressure (mmHg) and dimensionless \
         (dimensionless, count, none)"
    )]
    Unknown {
        /// The unknown name.
        name: String,
        /// The whole unit string.
        unit: String,
    },
    /// The string is not a unit expression.
    #[error("'{unit}' is not a unit expression: {reason}")]
    Syntax {
        /// The whole unit string.
        unit: String,
        /// What is wrong with it.
        reason: String,
    },
    /// A unit which is no dose unit.
    #[error(
        "a dose must be a mass, an amount of substance or an activity, or one of them per body \
         weight, not '{dimension}' ('{unit}')"
    )]
    NotADose {
        /// The unit string.
        unit: String,
        /// Its dimension.
        dimension: String,
    },
}

/// One named unit of a [`Unit`] with its exponent.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    /// The long name, `milligram`.
    pub name: String,
    /// The short symbol, `mg`.
    pub symbol: String,
    /// The dimension of the named unit.
    pub dimension: Dimension,
    /// The factor of the named unit to the base units of the registry.
    pub factor: f64,
    /// The (nonzero) exponent.
    pub exponent: i32,
}

/// A unit: named units with integer exponents, sorted by long name.
///
/// Terms cancel by identity, so `milligram` and `gram` stay distinct. The
/// dimensionless unit has no terms; `none` and `count` are terms of their own
/// which happen to have no dimension, so `none` differs from `dimensionless`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Unit {
    terms: Vec<Term>,
}

impl Unit {
    /// The unit without terms, `dimensionless`.
    pub fn dimensionless() -> Unit {
        Unit::default()
    }

    /// The terms, sorted by long name.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// The dimension of the unit.
    pub fn dimension(&self) -> Dimension {
        self.terms.iter().fold(Dimension::NONE, |d, t| {
            d * t.dimension.pow(i8::try_from(t.exponent).unwrap_or(0))
        })
    }

    /// The factor of the unit to the base units of the registry.
    pub fn factor(&self) -> f64 {
        self.terms
            .iter()
            .map(|t| t.factor.powi(t.exponent))
            .product()
    }

    /// The product with another unit.
    pub fn times(&self, other: &Unit) -> Unit {
        self.combine(other, 1)
    }

    /// The quotient by another unit.
    pub fn over(&self, other: &Unit) -> Unit {
        self.combine(other, -1)
    }

    /// The unit raised to an integer power.
    pub fn pow(&self, n: i32) -> Unit {
        if n == 0 {
            return Unit::dimensionless();
        }
        let terms = self
            .terms
            .iter()
            .map(|t| Term {
                exponent: t.exponent * n,
                ..t.clone()
            })
            .collect();
        Unit { terms }
    }

    fn combine(&self, other: &Unit, sign: i32) -> Unit {
        let mut terms = self.terms.clone();
        for o in &other.terms {
            match terms.iter_mut().find(|t| t.name == o.name) {
                Some(t) => t.exponent += sign * o.exponent,
                None => terms.push(Term {
                    exponent: sign * o.exponent,
                    ..o.clone()
                }),
            }
        }
        terms.retain(|t| t.exponent != 0);
        terms.sort_by(|a, b| a.name.cmp(&b.name));
        Unit { terms }
    }
}

/// A base unit of the registry.
struct Base {
    name: &'static str,
    symbols: &'static [&'static str],
    dimension: Dimension,
    factor: f64,
    prefixable: bool,
}

const fn base(
    name: &'static str,
    symbols: &'static [&'static str],
    dimension: Dimension,
    factor: f64,
    prefixable: bool,
) -> Base {
    Base {
        name,
        symbols,
        dimension,
        factor,
        prefixable,
    }
}

/// The named units. The first symbol is the one the short form writes.
const BASES: [Base; 11] = [
    base("gram", &["g"], Dimension::MASS, 1e-3, true),
    base("liter", &["l", "L"], Dimension::VOLUME, 1e-3, true),
    base("second", &["s", "sec"], Dimension::TIME, 1.0, true),
    base("minute", &["min"], Dimension::TIME, 60.0, false),
    base("hour", &["h", "hr"], Dimension::TIME, 3600.0, false),
    base("day", &["d"], Dimension::TIME, 86400.0, false),
    base("mole", &["mol"], Dimension::SUBSTANCE, 1.0, true),
    base("IU", &["IU"], Dimension::ACTIVITY, 1.0, true),
    base(
        "millimeter_Hg",
        &["mmHg"],
        Dimension::PRESSURE,
        133.322_387_415,
        false,
    ),
    base("count", &["count"], Dimension::NONE, 1.0, false),
    base("none", &["none"], Dimension::NONE, 1.0, false),
];

/// An SI prefix. Prefixes apply only to the units flagged `prefixable` (gram,
/// liter, second, mole, IU): nothing in the package spells a prefixed hour, and
/// rejecting `kilohour` is deliberate, a curated registry is not a pint clone.
///
/// A prefix is: the long name, the spellings accepted as a symbol (the first
/// is written) and the power of ten.
struct Prefix {
    name: &'static str,
    symbols: &'static [&'static str],
    exponent: i32,
}

const PREFIXES: [Prefix; 16] = [
    Prefix {
        name: "pico",
        symbols: &["p"],
        exponent: -12,
    },
    Prefix {
        name: "nano",
        symbols: &["n"],
        exponent: -9,
    },
    Prefix {
        name: "micro",
        symbols: &["\u{3bc}", "\u{b5}", "u"],
        exponent: -6,
    },
    Prefix {
        name: "milli",
        symbols: &["m"],
        exponent: -3,
    },
    Prefix {
        name: "centi",
        symbols: &["c"],
        exponent: -2,
    },
    Prefix {
        name: "deci",
        symbols: &["d"],
        exponent: -1,
    },
    Prefix {
        name: "kilo",
        symbols: &["k"],
        exponent: 3,
    },
    Prefix {
        name: "mega",
        symbols: &["M"],
        exponent: 6,
    },
    Prefix {
        name: "giga",
        symbols: &["G"],
        exponent: 9,
    },
    Prefix {
        name: "tera",
        symbols: &["T"],
        exponent: 12,
    },
    Prefix {
        name: "femto",
        symbols: &["f"],
        exponent: -15,
    },
    Prefix {
        name: "atto",
        symbols: &["a"],
        exponent: -18,
    },
    Prefix {
        name: "peta",
        symbols: &["P"],
        exponent: 15,
    },
    Prefix {
        name: "exa",
        symbols: &["E"],
        exponent: 18,
    },
    Prefix {
        name: "zepto",
        symbols: &["z"],
        exponent: -21,
    },
    Prefix {
        name: "yocto",
        symbols: &["y"],
        exponent: -24,
    },
];

fn term_of(b: &Base) -> Term {
    Term {
        name: b.name.to_string(),
        symbol: b.symbols[0].to_string(),
        dimension: b.dimension,
        factor: b.factor,
        exponent: 1,
    }
}

fn prefixed(p: &Prefix, b: &Base) -> Term {
    Term {
        name: format!("{}{}", p.name, b.name),
        symbol: format!("{}{}", p.symbols[0], b.symbols[0]),
        dimension: b.dimension,
        factor: b.factor * 10f64.powi(p.exponent),
        exponent: 1,
    }
}

/// The named unit of a token, with exponent 1.
fn lookup(token: &str) -> Option<Term> {
    if let Some(b) = BASES
        .iter()
        .find(|b| b.name == token || b.symbols.contains(&token))
    {
        return Some(term_of(b));
    }
    for p in &PREFIXES {
        if let Some(rest) = token.strip_prefix(p.name)
            && let Some(b) = BASES.iter().find(|b| b.prefixable && b.name == rest)
        {
            return Some(prefixed(p, b));
        }
        for s in p.symbols {
            if let Some(rest) = token.strip_prefix(s)
                && let Some(b) = BASES
                    .iter()
                    .find(|b| b.prefixable && b.symbols.contains(&rest))
            {
                return Some(prefixed(p, b));
            }
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(String),
    Name(String),
    Star,
    Slash,
    Pow,
    Open,
    Close,
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '*' && chars.get(i + 1) == Some(&'*') {
            tokens.push(Token::Pow);
            i += 2;
        } else if c == '*' {
            tokens.push(Token::Star);
            i += 1;
        } else if c == '^' {
            tokens.push(Token::Pow);
            i += 1;
        } else if c == '/' {
            tokens.push(Token::Slash);
            i += 1;
        } else if c == '(' {
            tokens.push(Token::Open);
            i += 1;
        } else if c == ')' {
            tokens.push(Token::Close);
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            tokens.push(Token::Number(chars[start..i].iter().collect()));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(Token::Name(chars[start..i].iter().collect()));
        } else if c == '-' || c == '+' {
            tokens.push(Token::Number(c.to_string()));
            i += 1;
        } else {
            return Err(format!("unexpected character '{c}'"));
        }
    }
    Ok(tokens)
}

/// Failure of the parser: a syntax problem or an unknown name.
enum Failure {
    Syntax(String),
    Unknown(String),
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// `expr = term (('*' | '/') term)*`, left-associative.
    fn expr(&mut self) -> Result<Unit, Failure> {
        let mut left = self.term()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.pos += 1;
                    left = left.times(&self.term()?);
                }
                Some(Token::Slash) => {
                    self.pos += 1;
                    left = left.over(&self.term()?);
                }
                _ => return Ok(left),
            }
        }
    }

    /// `term = atom ('**' integer)?`
    fn term(&mut self) -> Result<Unit, Failure> {
        let atom = self.atom()?;
        if self.peek() == Some(&Token::Pow) {
            self.pos += 1;
            let n = self.integer()?;
            return Ok(atom.pow(n));
        }
        Ok(atom)
    }

    /// An optionally signed integer exponent.
    fn integer(&mut self) -> Result<i32, Failure> {
        let mut sign = 1;
        while let Some(Token::Number(t)) = self.peek() {
            match t.as_str() {
                "-" => sign = -sign,
                "+" => {}
                digits => {
                    let n: i32 = digits
                        .parse()
                        .map_err(|_| Failure::Syntax(format!("bad exponent '{digits}'")))?;
                    self.pos += 1;
                    return Ok(sign * n);
                }
            }
            self.pos += 1;
        }
        Err(Failure::Syntax(
            "an integer exponent is expected".to_string(),
        ))
    }

    /// `atom = name | '1' | '(' expr ')'`
    fn atom(&mut self) -> Result<Unit, Failure> {
        let token = self.peek().cloned();
        match token {
            Some(Token::Name(name)) => {
                self.pos += 1;
                if name == "dimensionless" {
                    return Ok(Unit::dimensionless());
                }
                match lookup(&name) {
                    Some(term) => Ok(Unit { terms: vec![term] }),
                    None => Err(Failure::Unknown(name)),
                }
            }
            Some(Token::Number(n)) if n == "1" => {
                self.pos += 1;
                Ok(Unit::dimensionless())
            }
            Some(Token::Number(n)) => Err(Failure::Syntax(format!(
                "only a leading 1 is a number of a unit, not '{n}'"
            ))),
            Some(Token::Open) => {
                self.pos += 1;
                let inner = self.expr()?;
                if self.peek() == Some(&Token::Close) {
                    self.pos += 1;
                    Ok(inner)
                } else {
                    Err(Failure::Syntax(
                        "a closing parenthesis is missing".to_string(),
                    ))
                }
            }
            Some(t) => Err(Failure::Syntax(format!("unexpected {t:?}"))),
            None => Err(Failure::Syntax("the expression ends too early".to_string())),
        }
    }
}

/// Parse a unit string: products with `*`, quotients with `/` (left-associative,
/// so `mg/l/hr` is `mg / (l * hr)`), integer powers with `**`, parentheses and
/// a leading `1`.
///
/// # Errors
///
/// [`UnitError`] if the string is empty, not an expression or names a unit
/// which is not in the registry.
pub fn parse_unit(s: &str) -> Result<Unit, UnitError> {
    if s.trim().is_empty() {
        return Err(UnitError::Empty);
    }
    let syntax = |reason: String| UnitError::Syntax {
        unit: s.to_string(),
        reason,
    };
    let tokens = tokenize(s).map_err(syntax)?;
    let mut parser = Parser { tokens, pos: 0 };
    let result = parser.expr().and_then(|u| {
        if parser.pos == parser.tokens.len() {
            Ok(u)
        } else {
            Err(Failure::Syntax(format!(
                "unexpected {:?}",
                parser.tokens[parser.pos]
            )))
        }
    });
    match result {
        Ok(u) => Ok(u),
        Err(Failure::Syntax(reason)) => Err(syntax(reason)),
        Err(Failure::Unknown(name)) => Err(UnitError::Unknown {
            name,
            unit: s.to_string(),
        }),
    }
}
