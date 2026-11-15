//! The curated unit registry and the parser.
//!
//! Covers mass, volume (`length ** 3`), time, amount of substance, activity
//! (`IU`), pressure and the dimensionless `dimensionless`, `count` and `none`.
//! Temperature is deliberately absent.
//!
//! Of the units it does hold it accepts every spelling pint does, because a host
//! string comes from code written against pint: a prefix name or symbol with a
//! unit name, symbol or alias, and one trailing plural `s`, see [`lookup`].
//! What it deliberately does not mirror is pint beyond those units: a prefix
//! outside [`PREFIXES`] (no `hectoliter`), a prefixed unit which is not
//! `prefixable` (no `kilohour`), pint's implicit product (`mg l`) and its
//! fractional exponents (`hr ** 0.5`), all of which are an error here.

use std::fmt;
use std::ops::{Div, Mul};

/// Signed exponents of the base dimensions of the registry.
///
/// Pressure is `mass / (length * time ** 2)` and needs no axis of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Dimension {
    /// Exponent of mass.
    pub mass: i32,
    /// Exponent of length.
    pub length: i32,
    /// Exponent of time.
    pub time: i32,
    /// Exponent of amount of substance.
    pub substance: i32,
    /// Exponent of activity (`IU`).
    pub activity: i32,
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
    pub const fn new(mass: i32, length: i32, time: i32, substance: i32, activity: i32) -> Self {
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
    pub const fn pow(self, n: i32) -> Dimension {
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

/// The longest a host string is echoed into a [`UnitError`].
///
/// A unit string comes from outside the crate and may be arbitrarily long, and
/// an error message bound for a javascript console or a python traceback must
/// not carry a multi-megabyte input whole. Every variant of [`UnitError`] holds
/// its input already clipped, so a message is bounded wherever it is formatted.
pub const MAX_ERROR_TEXT: usize = 40;

/// `s` for an error message: its first [`MAX_ERROR_TEXT`] characters, with a
/// trailing `...` when it was cut.
pub(crate) fn clipped(s: &str) -> String {
    let mut out: String = s.chars().take(MAX_ERROR_TEXT).collect();
    if s.chars().nth(MAX_ERROR_TEXT).is_some() {
        out.push_str("...");
    }
    out
}

/// An error of parsing or checking a unit.
///
/// Every string field is clipped to [`MAX_ERROR_TEXT`] characters.
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
        /// The unit string.
        unit: String,
    },
    /// The string is not a unit expression.
    #[error("'{unit}' is not a unit expression: {reason}")]
    Syntax {
        /// The unit string.
        unit: String,
        /// What is wrong with it.
        reason: String,
    },
    /// An exponent beyond the supported range.
    #[error("an exponent of '{unit}' is beyond the supported range of +-{max}")]
    Exponent {
        /// The unit string.
        unit: String,
        /// The largest absolute exponent.
        max: i32,
    },
    /// Parentheses nested too deeply.
    #[error("'{unit}' nests parentheses deeper than {max} levels")]
    Nesting {
        /// The unit string.
        unit: String,
        /// The deepest supported nesting.
        max: usize,
    },
    /// A placeholder of a unit expression which the context does not provide.
    #[error("'{expression}' needs a {placeholder} unit")]
    MissingPlaceholder {
        /// The unit expression.
        expression: String,
        /// The name of the placeholder, without braces.
        placeholder: String,
    },
    /// A placeholder which is none of `unit`, `time`, `dose`, `amount`, `volume`.
    #[error("'{expression}' uses the unknown placeholder '{placeholder}'")]
    UnknownPlaceholder {
        /// The unit expression.
        expression: String,
        /// The text between the braces.
        placeholder: String,
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
        self.terms
            .iter()
            .fold(Dimension::NONE, |d, t| d * t.dimension.pow(t.exponent))
    }

    /// The factor of the unit to the base units of the registry.
    pub fn factor(&self) -> f64 {
        self.terms
            .iter()
            .map(|t| t.factor.powi(t.exponent))
            .product()
    }

    /// The product with another unit, `None` if an exponent leaves
    /// [`MAX_EXPONENT`].
    pub fn times(&self, other: &Unit) -> Option<Unit> {
        self.combine(other, 1)
    }

    /// The quotient by another unit, `None` if an exponent leaves
    /// [`MAX_EXPONENT`].
    pub fn over(&self, other: &Unit) -> Option<Unit> {
        self.combine(other, -1)
    }

    /// The unit raised to an integer power, `None` if an exponent leaves
    /// [`MAX_EXPONENT`].
    pub fn pow(&self, n: i32) -> Option<Unit> {
        if n == 0 {
            return Some(Unit::dimensionless());
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            terms.push(Term {
                exponent: bounded(t.exponent.checked_mul(n)?)?,
                ..t.clone()
            });
        }
        Some(Unit { terms })
    }

    fn combine(&self, other: &Unit, sign: i32) -> Option<Unit> {
        let mut terms = self.terms.clone();
        for o in &other.terms {
            let delta = sign.checked_mul(o.exponent)?;
            match terms.iter_mut().find(|t| t.name == o.name) {
                Some(t) => t.exponent = bounded(t.exponent.checked_add(delta)?)?,
                None => terms.push(Term {
                    exponent: delta,
                    ..o.clone()
                }),
            }
        }
        terms.retain(|t| t.exponent != 0);
        terms.sort_by(|a, b| a.name.cmp(&b.name));
        Some(Unit { terms })
    }
}

/// The largest absolute exponent a unit may carry. A host string must never
/// overflow the arithmetic, so a larger exponent is an error.
pub const MAX_EXPONENT: i32 = 64;

/// The deepest nesting of parentheses the parser follows, which keeps its
/// recursion far from the stack limit.
pub const MAX_DEPTH: usize = 64;

/// The exponent itself when it is within [`MAX_EXPONENT`].
fn bounded(e: i32) -> Option<i32> {
    (e.abs() <= MAX_EXPONENT).then_some(e)
}

/// A base unit of the registry.
struct Base {
    name: &'static str,
    /// Every spelling besides the name which pint accepts for this unit: its
    /// symbol first, which is the one the short form writes, then its aliases.
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

/// The named units. The first symbol is the one the short form writes, the rest
/// are the other spellings pint accepts (`L`, `litre` and `ℓ` for `liter`).
const BASES: [Base; 12] = [
    base("gram", &["g"], Dimension::MASS, 1e-3, true),
    base(
        "liter",
        &["l", "L", "litre", "\u{2113}"],
        Dimension::VOLUME,
        1e-3,
        true,
    ),
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
    base("percent", &["%"], Dimension::NONE, 0.01, false),
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
        // `mc` is the CDISC PKUNIT synonym of `ug` and the usual US clinical
        // spelling, `mu` the ASCII transliteration of the Greek letter
        symbols: &["\u{3bc}", "\u{b5}", "u", "mu", "mc"],
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

/// The spellings of a prefix: its name, then its symbols.
fn prefix_spellings(p: &Prefix) -> impl Iterator<Item = &'static str> {
    std::iter::once(p.name).chain(p.symbols.iter().copied())
}

/// The base unit a spelling names, `None` when it is shorter than `min_chars`.
///
/// pint keeps a unit's name, its symbol and its aliases in one table, so every
/// spelling of a unit stands for it wherever one of them does.
fn base_of(spelling: &str, min_chars: usize) -> Option<&'static Base> {
    if spelling.chars().count() < min_chars {
        return None;
    }
    BASES
        .iter()
        .find(|b| b.name == spelling || b.symbols.contains(&spelling))
}

/// The named unit of a singular token whose base spelling has at least
/// `min_chars` characters: a bare base, or a prefix followed by a prefixable one.
fn resolve(token: &str, min_chars: usize) -> Option<Term> {
    if let Some(b) = base_of(token, min_chars) {
        return Some(term_of(b));
    }
    for p in &PREFIXES {
        for s in prefix_spellings(p) {
            if let Some(rest) = token.strip_prefix(s)
                && let Some(b) = base_of(rest, min_chars).filter(|b| b.prefixable)
            {
                return Some(prefixed(p, b));
            }
        }
    }
    None
}

/// The named unit of a token, with exponent 1.
///
/// Mirrors pint's `parse_unit_name`. pint pairs any spelling of a prefix (its
/// name or one of its symbols) with any spelling of a unit (its name, its symbol
/// or one of its aliases), so all four combinations name the same unit:
/// `millimole`, `mmol`, `millimol` and `mmole`. A single trailing `s` is a plural
/// of any spelling whose base part keeps more than one character, which is why
/// `mols` and `mmoles` are a mole while `ls` and `mgs` are nothing.
///
/// The order is load-bearing and is pint's: it takes the first candidate of the
/// empty suffix before the plural one, and of the empty prefix before a real one.
///
/// - A bare spelling comes before a prefixed one, so `min` is a minute and not a
///   milli-inch, `d` is a day and not a decisomething, and `mmHg` is itself and
///   not a milli-mHg.
/// - The singular comes before the plural, so `ds` is a decisecond, as in pint,
///   and not a plural of the day symbol `d`.
///
/// The registry is a curated subset of pint, so it accepts fewer *units* (no
/// temperature, no prefixed hour, see [`Prefix`]) - but of the units it holds it
/// accepts every spelling pint does, because a host string comes from code
/// written against pint.
fn lookup(token: &str) -> Option<Term> {
    // the whole token, then the token without one trailing plural `s`, whose base
    // part pint requires to keep more than one character
    resolve(token, 1).or_else(|| resolve(token.strip_suffix('s')?, 2))
}

impl Unit {
    /// The unit of one named unit of the registry, by long name or symbol.
    ///
    /// For a name the registry is known to hold (`liter`, `hour`, `kilogram`),
    /// the same term `parse_unit` produces. Crate internal: callers outside
    /// this crate parse strings.
    ///
    /// # Panics
    ///
    /// If the name is not in the registry, a programming error.
    pub(crate) fn named(name: &str) -> Unit {
        let term = lookup(name).unwrap_or_else(|| panic!("'{name}' is not in the registry"));
        Unit { terms: vec![term] }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(String),
    Sign(char),
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
        } else if c == '%' {
            tokens.push(Token::Name("%".to_string()));
            i += 1;
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(Token::Name(chars[start..i].iter().collect()));
        } else if c == '-' || c == '+' {
            tokens.push(Token::Sign(c));
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
    Exponent,
    Nesting,
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
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
                    left = left.times(&self.term()?).ok_or(Failure::Exponent)?;
                }
                Some(Token::Slash) => {
                    self.pos += 1;
                    left = left.over(&self.term()?).ok_or(Failure::Exponent)?;
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
            return atom.pow(n).ok_or(Failure::Exponent);
        }
        Ok(atom)
    }

    /// An integer exponent with at most one sign, within [`MAX_EXPONENT`].
    fn integer(&mut self) -> Result<i32, Failure> {
        let mut sign = 1;
        if let Some(Token::Sign(c)) = self.peek() {
            if *c == '-' {
                sign = -1;
            }
            self.pos += 1;
        }
        match self.peek() {
            Some(Token::Number(digits)) => {
                let n: i32 = digits.parse().map_err(|_| Failure::Exponent)?;
                self.pos += 1;
                bounded(sign * n).ok_or(Failure::Exponent)
            }
            _ => Err(Failure::Syntax(
                "an integer exponent is expected".to_string(),
            )),
        }
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
                "only a leading 1 is a number of a unit, not '{}'",
                clipped(&n)
            ))),
            Some(Token::Open) => {
                self.pos += 1;
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return Err(Failure::Nesting);
                }
                let inner = self.expr()?;
                self.depth -= 1;
                if self.peek() == Some(&Token::Close) {
                    self.pos += 1;
                    Ok(inner)
                } else {
                    Err(Failure::Syntax(
                        "a closing parenthesis is missing".to_string(),
                    ))
                }
            }
            Some(t) => Err(Failure::Syntax(format!(
                "unexpected {}",
                clipped(&format!("{t:?}"))
            ))),
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
        unit: clipped(s),
        reason,
    };
    let tokens = tokenize(s).map_err(syntax)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let result = parser.expr().and_then(|u| {
        if parser.pos == parser.tokens.len() {
            Ok(u)
        } else {
            Err(Failure::Syntax(format!(
                "unexpected {}",
                clipped(&format!("{:?}", parser.tokens[parser.pos]))
            )))
        }
    });
    match result {
        Ok(u) => Ok(u),
        Err(Failure::Syntax(reason)) => Err(syntax(reason)),
        Err(Failure::Exponent) => Err(UnitError::Exponent {
            unit: clipped(s),
            max: MAX_EXPONENT,
        }),
        Err(Failure::Nesting) => Err(UnitError::Nesting {
            unit: clipped(s),
            max: MAX_DEPTH,
        }),
        Err(Failure::Unknown(name)) => Err(UnitError::Unknown {
            name: clipped(&name),
            unit: clipped(s),
        }),
    }
}
