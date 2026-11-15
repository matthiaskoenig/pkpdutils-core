use rayon::prelude::*;
use std::time::Instant;

const NOUT: usize = 8;
// cmax, tmax, clast, tlast, auc_last, lambda_z, thalf, auc_inf_obs

fn nca_row(t: &[f64], c: &[f64], out: &mut [f64]) {
    let n = t.len();
    let mut imax = 0;
    for i in 1..n {
        if c[i] > c[imax] {
            imax = i;
        }
    }
    let last = n - 1;
    let mut auc = 0.0;
    for i in 0..last {
        let (c1, c2, dt) = (c[i], c[i + 1], t[i + 1] - t[i]);
        auc += if c2 < c1 && c1 > 0.0 && c2 > 0.0 && c1 != c2 {
            (c1 - c2) / (c1 / c2).ln() * dt
        } else {
            0.5 * (c1 + c2) * dt
        };
    }
    // terminal: windows s..=last, s > imax, >= 3 points, best adjusted R2
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    let mut best = (f64::NEG_INFINITY, f64::NAN);
    let (xo, yo) = (t[last], c[last].ln());
    let mut m = 0.0;
    let mut s = last as isize;
    while s > imax as isize {
        let i = s as usize;
        if c[i] > 0.0 {
            let x = t[i] - xo;
            let y = c[i].ln() - yo;
            m += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
            if m >= 3.0 {
                let sxxc = sxx - sx * sx / m;
                let sxyc = sxy - sx * sy / m;
                let syyc = syy - sy * sy / m;
                if sxxc > 0.0 && syyc > 0.0 {
                    let slope = sxyc / sxxc;
                    let ssres = syyc - slope * sxyc;
                    let r2 = (1.0 - ssres / syyc).clamp(0.0, 1.0);
                    let adj = 1.0 - (1.0 - r2) * (m - 1.0) / (m - 2.0);
                    if adj > best.0 {
                        best = (adj, slope);
                    }
                }
            }
        }
        s -= 1;
    }
    let lz = -best.1;
    out[0] = c[imax];
    out[1] = t[imax];
    out[2] = c[last];
    out[3] = t[last];
    out[4] = auc;
    out[5] = lz;
    out[6] = std::f64::consts::LN_2 / lz;
    out[7] = auc + c[last] / lz;
}

fn run_serial(t: &[f64], v: &[f64], nt: usize, out: &mut [f64]) {
    for (r, o) in v.chunks(nt).zip(out.chunks_mut(NOUT)) {
        nca_row(t, r, o);
    }
}

fn run_rayon(t: &[f64], v: &[f64], nt: usize, out: &mut [f64], rows_per_chunk: usize) {
    v.par_chunks(nt * rows_per_chunk)
        .zip(out.par_chunks_mut(NOUT * rows_per_chunk))
        .for_each(|(vc, oc)| run_serial(t, vc, nt, oc));
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.next().max(1e-300), self.next());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

fn gen(rows: usize, nt: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let t: Vec<f64> = (0..nt).map(|i| 0.25 + (24.0 - 0.25) * i as f64 / (nt.max(2) - 1) as f64).collect();
    let mut v = Vec::with_capacity(rows * nt);
    for _ in 0..rows {
        let vol = 20.0 + 40.0 * rng.next();
        let k = 0.05 + 0.25 * rng.next();
        for &ti in &t {
            v.push(100.0 / vol * (-k * ti).exp() * (0.05 * rng.normal()).exp());
        }
    }
    (t, v)
}

fn hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

fn check() {
    // monoexponential, dense
    let (c0, k) = (10.0, 0.2);
    let t: Vec<f64> = (0..=4800).map(|i| i as f64 * 0.01).collect();
    let c: Vec<f64> = t.iter().map(|x| c0 * (-k * x).exp()).collect();
    let mut o = [0.0; NOUT];
    nca_row(&t, &c, &mut o);
    println!("mono: lambda_z={:.9} (k={}) thalf={:.9} (ln2/k={:.9}) auc_inf_obs={:.6} (c0/k={})", o[5], k, o[6], std::f64::consts::LN_2 / k, o[7], c0 / k);
    assert!((o[5] - k).abs() / k < 1e-6);
    assert!((o[6] - std::f64::consts::LN_2 / k).abs() < 1e-5);
    assert!((o[7] - c0 / k).abs() / (c0 / k) < 1e-3);
    // hand-checked linear trapezoid: (0,0),(1,4),(3,2): 0.5*4*1 + 0.5*(4+2)*2 -> log-down gives log; use rising+flat check
    // points (0,2),(1,4),(3,4): up-segment 3.0, flat segment 8.0 -> 11.0
    let mut o = [0.0; NOUT];
    nca_row(&[0.0, 1.0, 3.0], &[2.0, 4.0, 4.0], &mut o);
    println!("trapezoid: auc_last={} (expected 11)", o[4]);
    assert_eq!(o[4], 11.0);
    // c1 > c2 with log-down: (0,4),(2,2): (4-2)/ln2*2
    let mut o = [0.0; NOUT];
    nca_row(&[0.0, 2.0, 4.0], &[4.0, 2.0, 2.0], &mut o);
    let exp = 2.0 / std::f64::consts::LN_2 * 2.0 + 4.0;
    println!("log-down: auc_last={} (expected {})", o[4], exp);
    assert!((o[4] - exp).abs() < 1e-12);
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("check") {
        check();
        return;
    }
    let shapes = [(1usize, 12usize), (1000, 200), (10000, 200), (100000, 12)];
    let (rows, nt) = shapes[args[1].parse::<usize>().unwrap()];
    let mode = args[2].as_str();
    let (t, v) = gen(rows, nt);
    let mut out = vec![0.0; rows * NOUT];
    let chunk = (rows / (rayon::current_num_threads() * 8)).max(1);
    let mut f = |out: &mut Vec<f64>| match mode {
        "serial" => run_serial(&t, &v, nt, out),
        _ => run_rayon(&t, &v, nt, out, chunk),
    };
    let mut rounds = vec![];
    for _ in 0..3 {
        for _ in 0..3 {
            f(&mut out);
        }
        let mut ts = vec![];
        for _ in 0..20 {
            let s = Instant::now();
            f(&mut out);
            ts.push(s.elapsed().as_secs_f64() * 1e3);
        }
        rounds.push(median(ts));
    }
    let sum: f64 = out.iter().filter(|x| x.is_finite()).sum();
    println!("{rows}x{nt} {mode} threads={} medians_ms={:?} median={:.4} us_per_row={:.3} hwm_kb={} chk={:.3}", if mode=="serial" {1} else {rayon::current_num_threads()}, rounds.iter().map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>(), median(rounds.clone()), median(rounds.clone())*1e3/rows as f64, hwm_kb(), sum);
}
