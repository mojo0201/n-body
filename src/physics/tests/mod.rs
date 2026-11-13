//! Physics tests.
//!
//! Loaded as `physics::tests` via `#[cfg(test)] mod tests;` in
//! `physics/mod.rs`; each file below uses `use super::super::*;` and sees every private item (`Tree`,
//! `collision_pass`, `find_collisions`, `pair_hit`, `interact_frac`,
//! `impact_offset`, `dt_limit`, `rung_for`, `disrupt`, `coalesce_debris`,
//! `belt_spec_for`, `spawn_belts`, `belt_mass_earth`, `kirkwood_gaps`,
//! `radius_from_density`, the IC samplers, and so on) without widening the
//! crate's public API.
//!
//! Test naming convention: a letter prefix groups tests by area; a
//! descriptive snake_case name follows. `cargo test -- --list` shows the
//! grouping at a glance.
//!
//!   a  conservation invariants (L, p, mass) across merges
//!   b  hit-and-run
//!   c  debris / moon-forming impacts
//!   d  grazing vs head-on
//!   e  tidal disruption timing
//!   f  radius laws
//!   g  initial conditions (statistical)
//!   h  rubble coalescence
//!   i  radius / survivor / primary / Chandrasekhar
//!   j  compaction invariants
//!   k  binaries and circumbinary stability
//!   l  Barnes-Hut vs direct
//!   m  collision detection (hash vs brute force, immunity)
//!   n  integrator (rungs, dt limits, energy conservation)
//!   o  orbit_info, circularize, host_distance
//!   p  escape culling
//!   q  energy diagnostics
//!   r  kind transitions via merge
//!   s  user edits
//!   t  multi-system spawn
//!   u  RNG / samplers
//!   v  geometry helpers (Hill, Roche, impact, interact_frac, dt_limit)
//!   w  naming helpers (class/threshold/format)
//!   x  star luminosity, temperature, lifetime, habitability
//!   y  samplers (Rayleigh, power-law, normal, IMF median)
//!   z  small utilities (time unit, index_of_id, log ring buffer, cell key)
//!   aa spawn variants
//!   ab fast (hot) collision path matches exhaustive scan
//!   ac rubble bounce (momentum, spin, immunity)
//!   ad fragmentation (frag_masses, density, rigidity, disrupt)
//!   ae integrator internals (rungs, flashes, Yoshida beats leapfrog)
//!   af silent merge (no log, no flash, conservation)
//!   ag orbit / anomaly helpers
//!   ah osculating-ellipse geometry
//!   ai accuracy (tight BH error, tight energy drift, element preservation)
//!   aj realism (Solar System reproduces real periods)
//!   ak fuzz smoke (invariants across many seeds; tree-rebuild stress)
//!   al stellar evolution (aging, remnants, adiabatic expansion, belts,
//!      bulk density, random spawn)
//!   am force dispatch (direct sum, symmetric pair sum, chunked parallel,
//!      tree integration)
//!
//! Belts and bulk density are tested under `al_*` (they were introduced by
//! the evolution-then-belts prompt pair) and under `mean_anomaly_*`,
//! `ices_are_lighter_than_rocks`, `belt_*`, `kirkwood_*`, `trojans_*`,
//! `random_systems_never_exceed_body_count`, `default_density_*`.
//!
//! Files: shared helpers, then one file per area (`cargo test --lib
//! physics::tests::collisions` runs one area).

mod collisions;
mod conservation;
mod debris;
mod dust;
mod evolution;
mod forces;
mod fuzz;
mod kepler;
mod orbits;
mod presets;
mod shared;
mod spawn;
mod util;

#[cfg(test)]
mod manifest_guard {
    //! Verifies the `mod.rs` manifest against the directory and against the
    //! `//!   xx  description` block in the module doc comment.
    //!
    //! Run with `cargo test --lib physics::tests::manifest_guard`.
    //!
    //! ## What is checked
    //!
    //! 1. File ↔ declaration symmetry: every `.rs` file and submodule
    //!    directory is declared, and every declaration resolves to a file
    //!    (`foo.rs`, `foo/mod.rs`, or an explicit `#[path = "…"]`).
    //! 2. Doc-comment accuracy: every documented `xx` prefix has at least
    //!    one `xx_*` test, and every test whose prefix *looks like* a group
    //!    prefix (1–2 lowercase letters) is documented.
    //! 3. Parser self-consistency: the two source parsers are themselves
    //!    tested, so a parser bug cannot make the guards pass by returning
    //!    empty / wrong results.
    //!
    //! ## Design notes
    //!
    //! - The parser tracks **raw strings** (`r"…"`, `r#"…"#`, `r##"…"##`,
    //!   …) so that fixture text embedded in a test body is never treated
    //!   as real code.
    //! - Fixtures used by the parser self-tests are built with `concat!`,
    //!   so their *source* lines begin with `"` rather than with `mod`,
    //!   `fn`, or `#[`, and can never be misread by the same parser they
    //!   test.
    //! - Doc-comment parsing requires exactly three spaces after `//!`
    //!   (matching the aligned `//!   xx  description` format) and at
    //!   least one space between the prefix and its description. That
    //!   accepts both single-letter (`a  …`) and two-letter (`aa …`)
    //!   groups while still rejecting six-space continuation lines.
    //! - Tests with a descriptive prefix longer than `MAX_GROUP_PREFIX_LEN`
    //!   (e.g. `preset_library_loads_all`, `vega_has_no_planets_…`) or
    //!   with a non-lowercase character in the prefix (e.g. `m1_ring_…`)
    //!   are *not* claiming to be part of a group, so they need no
    //!   documentation. Only 1–2-lowercase-letter prefixes are treated as
    //!   group claims and validated against the doc block. This is what
    //!   catches a typo like `ab_foo` when `ab` is not a real group,
    //!   without requiring every descriptive test to be manually listed.
    //! - All doc-drift problems are collected and reported together, so a
    //!   single run shows the complete list of missing / stale entries.
    //!
    //! ## Known limitations
    //!
    //! - Nested block comments (`/* /* x */ y */`) are not tracked. None
    //!   exist in this tree; adding one would surface as an obvious
    //!   guard failure.
    //! - `format!` / `write!` payloads are not tracked. The parser skips
    //!   them by treating `"` as the end of the line's code section.

    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    const INFRASTRUCTURE_FILES: &[&str] = &["mod", "shared"];
    const MAX_GROUP_PREFIX_LEN: usize = 2;

    fn tests_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/physics/tests")
    }

    // ---- Lexer: comment and raw-string aware -----------------------------

    /// Tracks multi-line comment and raw-string state across calls to
    /// `strip`. Returns the code portion of each line (empty if the line
    /// was entirely inside a comment or raw string).
    struct Stripper {
        in_block: bool,
        /// Set to the closing delimiter (`"`, `"#`, `"##`, …) while inside
        /// a raw string.
        raw_close: Option<String>,
    }

    impl Stripper {
        fn new() -> Self {
            Self {
                in_block: false,
                raw_close: None,
            }
        }

        /// Strip comments and skip lines inside raw strings.
        fn strip(&mut self, line: &str) -> String {
            if let Some(close) = &self.raw_close {
                if line.contains(close.as_str()) {
                    self.raw_close = None;
                }
                return String::new();
            }

            if let Some((open, close)) = find_raw_open(line) {
                let rest = &line[open..];
                if rest.contains(close.as_str()) {
                    let before = &line[..open - 1];
                    let after_close = rest.find(close.as_str()).unwrap() + close.len();
                    let after = &rest[after_close..];
                    let combined = format!("{before} {after}");
                    return self.finish(combined.trim());
                }
                self.raw_close = Some(close);
                return String::new();
            }

            self.finish(line)
        }

        fn finish(&mut self, line: &str) -> String {
            let mut t = line.trim().to_string();
            loop {
                if self.in_block {
                    match t.find("*/") {
                        Some(i) => {
                            t = t[i + 2..].trim_start().to_string();
                            self.in_block = false;
                        }
                        None => return String::new(),
                    }
                }
                match t.find("/*") {
                    Some(start) => match t[start + 2..].find("*/") {
                        Some(end) => {
                            let before = t[..start].trim_end();
                            let after = &t[start + 2 + end + 2..];
                            t = format!("{before}{after}");
                        }
                        None => {
                            t.truncate(start);
                            self.in_block = true;
                            break;
                        }
                    },
                    None => break,
                }
            }
            if let Some(i) = t.find("//") {
                t.truncate(i);
            }
            t.trim_end().to_string()
        }
    }

    fn find_raw_open(line: &str) -> Option<(usize, String)> {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'r' && (i == 0 || !is_ident_byte(bytes[i - 1])) {
                let mut j = i + 1;
                let mut hashes = 0usize;
                while j < bytes.len() && bytes[j] == b'#' {
                    hashes += 1;
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'"' {
                    let close = format!("\"{}", "#".repeat(hashes));
                    return Some((j, close));
                }
            }
            i += 1;
        }
        None
    }

    fn is_ident_byte(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    // ---- Module-declaration parser ---------------------------------------

    #[derive(Debug, PartialEq, Eq)]
    struct ModuleDecl {
        name: String,
        path_override: Option<String>,
    }

    fn strip_visibility(s: &str) -> &str {
        let t = s.trim_start();
        if let Some(rest) = t.strip_prefix("pub(") {
            match rest.find(')') {
                Some(i) => rest[i + 1..].trim_start(),
                None => t,
            }
        } else if let Some(rest) = t.strip_prefix("pub ") {
            rest.trim_start()
        } else {
            t
        }
    }

    fn parse_module_decls(src: &str) -> Vec<ModuleDecl> {
        let mut out = Vec::new();
        let mut pending_path: Option<String> = None;
        let mut strip = Stripper::new();

        for raw in src.lines() {
            let t = strip.strip(raw);
            if t.is_empty() {
                continue;
            }

            if t.starts_with("#[path") {
                if let Some(eq) = t.find('=') {
                    let val = t[eq + 1..].trim().trim_end_matches(']').trim();
                    let val = val.trim_matches(|c| c == '"' || c == '\'');
                    pending_path = Some(val.to_string());
                }
                continue;
            }
            if t.starts_with("#[") || t.starts_with("#![") {
                continue;
            }

            let visible = strip_visibility(&t);
            if let Some(rest) = visible.strip_prefix("mod ") {
                let name: String = rest
                    .split(|c: char| c == ';' || c == '{' || c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .to_string();
                if name.is_empty() {
                    pending_path = None;
                    continue;
                }
                if rest.contains(';') {
                    out.push(ModuleDecl {
                        name,
                        path_override: pending_path.take(),
                    });
                } else {
                    pending_path = None;
                }
                continue;
            }
            pending_path = None;
        }
        out
    }

    // ---- `#[test] fn` parser ---------------------------------------------

    fn skip_attrs(line: &str) -> &str {
        let mut s = line;
        loop {
            let t = s.trim_start();
            if !t.starts_with("#[") {
                return t;
            }
            match t.find(']') {
                Some(i) => s = &t[i + 1..],
                None => return t,
            }
        }
    }

    fn extract_fn_name(rest: &str) -> Option<String> {
        let name: String = rest
            .split(|c: char| c == '(' || c == '<' || c.is_whitespace())
            .next()?
            .to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        Some(name)
    }

    fn collect_test_names(src: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut pending_test = false;
        let mut strip = Stripper::new();

        for raw in src.lines() {
            let t = strip.strip(raw);
            if t.is_empty() {
                continue;
            }

            if t.starts_with("#[") {
                if t.starts_with("#[test]") || t.starts_with("#[test ") || t.starts_with("#[test(")
                {
                    pending_test = true;
                }
                let after = skip_attrs(&t);
                if !after.is_empty() {
                    let vis = strip_visibility(after);
                    if let Some(rest) = vis.strip_prefix("fn ") {
                        if pending_test {
                            if let Some(name) = extract_fn_name(rest) {
                                out.push(name);
                            }
                        }
                        pending_test = false;
                    }
                }
                continue;
            }

            let vis = strip_visibility(&t);
            if let Some(rest) = vis.strip_prefix("fn ") {
                if pending_test {
                    if let Some(name) = extract_fn_name(rest) {
                        out.push(name);
                    }
                }
                pending_test = false;
                continue;
            }
            pending_test = false;
        }
        out
    }

    // ---- Doc-comment parser ----------------------------------------------

    fn documented_prefixes() -> Vec<String> {
        let src =
            std::fs::read_to_string(tests_dir().join("mod.rs")).expect("read physics/tests/mod.rs");
        let mut out = Vec::new();
        for line in src.lines() {
            // Real entries are `//!   <prefix> <desc>` — exactly three
            // spaces after `//!`. Continuations are indented six spaces;
            // after stripping the three-space header, their first
            // non-space character sits at index 0, so prefix_end == 0
            // rejects them.
            let Some(trimmed) = line.strip_prefix("//!   ") else {
                continue;
            };
            let prefix_end = trimmed
                .find(|c: char| !c.is_ascii_lowercase())
                .unwrap_or(trimmed.len());
            if prefix_end == 0 || prefix_end > MAX_GROUP_PREFIX_LEN {
                continue;
            }
            let after_prefix = &trimmed[prefix_end..];
            let spaces = after_prefix.chars().take_while(|c| *c == ' ').count();
            if spaces < 1 || after_prefix[spaces..].trim().is_empty() {
                continue;
            }
            out.push(trimmed[..prefix_end].to_string());
        }
        out
    }

    // ---- Diagnostics -----------------------------------------------------

    fn debug_report() -> String {
        let decls = module_decls();
        let decl_names: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
        let doc = documented_prefixes();
        let mut files: Vec<String> = std::fs::read_dir(tests_dir())
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        format!(
            "  declared modules: {decl_names:?}\n  \
               documented prefixes: {doc:?}\n  \
               files in directory: {files:?}"
        )
    }

    // ---- Guard tests -----------------------------------------------------

    fn module_decls() -> Vec<ModuleDecl> {
        let src =
            std::fs::read_to_string(tests_dir().join("mod.rs")).expect("read physics/tests/mod.rs");
        parse_module_decls(&src)
    }

    #[test]
    fn every_file_is_declared() {
        let dir = tests_dir();
        let declared: BTreeSet<String> = module_decls().into_iter().map(|m| m.name).collect();

        for entry in std::fs::read_dir(&dir).expect("read tests dir").flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if file_name.starts_with('.') {
                continue;
            }
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                if stem == "mod" {
                    continue;
                }
                assert!(
                    declared.contains(stem),
                    "src/physics/tests/{stem}.rs is not declared in mod.rs — \
                     add `mod {stem};` or delete the file\n{}",
                    debug_report()
                );
            } else if path.is_dir() {
                assert!(
                    path.join("mod.rs").is_file(),
                    "src/physics/tests/{file_name}/ is a directory but has no mod.rs"
                );
                assert!(
                    declared.contains(file_name),
                    "src/physics/tests/{file_name}/ exists but is not declared in mod.rs\n{}",
                    debug_report()
                );
            }
        }
    }

    #[test]
    fn every_declaration_has_a_file() {
        let dir = tests_dir();
        for decl in module_decls() {
            if let Some(p) = &decl.path_override {
                let target = dir.join(p);
                assert!(
                    target.is_file(),
                    "mod.rs declares `mod {};` with #[path = {p:?}] but {} is missing\n{}",
                    decl.name,
                    target.display(),
                    debug_report()
                );
            } else {
                let as_file = dir.join(format!("{}.rs", decl.name));
                let as_dir = dir.join(&decl.name).join("mod.rs");
                assert!(
                    as_file.is_file() || as_dir.is_file(),
                    "mod.rs declares `mod {};` but neither {} nor {} exists\n{}",
                    decl.name,
                    as_file.display(),
                    as_dir.display(),
                    debug_report()
                );
            }
        }
    }

    #[test]
    fn doc_comment_letter_prefixes_are_accurate() {
        let documented: BTreeSet<String> = documented_prefixes().into_iter().collect();
        let mut present: BTreeSet<String> = Default::default();

        for entry in std::fs::read_dir(tests_dir())
            .expect("read tests dir")
            .flatten()
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if INFRASTRUCTURE_FILES.contains(&stem) {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            for name in collect_test_names(&src) {
                let prefix = name.split('_').next().unwrap_or("");
                // Only 1–2-lowercase-letter prefixes are treated as a
                // claim to be part of a group. Anything longer, or with a
                // non-lowercase character, is a descriptive prefix and
                // needs no documentation. This catches typos like
                // `ab_foo` when `ab` is not a documented group, without
                // requiring every descriptive test to be listed.
                if !prefix.is_empty()
                    && prefix.len() <= MAX_GROUP_PREFIX_LEN
                    && prefix.chars().all(|c| c.is_ascii_lowercase())
                {
                    present.insert(prefix.to_string());
                }
            }
        }

        let mut errors: Vec<String> = Vec::new();
        for p in &documented {
            if !present.contains(p) {
                errors.push(format!(
                    "documented prefix `{p}` has no `{p}_*` test — remove \
                     `//!   {p}  …` from mod.rs or add a test with that prefix"
                ));
            }
        }
        for p in &present {
            if !documented.contains(p) {
                errors.push(format!(
                    "`{p}_*` uses a {}-letter lowercase prefix (looks like a \
                     group) but `{p}` is not documented — either add \
                     `//!   {p}  <description>` to mod.rs, or rename the test \
                     with a longer, descriptive prefix",
                    p.len()
                ));
            }
        }
        assert!(
            errors.is_empty(),
            "manifest guard: {} problem(s):\n  - {}\n\n{}",
            errors.len(),
            errors.join("\n  - "),
            debug_report()
        );
    }

    // ---- Parser self-tests -----------------------------------------------
    //
    // Fixtures are built with `concat!` so that every physical source line
    // starts with `"`. That way this file's own source has no line that
    // begins with `mod `, `fn `, or `#[`, and the parsers cannot mistake
    // the fixtures for real declarations while scanning `mod.rs`.

    const DECLS_FIXTURE: &str = concat!(
        "mod plain;\n",
        "pub mod public;\n",
        "pub(crate) mod crate_vis;\n",
        "pub(super) mod super_vis;\n",
        "pub(in crate) mod in_vis;\n",
        "#[cfg(test)]\n",
        "mod cfg_test;\n",
        "#[path = \"other.rs\"]\n",
        "mod with_path;\n",
        "mod inline { fn helper() {} }\n",
        "// mod commented_out;\n",
        "/* mod block_commented; */\n",
        "/* multi\n",
        "   mod also_commented;\n",
        "*/\n",
    );

    const TESTS_FIXTURE: &str = concat!(
        "fn helper_fn() {}\n",
        "pub fn public_helper() {}\n",
        "#[test]\n",
        "fn a_basic() {}\n",
        "#[test]\n",
        "#[should_panic]\n",
        "fn b_panics() {}\n",
        "#[test] fn c_inline() {}\n",
        "// #[test]\n",
        "// fn d_commented() {}\n",
        "#[cfg(test)]\n",
        "mod inner {\n",
        "    #[test]\n",
        "    fn e_nested() {}\n",
        "    fn not_a_test() {}\n",
        "}\n",
        "#[cfg(not(test))]\n",
        "fn f_cfg() {}\n",
        "const FIXTURE: &str = r#\"\n",
        "    mod plain_in_fixture;\n",
        "    #[test] fn zz_in_fixture() {}\n",
        "\"#;\n",
    );

    #[test]
    fn parser_handles_all_declaration_forms() {
        let decls = parse_module_decls(DECLS_FIXTURE);
        let names: Vec<&str> = decls.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "plain",
                "public",
                "crate_vis",
                "super_vis",
                "in_vis",
                "cfg_test",
                "with_path"
            ]
        );
        let with_path = decls.iter().find(|d| d.name == "with_path").unwrap();
        assert_eq!(with_path.path_override.as_deref(), Some("other.rs"));
    }

    #[test]
    fn parser_collects_test_names() {
        let names = collect_test_names(TESTS_FIXTURE);
        assert_eq!(names, vec!["a_basic", "b_panics", "c_inline", "e_nested"]);
    }

    /// The raw-string-aware stripper must not report `mod`/`fn` lines that
    /// appear inside `r#"…"#`.
    #[test]
    fn parser_ignores_raw_string_contents() {
        let src = concat!(
            "#[test]\n",
            "fn real_test() {}\n",
            "const FIXTURE: &str = r#\"\n",
            "    mod fake_module;\n",
            "    #[test] fn fake_test() {}\n",
            "\"#;\n",
        );
        assert_eq!(collect_test_names(src), vec!["real_test"]);
        assert!(parse_module_decls(src).is_empty());
    }
}
