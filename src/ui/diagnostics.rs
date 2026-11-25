//! Split out of the former single-file `ui.rs`; see `mod.rs`.


use egui::{Color32, RichText};

use crate::physics::{self as ph, Bodies, BodyKind};
use crate::theme::theme;
use super::*;

/// Orbit elements of the selected body about its parent, over time.
pub(super) fn light_curve_window(ctx: &egui::Context, st: &mut UiState, view: &ViewInfo, b: &Bodies) {
    let si = st.units_si;
    let mut open = st.light.visible;
    let star = view.selected.and_then(|id| b.index_of_id(id)).filter(|&i| b.kind[i] == BodyKind::Star);
    egui::Window::new("Light curve")
        .open(&mut open)
        .default_pos([640.0, 520.0])
        .default_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            let Some(s) = star else {
                ui.label(RichText::new("Select a star (click it, or N / P).").color(theme().colors.dim()));
                return;
            };
            ui.label(RichText::new(format!("{} as seen from far away", ph::tag(b, s))).strong());
            ui.add(egui::Slider::new(&mut st.light.azimuth_deg, 0.0..=360.0).text("azimuth °"));
            ui.add(egui::Slider::new(&mut st.light.elevation_deg, -90.0..=90.0).text("elevation °"))
                .on_hover_text("0° looks along the reference plane, where orbits are edge-on and planets transit; 90° looks down on it and nothing transits");
            let dip = st.light.samples.iter().map(|p| 1.0 - p.flux).fold(0.0, f64::max);
            ui.label(format!("deepest dip so far: {:.4} %", dip * 100.0));
            ui.horizontal(|ui| {
                if ui.button("clear").clicked() {
                    st.light.samples.clear();
                }
            });
            ui.label(RichText::new("One sample per frame; at high speed short transits fall between samples.").small().color(theme().colors.dim()));
            let spec = PlotSpec {
                title: "relative brightness",
                empty: "no data",
                color: theme().colors.accent(),
                warn_at: f64::INFINITY,
                symmetric: false,
                fmt: |v| format!("{v:.5}"),
            };
            plot_series(ui, &st.light.samples, si, &spec, |p| Some(p.flux));
        });
    st.light.visible = open;
}

pub(super) fn elements_window(ctx: &egui::Context, st: &mut UiState, view: &ViewInfo, b: &Bodies) {
    let si = st.units_si;
    let mut open = st.elem.visible;
    let selected = view.selected.and_then(|id| b.index_of_id(id));
    egui::Window::new("Orbit elements")
        .open(&mut open)
        .default_pos([640.0, 320.0])
        .default_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            let Some(i) = selected else {
                ui.label(RichText::new("Select a body (click it, or N / P).").color(theme().colors.dim()));
                return;
            };
            let Some(par) = parent_of(b, i) else {
                ui.label(format!("{} has no parent body to orbit.", ph::tag(b, i)));
                return;
            };
            ui.label(RichText::new(format!("{}  around  {}", ph::tag(b, i), ph::tag(b, par))).strong());
            elements_summary(ui, &st.elem);
            ui.horizontal(|ui| {
                if ui.button("clear").clicked() {
                    st.elem.samples.clear();
                }
                if ui.button("copy CSV").on_hover_text("copy the history to the clipboard").clicked() {
                    ui.ctx().copy_text(st.elem.to_csv());
                }
            });
            ui.separator();
            elements_plots(ui, &st.elem, si);
        });
    st.elem.visible = open;
}

/// Latest elements as text, with the range seen so far.
pub(super) fn elements_summary(ui: &mut egui::Ui, elem: &ElemHistory) {
    let Some(last) = elem.samples.back() else {
        ui.label(RichText::new("collecting… (the simulation must be running)").color(theme().colors.dim()));
        return;
    };
    let range = |f: fn(&ElemSample) -> f64| -> (f64, f64) {
        elem.samples.iter().map(f).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
    };
    let (e_lo, e_hi) = range(|s| s.ecc);
    let (i_lo, i_hi) = range(|s| s.inc_deg);
    egui::Grid::new("elem_summary").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
        ui.label("semi-major axis");
        ui.label(RichText::new(last.a_au.map_or("unbound".to_string(), fmt_au)).monospace());
        ui.end_row();
        ui.label("periapsis");
        ui.label(RichText::new(last.peri_au.map_or("-".to_string(), fmt_au)).monospace());
        ui.end_row();
        ui.label("eccentricity");
        ui.label(RichText::new(format!("{:.3}   (seen {:.3} to {:.3})", last.ecc, e_lo, e_hi)).monospace());
        ui.end_row();
        ui.label("inclination");
        ui.label(RichText::new(format!("{:.2}°   (seen {:.2}° to {:.2}°)", last.inc_deg, i_lo, i_hi)).monospace());
        ui.end_row();
    });
}

pub(super) fn fmt_au(v: f64) -> String {
    if v >= 0.05 {
        format!("{v:.3} AU")
    } else {
        format!("{:.2} Gm", v * ph::au_in_gm())
    }
}

pub(super) fn elements_plots(ui: &mut egui::Ui, elem: &ElemHistory, si: bool) {
    let accent = theme().colors.accent();
    let plots: [(&str, fn(f64) -> String, bool, fn(&ElemSample) -> Option<f64>); 3] = [
        ("semi-major axis", fmt_au, false, |s| s.a_au),
        ("eccentricity", |v| format!("{v:.3}"), false, |s| Some(s.ecc)),
        ("inclination", |v| format!("{v:.2}°"), false, |s| Some(s.inc_deg)),
    ];
    for (title, fmt, symmetric, get) in plots {
        let spec = PlotSpec {
            title,
            empty: "unbound",
            color: accent,
            warn_at: f64::INFINITY,
            symmetric,
            fmt,
        };
        plot_series(ui, &elem.samples, si, &spec, get);
    }
}

pub(super) fn diag_window(
    ctx: &egui::Context,
    st: &mut UiState,
    view: &ViewInfo,
    b: &Bodies,
    _act: &mut Actions,
) {
    let si = st.units_si;
    let mut open = st.diag.visible;
    egui::Window::new("Diagnostics")
        .open(&mut open)
        .default_pos([640.0, 60.0])
        .default_width(380.0)
        .resizable(true)
        .show(ctx, |ui| {
            diag_toolbar(ui, st);
            diag_summary(ui, &st.diag, view, b);
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height((ctx.content_rect().height() - 220.0).max(160.0))
                .show(ui, |ui| diag_plots(ui, &st.diag, si));
        });
    st.diag.visible = open;
}

/// Clear, copy-as-CSV and the history length.
pub(super) fn diag_toolbar(ui: &mut egui::Ui, st: &mut UiState) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("clear")
            .on_hover_text("forget the history and the angular-momentum baseline")
            .clicked()
        {
            st.diag.clear();
        }
        if ui
            .button("copy CSV")
            .on_hover_text("copy the sample history to the clipboard")
            .clicked()
        {
            ui.ctx().copy_text(st.diag.to_csv());
        }
        ui.label("history");
        let mut maxs = st.diag.max_samples;
        egui::ComboBox::from_id_salt("diag_history")
            .selected_text(format!("{} min", maxs / 60))
            .show_ui(ui, |ui| {
                for (v, name) in [(300usize, "5 min"), (900, "15 min"), (1800, "30 min")] {
                    ui.selectable_value(&mut maxs, v, name);
                }
            });
        st.diag.max_samples = maxs;
    });
}

/// Latest values of the conserved quantities and solver counters.
pub(super) fn diag_summary(ui: &mut egui::Ui, diag: &DiagHistory, view: &ViewInfo, b: &Bodies) {
    let last = diag.samples.back();
    let col_w = (ui.available_width() - 14.0) / 2.0;
    egui::Grid::new("diag_summary")
        .num_columns(2)
        .spacing([14.0, 2.0])
        .min_col_width(col_w)
        .show(ui, |ui| {
            kv(ui, "bodies", format!("{}", b.len()));
            kv(ui, "energy drift", fmt_pct(last.and_then(|s| s.drift)));
            kv(ui, "energy dissipated", fmt_pct(last.and_then(|s| s.dissipated)));
            kv(ui, "ang. momentum change", fmt_pct(last.and_then(|s| s.l_drift)));
            kv(ui, "steps / s", format!("{:.0}", view.steps_per_sec));
            kv(ui, "fine steps", format!("{}", b.stats.last_nsub));
            kv(ui, "events logged", format!("{}", b.events.len()));
        });
    if b.len() > ph::ENERGY_MAX_N {
        ui.label(
            RichText::new(format!(
                "Energy values are off above {} bodies (too slow to compute).",
                ph::ENERGY_MAX_N
            ))
            .small()
            .color(theme().colors.warn()),
        );
    }
}

/// The time-series plots, one per tracked quantity.
pub(super) fn diag_plots(ui: &mut egui::Ui, d: &DiagHistory, si: bool) {
    let plots: [(PlotSpec, fn(&DiagSample) -> Option<f64>); 5] = [
        (
            PlotSpec {
                title: "bodies",
                empty: "no data",
                color: theme().colors.good(),
                warn_at: f64::INFINITY,
                symmetric: false,
                fmt: fmt_int,
            },
            |s| Some(s.n as f64),
        ),
        (
            PlotSpec {
                title: "energy drift",
                empty: "n/a (system too large)",
                color: theme().colors.accent(),
                warn_at: 1.0,
                symmetric: true,
                fmt: fmt_pct_value,
            },
            |s| s.drift,
        ),
        (
            PlotSpec {
                title: "energy dissipated",
                empty: "n/a (system too large)",
                color: Color32::from_rgb(255, 170, 90),
                warn_at: f64::INFINITY,
                symmetric: false,
                fmt: fmt_pct_value,
            },
            |s| s.dissipated,
        ),
        (
            PlotSpec {
                title: "angular momentum change",
                empty: "no baseline",
                color: Color32::from_rgb(220, 130, 255),
                warn_at: 1.0,
                symmetric: true,
                fmt: fmt_pct_value,
            },
            |s| s.l_drift,
        ),
        (
            PlotSpec {
                title: "fine steps (close encounters)",
                empty: "no data",
                color: Color32::from_rgb(150, 220, 255),
                warn_at: f64::INFINITY,
                symmetric: false,
                fmt: fmt_int,
            },
            |s| Some(s.nsub as f64),
        ),
    ];
    for (spec, pick) in &plots {
        plot_series(ui, &d.samples, si, spec, pick);
        ui.add_space(4.0);
    }
    plot_ecc(ui, d);
}

