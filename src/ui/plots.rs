//! Split out of the former single-file `ui.rs`; see `mod.rs`.


use egui::{Align2, Color32, FontId, Stroke};

use crate::theme::theme;
use super::*;

/// Allocate a plot rectangle of the full available width. The painter is
/// clipped to the rectangle so drawing cannot spill into other widgets.
pub(super) fn plot_rect(ui: &mut egui::Ui, height: f32) -> (egui::Rect, egui::Painter, egui::Response) {
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::CornerRadius::same(2), Color32::from_gray(20));
    (rect, painter, resp)
}

pub(super) fn draw_axes(rect: egui::Rect, painter: &egui::Painter) {
    let s = Stroke::new(1.0_f32, Color32::from_gray(60));
    painter.line_segment([rect.left_bottom(), rect.right_bottom()], s);
    painter.line_segment([rect.left_bottom(), rect.left_top()], s);
}

/// Y range with 10 % padding. A flat series gets a range of at least 1.
pub(super) fn padded_range(lo: f64, hi: f64) -> (f64, f64) {
    let pad = (hi - lo) * 0.1;
    let pad = if pad > 0.0 {
        pad
    } else {
        (hi.abs() * 0.05).max(1.0)
    };
    (lo - pad, hi + pad)
}

pub(super) fn fmt_int(v: f64) -> String {
    format!("{}", v.round() as i64)
}

pub(super) fn fmt_pct_value(v: f64) -> String {
    fmt_pct(Some(v))
}

/// Look of one time-series plot.
pub(super) struct PlotSpec<'a> {
    pub(super) title: &'a str,
    pub(super) empty: &'a str,
    pub(super) color: Color32,
    /// Use the warning colour when the latest value is larger than this.
    pub(super) warn_at: f64,
    /// Centre the y axis on zero (for drift values).
    pub(super) symmetric: bool,
    pub(super) fmt: fn(f64) -> String,
}

/// Draw one line plot of `get` over the sample history. Hover shows the
/// time and value of the nearest sample.
pub(super) fn plot_series<S: Timed>(
    ui: &mut egui::Ui,
    samples: &std::collections::VecDeque<S>,
    si: bool,
    spec: &PlotSpec,
    get: impl Fn(&S) -> Option<f64>,
) {
    let (rect, painter, resp) = plot_rect(ui, 80.0);
    draw_axes(rect, &painter);

    let vals: Vec<Option<f64>> = samples
        .iter()
        .map(|s| get(s).filter(|v| v.is_finite()))
        .collect();
    let Some((lo, hi)) = finite_range(&vals) else {
        let msg = if samples.is_empty() { "collecting…" } else { spec.empty };
        painter.text(rect.center(), Align2::CENTER_CENTER, msg, FontId::monospace(10.0), theme().colors.dim());
        return;
    };

    let (y0, y1) = if spec.symmetric {
        let m = lo.abs().max(hi.abs()).max(1e-12) * 1.1;
        (-m, m)
    } else {
        padded_range(lo, hi)
    };
    let t0 = samples.front().map(|s| s.time()).unwrap_or(0.0);
    let t1 = samples.back().map(|s| s.time()).unwrap_or(0.0);
    let frame = PlotFrame { rect, t0, t_span: (t1 - t0).max(1e-9), y0, y1 };

    if spec.symmetric {
        let zy = rect.center().y;
        painter.line_segment(
            [egui::pos2(rect.left(), zy), egui::pos2(rect.right(), zy)],
            Stroke::new(1.0_f32, Color32::from_gray(50)),
        );
    }

    let current = vals.iter().rev().flatten().next().copied().unwrap_or(0.0);
    let line_col = if current.abs() > spec.warn_at { theme().colors.warn() } else { spec.color };
    draw_series_line(&painter, &frame, samples, &vals, line_col);
    draw_series_labels(&painter, &frame, spec, si, current, t1);
    if let Some(pos) = resp.hover_pos() {
        draw_series_hover(&painter, &frame, samples, &vals, spec, si, pos);
    }
}

/// Where a plot sits on screen and what time and value range it covers.
pub(super) struct PlotFrame {
    rect: egui::Rect,
    t0: f64,
    t_span: f64,
    y0: f64,
    y1: f64,
}

impl PlotFrame {
    fn map(&self, t: f64, v: f64) -> egui::Pos2 {
        let fx = ((t - self.t0) / self.t_span).clamp(0.0, 1.0) as f32;
        let fy = ((v - self.y0) / (self.y1 - self.y0)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            self.rect.left() + fx * self.rect.width(),
            self.rect.bottom() - fy * self.rect.height(),
        )
    }
}

/// Smallest and largest value, or `None` when there are none.
pub(super) fn finite_range(vals: &[Option<f64>]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in vals.iter().flatten() {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    lo.is_finite().then_some((lo, hi))
}

/// The line through the samples. A gap (value not available) breaks it.
pub(super) fn draw_series_line<S: Timed>(
    painter: &egui::Painter,
    frame: &PlotFrame,
    samples: &std::collections::VecDeque<S>,
    vals: &[Option<f64>],
    color: Color32,
) {
    let mut prev: Option<egui::Pos2> = None;
    for (s, v) in samples.iter().zip(vals.iter()) {
        match v {
            Some(v) => {
                let p = frame.map(s.time(), *v);
                if let Some(q) = prev {
                    painter.line_segment([q, p], Stroke::new(1.0_f32, color));
                }
                prev = Some(p);
            }
            None => prev = None,
        }
    }
    if let Some(p) = prev {
        painter.circle_filled(p, 2.0, color);
    }
}

/// Title with the current value, and the time at each end of the axis.
pub(super) fn draw_series_labels(
    painter: &egui::Painter,
    frame: &PlotFrame,
    spec: &PlotSpec,
    si: bool,
    current: f64,
    t1: f64,
) {
    let rect = frame.rect;
    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("{}  {}", spec.title, (spec.fmt)(current)),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        fmt_time(frame.t0, si),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        fmt_time(t1, si),
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
}

/// Cursor line, marker and readout for the sample nearest the pointer.
pub(super) fn draw_series_hover<S: Timed>(
    painter: &egui::Painter,
    frame: &PlotFrame,
    samples: &std::collections::VecDeque<S>,
    vals: &[Option<f64>],
    spec: &PlotSpec,
    si: bool,
    pos: egui::Pos2,
) {
    let rect = frame.rect;
    let fx = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64;
    let th = frame.t0 + fx * frame.t_span;
    let nearest = samples.iter().enumerate().min_by(|p, q| {
        (p.1.time() - th)
            .abs()
            .partial_cmp(&(q.1.time() - th).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some((k, s)) = nearest else { return };
    let Some(v) = vals[k] else { return };
    let p = frame.map(s.time(), v);
    painter.line_segment(
        [egui::pos2(p.x, rect.top()), egui::pos2(p.x, rect.bottom())],
        Stroke::new(1.0_f32, Color32::from_gray(90)),
    );
    painter.circle_filled(p, 3.0, spec.color);
    painter.text(
        rect.right_top() + egui::vec2(-4.0, 2.0),
        Align2::RIGHT_TOP,
        format!("{}  {}", fmt_time(s.time(), si), (spec.fmt)(v)),
        FontId::monospace(10.0),
        Color32::WHITE,
    );
}

pub(super) fn plot_ecc(ui: &mut egui::Ui, diag: &DiagHistory) {
    let (rect, painter, resp) = plot_rect(ui, 80.0);
    draw_axes(rect, &painter);

    if diag.eccs.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "no planets",
            FontId::monospace(10.0),
            theme().colors.dim(),
        );
        return;
    }

    let max_count = diag.ecc_bins.iter().copied().max().unwrap_or(1).max(1);
    let n = diag.ecc_bin_count;
    let w = rect.width() / n as f32;
    for (k, &cnt) in diag.ecc_bins.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        let x0 = rect.left() + k as f32 * w;
        let x1 = x0 + w * 0.9;
        let h = (cnt as f32 / max_count as f32) * rect.height();
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(x0, rect.bottom() - h),
                egui::pos2(x1, rect.bottom()),
            ),
            egui::CornerRadius::ZERO,
            theme().colors.accent(),
        );
    }

    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("eccentricity (n={}, peak={})", diag.eccs.len(), max_count),
        FontId::monospace(10.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        "0.0",
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        "1.0",
        FontId::monospace(9.0),
        theme().colors.dim(),
    );
    if let Some(pos) = resp.hover_pos() {
        let k = (((pos.x - rect.left()) / w) as usize).min(n - 1);
        let lo = k as f64 * diag.ecc_bin_width;
        painter.text(
            rect.right_top() + egui::vec2(-4.0, 2.0),
            Align2::RIGHT_TOP,
            format!(
                "e {:.2}–{:.2}  {} planets",
                lo,
                lo + diag.ecc_bin_width,
                diag.ecc_bins[k]
            ),
            FontId::monospace(10.0),
            Color32::WHITE,
        );
    }
}

