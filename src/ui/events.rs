//! Split out of the former single-file `ui.rs`; see `mod.rs`.


use egui::{Align2, Color32, FontId, RichText};

use crate::physics::{self as ph, Bodies, EventKind};
use crate::theme::theme;
use super::*;

/// Map an event kind to a text color: merges are white-ish blue, impacts
/// orange, tidal purple, escapes gray, user edits green.
pub(super) fn event_color(k: EventKind) -> Color32 {
    match k {
        EventKind::Merge => Color32::from_rgb(160, 200, 255),
        EventKind::Impact => Color32::from_rgb(255, 170, 90),
        EventKind::Tidal => Color32::from_rgb(220, 130, 255),
        EventKind::Escape => Color32::from_gray(150),
        EventKind::User => Color32::from_rgb(130, 220, 150),
        EventKind::Evolve => Color32::from_rgb(255, 220, 130),
        EventKind::Comet => Color32::from_rgb(150, 220, 255),
        EventKind::Interstellar => Color32::from_rgb(220, 170, 255),
        EventKind::Eject => Color32::from_rgb(255, 140, 90),
    }
}

/// Bit for one event kind in `UiState::event_mask`.
pub(super) fn event_bit(k: EventKind) -> u16 {
    match k {
        EventKind::Merge => 1 << 0,
        EventKind::Impact => 1 << 1,
        EventKind::Tidal => 1 << 2,
        EventKind::Escape => 1 << 3,
        EventKind::User => 1 << 4,
        EventKind::Evolve => 1 << 5,
        EventKind::Comet => 1 << 6,
        EventKind::Interstellar => 1 << 7,
        EventKind::Eject => 1 << 8,
    }
}

pub(super) const EVENT_KINDS: [(EventKind, &str); 9] = [
    (EventKind::Merge, "merge"),
    (EventKind::Impact, "impact"),
    (EventKind::Tidal, "tidal"),
    (EventKind::Escape, "escape"),
    (EventKind::Eject, "eject"),
    (EventKind::Evolve, "evolve"),
    (EventKind::Comet, "comet"),
    (EventKind::Interstellar, "interstellar"),
    (EventKind::User, "edit"),
];

/// Event log window: a stacked timeline chart, a per-kind filter with counts,
/// and a table (when, type, what, where). Click a row to centre the view.
pub(super) fn events_window(ctx: &egui::Context, st: &mut UiState, b: &Bodies, act: &mut Actions) {
    let mut open = st.show_events;
    let si = st.units_si;
    egui::Window::new(format!("Events ({})", b.events.len()))
        .id(egui::Id::unique("events_window"))
        .open(&mut open)
        .default_pos([340.0, 80.0])
        .default_size([620.0, 520.0])
        .resizable(true)
        .show(ctx, |ui| {
            if b.events.is_empty() {
                ui.label(RichText::new("nothing has happened yet").color(theme().colors.dim()));
                return;
            }
            event_filter_chips(ui, st, &count_events_per_kind(b));
            let mask = st.event_mask;
            draw_event_timeline(ui, b, mask, si);
            ui.add_space(4.0);
            let table_height = (ctx.content_rect().height() - 330.0).max(160.0);
            events_table(ui, b, mask, si, table_height, act);
        });
    st.show_events = open;
}

/// Number of timeline bins in the event chart.
pub(super) const EVENT_TIMELINE_BINS: usize = 48;

/// Index of `kind` in `EVENT_KINDS` (kinds that share a filter bit share a slot).
pub(super) fn event_kind_index(kind: EventKind) -> Option<usize> {
    EVENT_KINDS
        .iter()
        .position(|(k, _)| event_bit(*k) == event_bit(kind))
}

pub(super) fn count_events_per_kind(b: &Bodies) -> [usize; 9] {
    let mut counts = [0usize; 9];
    for e in b.events.iter() {
        if let Some(k) = event_kind_index(e.kind) {
            counts[k] += 1;
        }
    }
    counts
}

/// One toggle chip per event kind, labelled with its count.
pub(super) fn event_filter_chips(ui: &mut egui::Ui, st: &mut UiState, counts: &[usize; 9]) {
    ui.horizontal_wrapped(|ui| {
        for (k, (kind, name)) in EVENT_KINDS.iter().enumerate() {
            let bit = event_bit(*kind);
            let on = st.event_mask & bit != 0;
            let colour = if on { event_color(*kind) } else { theme().colors.dim() };
            let label = RichText::new(format!("{} {}", name, counts[k])).small().color(colour);
            if ui.selectable_label(on, label).clicked() {
                st.event_mask ^= bit;
            }
        }
    });
}

/// Stacked timeline: events per time bin, coloured by kind, with the time
/// range and a hover read-out.
pub(super) fn draw_event_timeline(ui: &mut egui::Ui, b: &Bodies, mask: u16, si: bool) {
    const BINS: usize = EVENT_TIMELINE_BINS;
    let t0 = b.events.iter().map(|e| e.time).fold(f64::INFINITY, f64::min);
    let t1 = b.time.max(t0 + 1e-9);
    let span = (t1 - t0).max(1e-9);
    let mut bins = [[0u32; 9]; BINS];
    for e in b.events.iter().filter(|e| mask & event_bit(e.kind) != 0) {
        if let Some(k) = event_kind_index(e.kind) {
            let f = ((e.time - t0) / span).clamp(0.0, 1.0);
            let bi = ((f * BINS as f64) as usize).min(BINS - 1);
            bins[bi][k] += 1;
        }
    }
    let peak = bins.iter().map(|c| c.iter().sum::<u32>()).max().unwrap_or(1).max(1);
    let (rect, painter, resp) = plot_rect(ui, 90.0);
    draw_axes(rect, &painter);
    let bw = rect.width() / BINS as f32;
    for (bi, c) in bins.iter().enumerate() {
        let mut y = rect.bottom();
        for (k, &n) in c.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let h = n as f32 / peak as f32 * (rect.height() - 14.0);
            let x0 = rect.left() + bi as f32 * bw;
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0 + 0.5, y - h), egui::pos2(x0 + bw - 0.5, y)),
                egui::CornerRadius::ZERO,
                event_color(EVENT_KINDS[k].0),
            );
            y -= h;
        }
    }
    let dim = theme().colors.dim();
    painter.text(
        rect.left_top() + egui::vec2(4.0, 2.0),
        Align2::LEFT_TOP,
        format!("events over time (peak {} per bin)", peak),
        FontId::monospace(10.0),
        dim,
    );
    painter.text(
        rect.left_bottom() + egui::vec2(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        fmt_time(t0, si),
        FontId::monospace(9.0),
        dim,
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        Align2::RIGHT_BOTTOM,
        fmt_time(t1, si),
        FontId::monospace(9.0),
        dim,
    );
    if let Some(pos) = resp.hover_pos() {
        let bi = (((pos.x - rect.left()) / bw) as usize).min(BINS - 1);
        let lo = t0 + span * bi as f64 / BINS as f64;
        let hi = t0 + span * (bi + 1) as f64 / BINS as f64;
        let total: u32 = bins[bi].iter().sum();
        painter.text(
            rect.right_top() + egui::vec2(-4.0, 2.0),
            Align2::RIGHT_TOP,
            format!("{} – {}: {}", fmt_time(lo, si), fmt_time(hi, si), total),
            FontId::monospace(10.0),
            Color32::WHITE,
        );
    }
}

/// Number of newest events listed in the table.
pub(super) const EVENT_TABLE_ROWS: usize = 300;

/// The event table, newest first. Distance is measured from the primary's
/// current position; clicking a row asks the view to centre on the event.
pub(super) fn events_table(
    ui: &mut egui::Ui,
    b: &Bodies,
    mask: u16,
    si: bool,
    max_height: f32,
    act: &mut Actions,
) {
    let prim = ph::primary_index(b);
    let shown = b.events.iter().filter(|e| mask & event_bit(e.kind) != 0).count();
    ui.label(
        RichText::new(format!("{} shown, newest first (last {})", shown, EVENT_TABLE_ROWS))
            .small()
            .color(theme().colors.dim()),
    );
    egui::ScrollArea::both()
        .max_height(max_height)
        .auto_shrink([false, false])
        .id_salt("events_scroll")
        .show(ui, |ui| {
            egui::Grid::new("events_grid")
                .num_columns(5)
                .spacing([12.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    for h in ["time", "ago", "type", "what", "from primary"] {
                        ui.label(RichText::new(h).strong());
                    }
                    ui.end_row();
                    let rows = b
                        .events
                        .iter()
                        .rev()
                        .filter(|e| mask & event_bit(e.kind) != 0)
                        .take(EVENT_TABLE_ROWS);
                    for e in rows {
                        event_row(ui, b, e, prim, si, act);
                    }
                });
        });
}

/// One row of the event table.
pub(super) fn event_row(
    ui: &mut egui::Ui,
    b: &Bodies,
    e: &ph::Event,
    primary: Option<usize>,
    si: bool,
    act: &mut Actions,
) {
    let col = event_color(e.kind);
    let name = event_kind_index(e.kind).map(|k| EVENT_KINDS[k].1).unwrap_or("?");
    let dim = theme().colors.dim();
    ui.label(RichText::new(fmt_time(e.time, si)).monospace().small().color(dim));
    ui.label(
        RichText::new(fmt_time((b.time - e.time).max(0.0), si))
            .monospace()
            .small()
            .color(dim),
    );
    ui.label(RichText::new(name).small().color(col));
    let resp = ui
        .add(egui::Label::new(RichText::new(&e.text).small().color(col)).sense(egui::Sense::click()))
        .on_hover_text("click to centre the view here");
    if resp.clicked() {
        act.center_on = Some([e.x, e.y]);
    }
    let dist = primary
        .map(|p| ((e.x - b.x[p]).powi(2) + (e.y - b.y[p]).powi(2)).sqrt())
        .unwrap_or(f64::NAN);
    ui.label(RichText::new(fmt_len(dist, si)).monospace().small());
    ui.end_row();
}

// ---------------------------------------------------------------------------
// Selected body
// ---------------------------------------------------------------------------

