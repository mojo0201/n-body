//! Split out of the former single-file `ui.rs`; see `mod.rs`.

use std::time::{Duration, Instant};

use egui::{Color32, RichText};

use crate::theme::theme;
use super::*;

/// Systems window: one row per spawned system. Clicking a row moves the
/// camera to it and limits the body list to its bodies. Opens beside the
/// body list.
pub(super) fn systems_window(ctx: &egui::Context, st: &mut UiState, act: &mut Actions) {
    let mut open = st.show_systems;
    let mut pick: Option<Option<usize>> = None;
    egui::Window::new("Systems")
        .open(&mut open)
        .default_pos([940.0, 60.0])
        .default_width(380.0)
        .show(ctx, |ui| {
            if st.systems.len() < 2 {
                ui.label(
                    RichText::new(
                        "Only one system. Raise the \"systems\" slider under Generate (Random) to get more.",
                    )
                    .small()
                    .color(theme().colors.dim()),
                );
                return;
            }
            let cur = st
                .system_filter
                .and_then(|id| st.systems.iter().position(|s| s.root_id == id));
            systems_summary_row(ui, st, cur, &mut pick);
            let order = systems_order(st);
            egui::ScrollArea::vertical()
                .max_height((ctx.content_rect().height() - 220.0).max(160.0))
                .show(ui, |ui| systems_grid(ui, st, &order, cur, &mut pick));
            ui.label(
                RichText::new("Click a system to focus it and filter the body list.")
                    .small()
                    .color(theme().colors.dim()),
            );
        });
    st.show_systems = open;
    if let Some(p) = pick {
        apply_system_pick(st, act, p);
    }
}

/// The system count and the "show all" button.
pub(super) fn systems_summary_row(ui: &mut egui::Ui, st: &UiState, cur: Option<usize>, pick: &mut Option<Option<usize>>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} systems", st.systems.len())).color(theme().colors.dim()));
        if ui
            .add_enabled(cur.is_some(), egui::Button::new("show all"))
            .on_hover_text("remove the system filter and fit everything")
            .clicked()
        {
            *pick = Some(None);
        }
    });
}

/// Indices into `st.systems` in the chosen sort order, so that picking works
/// whatever the order.
pub(super) fn systems_order(st: &UiState) -> Vec<usize> {
    let mut order: Vec<usize> = (0..st.systems.len()).collect();
    let sys = &st.systems;
    order.sort_by(|&x, &y| {
        let (p, q) = (&sys[x], &sys[y]);
        match st.sys_sort {
            SysSort::Index => x.cmp(&y),
            SysSort::Name => p.label.cmp(&q.label),
            SysSort::Bodies => p.n.cmp(&q.n),
            SysSort::Mass => p.mass.partial_cmp(&q.mass).unwrap_or(std::cmp::Ordering::Equal),
            SysSort::Radius => p.radius.partial_cmp(&q.radius).unwrap_or(std::cmp::Ordering::Equal),
        }
    });
    if st.sys_sort_desc {
        order.reverse();
    }
    order
}

pub(super) fn systems_grid(
    ui: &mut egui::Ui,
    st: &mut UiState,
    order: &[usize],
    cur: Option<usize>,
    pick: &mut Option<Option<usize>>,
) {
    let si = st.units_si;
    egui::Grid::new("systems_grid")
        .num_columns(5)
        .spacing([12.0, 3.0])
        .striped(true)
        .show(ui, |ui| {
            for (name, key) in [
                ("#", SysSort::Index),
                ("main star", SysSort::Name),
                ("bodies", SysSort::Bodies),
                ("mass", SysSort::Mass),
                ("radius", SysSort::Radius),
            ] {
                systems_header_cell(ui, st, name, key);
            }
            ui.end_row();
            for &k in order {
                let s = &st.systems[k];
                ui.label(RichText::new(format!("{}", k + 1)).monospace().color(theme().colors.dim()));
                if ui.selectable_label(cur == Some(k), &s.label).clicked() {
                    *pick = Some(Some(k));
                }
                ui.label(RichText::new(format!("{}", s.n)).monospace());
                ui.label(RichText::new(fmt_mass(s.mass, si)).monospace());
                ui.label(RichText::new(fmt_len(s.radius, si)).monospace());
                ui.end_row();
            }
        });
}

/// A sortable column header: click to sort, click again to reverse.
pub(super) fn systems_header_cell(ui: &mut egui::Ui, st: &mut UiState, name: &str, key: SysSort) {
    let active = st.sys_sort == key;
    let arrow = match (active, st.sys_sort_desc) {
        (true, true) => " ▼",
        (true, false) => " ▲",
        (false, _) => "",
    };
    let color = if active { theme().colors.accent() } else { Color32::WHITE };
    let txt = RichText::new(format!("{}{}", name, arrow)).strong().color(color);
    if ui
        .add(egui::Label::new(txt).sense(egui::Sense::click()))
        .on_hover_text("click to sort")
        .clicked()
    {
        if active {
            st.sys_sort_desc = !st.sys_sort_desc;
        } else {
            st.sys_sort = key;
            st.sys_sort_desc = key != SysSort::Index && key != SysSort::Name;
        }
    }
}

/// Filter to a system and focus the camera on it, or clear the filter.
pub(super) fn apply_system_pick(st: &mut UiState, act: &mut Actions, pick: Option<usize>) {
    match pick {
        None => {
            st.system_filter = None;
            act.fit = true;
        }
        Some(k) => {
            let (id, x, y, r) = {
                let s = &st.systems[k];
                (s.root_id, s.x, s.y, s.radius)
            };
            st.system_filter = Some(id);
            act.focus = Some([x, y, r]);
        }
    }
    st.list_time = Instant::now() - Duration::from_secs(5);
}

