//! Split out of the former single-file `ui.rs`; see `mod.rs`.

use std::time::{Duration, Instant};

use egui::{Color32, RichText};

use crate::physics::Bodies;
use crate::theme::theme;
use super::*;

/// Make the next frame rebuild the body list (after a sort / filter change).
pub(super) fn force_list_refresh(st: &mut UiState) {
    st.list_time = Instant::now() - Duration::from_secs(5);
}

pub(super) fn list_window(
    ctx: &egui::Context,
    st: &mut UiState,
    view: &ViewInfo,
    b: &Bodies,
    act: &mut Actions,
) {
    if st.list_time.elapsed() > Duration::from_millis(300) || st.list_rows.is_empty() {
        refresh_rows(st, b);
        st.list_time = Instant::now();
    }
    // Bodies that merged or escaped since the last frame leave the ticked set.
    st.multi.retain(|id| b.index_of_id(*id).is_some());
    let si = st.units_si;
    let mut open = st.show_list;
    egui::Window::new("Bodies")
        .open(&mut open)
        .default_pos([320.0, 60.0])
        .default_width(600.0)
        .show(ctx, |ui| {
            list_toolbar(ui, st, b);
            list_bulk_actions(ui, st, act);
            egui::ScrollArea::vertical()
                .max_height(420.0)
                .show(ui, |ui| list_table(ui, st, view, si, act));
            ui.label(
                RichText::new(
                    "Click a name to select it. Tick boxes to edit many bodies at once. \
                     Distance, speed and period are relative to the body's parent, \
                     or to the primary star for bodies without one (belts, debris).",
                )
                .small()
                .color(theme().colors.dim()),
            );
        });
    st.show_list = open;
}

/// Body count, search box and the "show N" limit.
pub(super) fn list_toolbar(ui: &mut egui::Ui, st: &mut UiState, b: &Bodies) {
    ui.horizontal(|ui| {
        let sys_txt = st
            .system_filter
            .and_then(|id| st.systems.iter().position(|s| s.root_id == id))
            .map(|k| format!("system {}: ", k + 1))
            .unwrap_or_default();
        let shown = if sys_txt.is_empty() { b.len() } else { st.list_rows.len() };
        ui.label(RichText::new(format!("{}{} bodies", sys_txt, shown)).color(theme().colors.dim()));
        let r = ui.add(
            egui::TextEdit::singleline(&mut st.list_query)
                .hint_text("search name or class")
                .desired_width(170.0),
        );
        if r.changed() {
            force_list_refresh(st);
        }
        if !st.list_query.is_empty() && ui.small_button("✕").clicked() {
            st.list_query.clear();
            force_list_refresh(st);
        }
        ui.label("show");
        let mut lim = st.list_limit;
        egui::ComboBox::from_id_salt("list_limit")
            .selected_text(lim.to_string())
            .show_ui(ui, |ui| {
                for v in [50usize, 100, 250, 500, 1000] {
                    ui.selectable_value(&mut lim, v, v.to_string());
                }
            });
        if lim != st.list_limit {
            st.list_limit = lim;
            force_list_refresh(st);
        }
    });
}

/// "Tick all shown" and the batch edits that apply to the ticked bodies.
pub(super) fn list_bulk_actions(ui: &mut egui::Ui, st: &mut UiState, act: &mut Actions) {
    ui.horizontal_wrapped(|ui| {
        if ui.small_button("tick all shown").clicked() {
            for r in &st.list_rows {
                st.multi.insert(r.id);
            }
        }
        if st.multi.is_empty() {
            return;
        }
        let ids: Vec<u64> = st.multi.iter().copied().collect();
        ui.label(RichText::new(format!("{} ticked", ids.len())).color(theme().colors.accent()));
        if ui.small_button("mass ÷2").clicked() {
            act.multi_edit = Some((ids.clone(), Edit::MassMul(0.5)));
        }
        if ui.small_button("mass ×2").clicked() {
            act.multi_edit = Some((ids.clone(), Edit::MassMul(2.0)));
        }
        if ui.small_button("circularize").clicked() {
            act.multi_edit = Some((ids.clone(), Edit::Circularize));
        }
        egui::ComboBox::from_id_salt("multi_kind")
            .selected_text("set type…")
            .show_ui(ui, |ui| {
                for (k, name) in KIND_CHOICES {
                    if ui.selectable_label(false, name).clicked() {
                        act.multi_edit = Some((ids.clone(), Edit::SetKind(k)));
                    }
                }
            });
        if ui.small_button("delete").clicked() {
            act.multi_edit = Some((ids.clone(), Edit::Delete));
            st.multi.clear();
        }
        if ui.small_button("clear ticks").clicked() {
            st.multi.clear();
        }
    });
}

/// The sortable table of bodies.
pub(super) fn list_table(ui: &mut egui::Ui, st: &mut UiState, view: &ViewInfo, si: bool, act: &mut Actions) {
    const HEADERS: [(&str, SortKey); 7] = [
        ("name", SortKey::Id),
        ("class", SortKey::Id),
        ("mass", SortKey::Mass),
        ("radius", SortKey::Radius),
        ("dist", SortKey::Dist),
        ("speed", SortKey::Speed),
        ("period", SortKey::Period),
    ];
    /// The "class" column (index 1) is not sortable.
    const CLASS_COLUMN: usize = 1;
    egui::Grid::new("list_grid")
        .num_columns(8)
        .spacing([12.0, 3.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("");
            for (k, (name, key)) in HEADERS.iter().enumerate() {
                list_header_cell(ui, st, name, *key, k != CLASS_COLUMN);
            }
            ui.end_row();
            for row in &st.list_rows {
                let mut ticked = st.multi.contains(&row.id);
                if ui.checkbox(&mut ticked, "").changed() {
                    if ticked {
                        st.multi.insert(row.id);
                    } else {
                        st.multi.remove(&row.id);
                    }
                }
                let selected = view.selected == Some(row.id);
                if ui.selectable_label(selected, &row.label).clicked() {
                    act.select = Some(Some(row.id));
                }
                ui.label(&row.class);
                ui.label(RichText::new(fmt_mass(row.m, si)).monospace());
                ui.label(RichText::new(fmt_len(row.r, si)).monospace());
                ui.label(RichText::new(fmt_len(row.dist, si)).monospace());
                ui.label(RichText::new(fmt_speed(row.speed, si)).monospace());
                ui.label(RichText::new(fmt_time(row.period, si)).monospace());
                ui.end_row();
            }
        });
}

/// One header cell: shows the sort arrow and, when `sortable`, a click
/// toggles the direction (or switches the sort key).
pub(super) fn list_header_cell(
    ui: &mut egui::Ui,
    st: &mut UiState,
    name: &str,
    key: SortKey,
    sortable: bool,
) {
    let active = st.sort == key && sortable;
    let arrow = match (active, st.sort_desc) {
        (true, true) => " ▼",
        (true, false) => " ▲",
        _ => "",
    };
    let colour = if active { theme().colors.accent() } else { Color32::WHITE };
    let text = RichText::new(format!("{}{}", name, arrow)).strong().color(colour);
    if !sortable {
        ui.label(text);
        return;
    }
    if ui.add(egui::Label::new(text).sense(egui::Sense::click())).clicked() {
        if st.sort == key {
            st.sort_desc = !st.sort_desc;
        } else {
            st.sort = key;
            st.sort_desc = key != SortKey::Id;
        }
        force_list_refresh(st);
    }
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

pub(super) fn help_window(ctx: &egui::Context, st: &mut UiState) {
    let mut open = st.show_help;
    egui::Window::new("Help")
        .open(&mut open)
        .default_pos([340.0, 80.0])
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("help_keys").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
                for (k, d) in [
                    ("Space", "pause / resume"),
                    (".", "advance one step"),
                    ("R", "new system (next seed)"),
                    ("drag", "pan the view"),
                    ("wheel, + / −", "zoom toward the cursor"),
                    ("W A S D, arrows", "pan"),
                    ("click", "select a body (click empty space to deselect)"),
                    ("N / P", "select next / previous planet by distance"),
                    ("C", "follow the selected body"),
                    ("F, 0", "fit the whole system"),
                    ("O", "orbit lines"),
                    ("L", "labels"),
                    ("U", "physical / simulation units"),
                    ("B", "body list"),
                    ("Y", "system list"),
                    ("E", "events"),
                    ("K", "saves window"),
                    ("F5 / F9", "quick save / quick load"),
                    ("G", "diagnostics"),
                    ("X", "orbit elements of the selected body"),
                    ("I", "light curve of the selected star (transits, eclipses)"),
                    ("Z", "habitable zone bands"),
                    ("V", "vsync"),
                    ("`", "show / hide the control panel"),
                    ("Delete", "delete the selected body"),
                    ("Esc", "deselect"),
                    ("? or H", "this help"),
                ] {
                    ui.label(RichText::new(k).monospace().color(theme().colors.accent()));
                    ui.label(d);
                    ui.end_row();
                }
            });
            ui.separator();
            ui.label(RichText::new("About the physics").strong());
            ui.label(
                RichText::new(
                    "Bodies collide at their true physical radius, checked along the whole path of each step, so fast bodies cannot pass through each other. \
                     Slow impacts merge. Fast impacts throw out debris. A planet inside its Roche limit breaks apart. \
                     The heaviest body, or the star or black hole, always survives a merge. \
                     Energy lost to heat, break-ups and escapes is tracked separately from integrator drift.",
                )
                .small(),
            );
        });
    st.show_help = open;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

