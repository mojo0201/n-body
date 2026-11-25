//! Split out of the former single-file `ui.rs`; see `mod.rs`.

use std::time::{Duration, Instant, SystemTime};

use egui::{Align2, Color32, RichText};

use crate::physics::{self as ph};
use crate::theme::theme;
use super::*;

/// What the second click of a two-click button in the Saves window will do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveConfirm {
    Overwrite,
    Delete,
}

pub(super) fn saves_fmt_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else if b >= KB {
        format!("{:.0} kB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

pub(super) fn saves_fmt_age(t: Option<SystemTime>) -> String {
    let Some(secs) = t
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .map(|d| d.as_secs())
    else {
        return "–".to_string();
    };
    match secs {
        0..=59 => "just now".to_string(),
        60..=3599 => format!("{} min ago", secs / 60),
        3600..=86_399 => format!("{} h ago", secs / 3600),
        _ => format!("{} d ago", secs / 86_400),
    }
}

/// How long a "confirm" button stays armed in the Saves window.
pub(super) const SAVE_CONFIRM_WINDOW: Duration = Duration::from_secs(4);
/// How long the result message of a save / delete stays visible.
pub(super) const SAVE_MESSAGE_SHOWN: Duration = Duration::from_secs(10);

/// The Saves window: save under a name, load, delete. Files live in
/// `physics::SAVE_DIR`. Overwrite and delete need a second click.
pub(super) fn saves_window(ctx: &egui::Context, st: &mut UiState, act: &mut Actions) {
    let mut open = st.show_saves;
    egui::Window::new("Saves")
        .id(egui::Id::unique("saves_window"))
        .open(&mut open)
        .default_width(360.0)
        .resizable(true)
        .show(ctx, |ui| {
            if st.save_list_time.elapsed() > Duration::from_secs(3) {
                st.refresh_saves();
            }
            ui.label(
                RichText::new(format!(
                    "Folder: {}/    F5 quick save, F9 quick load",
                    ph::SAVE_DIR
                ))
                .small()
                .color(theme().colors.dim()),
            );
            save_box(ui, st, act);
            ui.separator();
            if st.save_list.is_empty() {
                ui.label(RichText::new("No saves yet.").color(theme().colors.dim()));
            } else {
                save_list_table(ui, st, act);
            }
            save_message(ui, st);
            if st.save_confirm.is_some() {
                // Redraw so a confirm button goes away when its time ends.
                ui.ctx().request_repaint_after(Duration::from_millis(500));
            }
        });
    st.show_saves = open;
}

/// The name box and save button. Overwriting an existing save needs a second
/// click within `SAVE_CONFIRM_WINDOW`.
pub(super) fn save_box(ui: &mut egui::Ui, st: &mut UiState, act: &mut Actions) {
    let dim = theme().colors.dim();
    let clean = ph::sanitize_save_name(&st.save_name);
    let exists = clean.as_deref().is_some_and(ph::save_exists);
    let armed = matches!(
        &st.save_confirm,
        Some((n, SaveConfirm::Overwrite, t))
            if Some(n.as_str()) == clean.as_deref() && t.elapsed() < SAVE_CONFIRM_WINDOW
    );
    ui.horizontal(|ui| {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut st.save_name)
                .hint_text("save name")
                .desired_width(180.0),
        );
        let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let label = match (exists, armed) {
            (true, true) => "confirm overwrite",
            (true, false) => "save…",
            _ => "save",
        };
        let clicked = ui
            .add_enabled(clean.is_some(), egui::Button::new(label))
            .on_disabled_hover_text("Use letters, digits, space, - or _.")
            .clicked();
        if let (true, Some(name)) = (clicked || enter, clean.clone()) {
            if exists && !armed {
                st.save_confirm = Some((name, SaveConfirm::Overwrite, Instant::now()));
            } else {
                act.save = Some(name);
                st.save_confirm = None;
            }
        }
    });
    if !st.save_name.trim().is_empty() && clean.as_deref() != Some(st.save_name.trim()) {
        let shown = clean.as_deref().unwrap_or("(no valid name)");
        ui.label(RichText::new(format!("Name will be: {shown}")).small().color(dim));
    }
    if exists && armed {
        ui.label(
            RichText::new("A save with this name exists. Click again to replace it.")
                .small()
                .color(theme().colors.warn()),
        );
    }
}

/// The saved games: name, size, age, and load / delete buttons.
pub(super) fn save_list_table(ui: &mut egui::Ui, st: &mut UiState, act: &mut Actions) {
    let dim = theme().colors.dim();
    let entries = st.save_list.clone();
    egui::ScrollArea::vertical()
        .id_salt("saves_scroll")
        .max_height(260.0)
        .show(ui, |ui| {
            egui::Grid::new("saves_grid")
                .num_columns(4)
                .spacing([10.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    for e in &entries {
                        if ui
                            .selectable_label(false, &e.name)
                            .on_hover_text("Click to copy the name to the box.")
                            .clicked()
                        {
                            st.save_name = e.name.clone();
                        }
                        ui.label(
                            RichText::new(saves_fmt_size(e.bytes))
                                .small()
                                .monospace()
                                .color(dim),
                        );
                        ui.label(RichText::new(saves_fmt_age(e.modified)).small().color(dim));
                        ui.horizontal(|ui| save_row_buttons(ui, st, act, &e.name));
                        ui.end_row();
                    }
                });
        });
}

/// "load" and the two-click "delete" for one save.
pub(super) fn save_row_buttons(ui: &mut egui::Ui, st: &mut UiState, act: &mut Actions, name: &str) {
    let red = Color32::from_rgb(220, 90, 90);
    if ui.small_button("load").clicked() {
        act.load = Some(name.to_string());
    }
    let delete_armed = matches!(
        &st.save_confirm,
        Some((n, SaveConfirm::Delete, t)) if n == name && t.elapsed() < SAVE_CONFIRM_WINDOW
    );
    if delete_armed {
        if ui.small_button(RichText::new("confirm").color(red)).clicked() {
            match ph::delete_save(name) {
                Ok(()) => st.set_save_msg(format!("Deleted {name}"), true),
                Err(er) => st.set_save_msg(format!("Delete failed: {er}"), false),
            }
            st.save_confirm = None;
            st.refresh_saves();
        }
    } else if ui.small_button("delete…").clicked() {
        st.save_confirm = Some((name.to_string(), SaveConfirm::Delete, Instant::now()));
    }
}

/// The result of the last save / delete, shown for a few seconds.
pub(super) fn save_message(ui: &mut egui::Ui, st: &UiState) {
    let Some((text, ok, t)) = &st.save_msg else {
        return;
    };
    if t.elapsed() >= SAVE_MESSAGE_SHOWN {
        return;
    }
    ui.add_space(4.0);
    let colour = if *ok { theme().colors.good() } else { theme().colors.warn() };
    ui.label(RichText::new(text).small().color(colour));
}

/// A short message at the bottom of the screen after F5 or F9, when the
/// Saves window is closed and the user would see nothing otherwise.
pub(super) fn save_toast(ctx: &egui::Context, st: &UiState) {
    if st.show_saves {
        return;
    }
    let Some((text, ok, t)) = &st.save_msg else {
        return;
    };
    let left = Duration::from_secs(4).saturating_sub(t.elapsed());
    if left.is_zero() {
        return;
    }
    egui::Area::new(egui::Id::unique("save_toast"))
        .anchor(Align2::CENTER_BOTTOM, [0.0, -48.0])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(RichText::new(text).color(if *ok {
                    theme().colors.good()
                } else {
                    theme().colors.warn()
                }));
            });
        });
    ctx.request_repaint_after(left.min(Duration::from_millis(250)));
}

