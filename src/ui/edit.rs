//! Split out of the former single-file `ui.rs`; see `mod.rs`.

use std::time::{Duration, Instant};

use egui::{Color32, RichText};

use crate::physics::Bodies;
use crate::theme::theme;
use super::*;

/// The Edit menu of the selected-body window.
///
/// Groups: Mass, Orbit, Satellites, Name, Type, Delete. A control that does
/// not apply to this body is greyed out with a hover text that gives the
/// reason, so the user can still find it. Sliders apply on release.
/// Delete needs a second click within four seconds.
pub(super) fn edit_menu(
    ui: &mut egui::Ui,
    st: &mut UiState,
    b: &Bodies,
    i: usize,
    parent: Option<usize>,
    planetish: bool,
    act: &mut Actions,
) {
    egui::CollapsingHeader::new(RichText::new("Edit").strong())
        .id_salt("edit_header")
        .default_open(false)
        .show(ui, |ui| {
            ui.label(
                RichText::new("Sliders apply when you release them.")
                    .small()
                    .color(theme().colors.dim()),
            );
            edit_mass_group(ui, st, b, i, act);
            edit_orbit_group(ui, st, parent.is_some(), planetish, act);
            edit_satellite_group(ui, st, b, i, planetish, act);
            edit_name_group(ui, st, b, i, act);
            edit_type_group(ui, b, i, act);
            edit_delete_group(ui, st, b.id[i], act);
        });
}

/// A slider "commits" when released (or changed without dragging, e.g. by
/// keyboard): the caller applies the factor and the slider snaps back to 1.
pub(super) fn slider_committed(r: &egui::Response) -> bool {
    r.drag_stopped() || (r.changed() && !r.dragged())
}

/// Mass: a ×-factor slider plus quick ÷10 / ÷2 / ×2 / ×10 buttons.
pub(super) fn edit_mass_group(ui: &mut egui::Ui, st: &mut UiState, b: &Bodies, i: usize, act: &mut Actions) {
    let si = st.units_si;
    ui.add_space(4.0);
    ui.label(RichText::new("Mass").strong());
    ui.label(
        RichText::new(format!(
            "{} → {}",
            fmt_mass(b.m[i], si),
            fmt_mass(b.m[i] * st.edit_mass, si)
        ))
        .small()
        .monospace()
        .color(theme().colors.dim()),
    );
    let r = ui.add(
        egui::Slider::new(&mut st.edit_mass, 0.1..=10.0)
            .logarithmic(true)
            .text("×"),
    );
    if slider_committed(&r) {
        if (st.edit_mass - 1.0).abs() > 1e-6 {
            act.edit = Some(Edit::MassMul(st.edit_mass));
        }
        st.edit_mass = 1.0;
    }
    ui.horizontal(|ui| {
        for (label, f) in [("÷10", 0.1), ("÷2", 0.5), ("×2", 2.0), ("×10", 10.0)] {
            if ui.small_button(label).clicked() {
                act.edit = Some(Edit::MassMul(f));
            }
        }
    });
}

/// Orbit: speed scaling relative to the parent, and circularize.
pub(super) fn edit_orbit_group(
    ui: &mut egui::Ui,
    st: &mut UiState,
    has_parent: bool,
    planetish: bool,
    act: &mut Actions,
) {
    ui.add_space(4.0);
    ui.label(RichText::new("Orbit").strong());
    ui.add_enabled_ui(has_parent, |ui| {
        let r = ui
            .add(
                egui::Slider::new(&mut st.edit_vel, 0.25..=2.0)
                    .logarithmic(true)
                    .text("speed ×"),
            )
            .on_hover_text("Scale the speed relative to the parent when released.")
            .on_disabled_hover_text("This body has no parent.");
        if slider_committed(&r) {
            if (st.edit_vel - 1.0).abs() > 1e-6 {
                act.edit = Some(Edit::VelMul(st.edit_vel));
            }
            st.edit_vel = 1.0;
        }
    });
    let can_circularize = planetish && has_parent;
    if ui
        .add_enabled(can_circularize, egui::Button::new("circularize"))
        .on_hover_text("Set the speed for a circular orbit.")
        .on_disabled_hover_text("Only a planet or dwarf planet with a parent can do this.")
        .clicked()
    {
        act.edit = Some(Edit::Circularize);
    }
}

/// Smallest planet mass (M⊕) that can be given moons.
pub(super) const MIN_MASS_FOR_MOONS: f64 = 0.1;
/// Smallest planet mass (M⊕) that can be given a ring.
pub(super) const MIN_MASS_FOR_RING: f64 = 10.0;

/// Satellites: add moons or a ring of small bodies.
pub(super) fn edit_satellite_group(
    ui: &mut egui::Ui,
    st: &mut UiState,
    b: &Bodies,
    i: usize,
    planetish: bool,
    act: &mut Actions,
) {
    ui.add_space(4.0);
    ui.label(RichText::new("Satellites").strong());
    let can_moon = planetish && b.m[i] >= MIN_MASS_FOR_MOONS;
    ui.horizontal(|ui| {
        ui.add_enabled(can_moon, egui::Slider::new(&mut st.edit_moons, 1..=8).text("moons"));
        if ui
            .add_enabled(can_moon, egui::Button::new("add"))
            .on_disabled_hover_text("Needs a planet with mass of 0.1 or more.")
            .clicked()
        {
            act.edit = Some(Edit::SpawnMoon(st.edit_moons));
        }
    });
    let can_ring = planetish && b.m[i] >= MIN_MASS_FOR_RING;
    ui.horizontal(|ui| {
        ui.add_enabled(
            can_ring,
            egui::Slider::new(&mut st.edit_ring, 5..=200).text("ring bodies"),
        );
        if ui
            .add_enabled(can_ring, egui::Button::new("add"))
            .on_disabled_hover_text("Needs a planet with mass of 10 or more.")
            .clicked()
        {
            act.edit = Some(Edit::AddRing(st.edit_ring));
        }
    });
}

/// Name: a text box and a rename button (Enter also renames).
pub(super) fn edit_name_group(ui: &mut egui::Ui, st: &mut UiState, b: &Bodies, i: usize, act: &mut Actions) {
    ui.add_space(4.0);
    ui.label(RichText::new("Name").strong());
    ui.horizontal(|ui| {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut st.edit_name)
                .hint_text("name")
                .desired_width(140.0),
        );
        let enter = resp.lost_focus() && ui.input(|inp| inp.key_pressed(egui::Key::Enter));
        let current = b.name[i].as_deref().unwrap_or("");
        let changed = st.edit_name.trim() != current;
        let clicked = ui
            .add_enabled(changed, egui::Button::new("rename"))
            .on_disabled_hover_text("The name is the same.")
            .clicked();
        if (clicked || (enter && changed)) && act.edit.is_none() {
            act.edit = Some(Edit::Rename(st.edit_name.clone()));
        }
    });
}

/// Type: pick a new body kind. A click on the current type does nothing.
pub(super) fn edit_type_group(ui: &mut egui::Ui, b: &Bodies, i: usize, act: &mut Actions) {
    ui.add_space(4.0);
    ui.label(RichText::new("Type").strong());
    ui.horizontal_wrapped(|ui| {
        for (k, name) in KIND_CHOICES {
            if ui.selectable_label(b.kind[i] == k, name).clicked() && b.kind[i] != k {
                act.edit = Some(Edit::SetKind(k));
            }
        }
    });
}

/// How long the "confirm delete" button stays armed.
pub(super) const DELETE_CONFIRM_WINDOW: Duration = Duration::from_secs(4);

/// Delete needs a second click within `DELETE_CONFIRM_WINDOW`.
pub(super) fn edit_delete_group(ui: &mut egui::Ui, st: &mut UiState, id: u64, act: &mut Actions) {
    let red = Color32::from_rgb(220, 90, 90);
    ui.add_space(6.0);
    ui.separator();
    let armed = matches!(
        st.edit_confirm_delete,
        Some((cid, t)) if cid == id && t.elapsed() < DELETE_CONFIRM_WINDOW
    );
    if armed {
        ui.horizontal(|ui| {
            if ui.button(RichText::new("confirm delete").color(red)).clicked() {
                act.edit = Some(Edit::Delete);
                st.edit_confirm_delete = None;
            }
            if ui.button("cancel").clicked() {
                st.edit_confirm_delete = None;
            }
        });
        // Redraw so the confirm button goes away when the time ends.
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    } else {
        st.edit_confirm_delete = None;
        if ui.button(RichText::new("delete…").color(red)).clicked() {
            st.edit_confirm_delete = Some((id, Instant::now()));
        }
    }
    ui.label(
        RichText::new("Edits start a new energy baseline.")
            .small()
            .color(theme().colors.dim()),
    );
}

