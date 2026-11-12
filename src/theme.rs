//! Typography and colour theme.
//!
//! Fonts are embedded in the binary. The theme is read from
//! `config/ui.json` at build time. Nothing is read from disk at run time.

use egui::Color32;

/// The four embedded font files.
pub struct EmbeddedFonts {
    pub mono_regular: &'static [u8],
    pub mono_bold: &'static [u8],
    pub sans_regular: &'static [u8],
    pub sans_medium: &'static [u8],
}

/// If a file is missing, `cargo build` fails with "couldn't read ...".
/// That is the expected failure. Put the four .ttf files in `assets/fonts/`.
pub fn embedded_fonts() -> EmbeddedFonts {
    EmbeddedFonts {
        mono_regular: include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf"),
        mono_bold: include_bytes!("../assets/fonts/JetBrainsMono-Bold.ttf"),
        sans_regular: include_bytes!("../assets/fonts/Inter-Regular.ttf"),
        sans_medium: include_bytes!("../assets/fonts/Inter-Medium.ttf"),
    }
}

/// Register the fonts. JetBrains Mono leads the Monospace family and Inter
/// leads the Proportional family. egui's built-in fonts stay behind them,
/// so the symbols in the stats panel (★ ☾ ✦ ✹ ● ⊕ ☉ ☰) still render.
pub fn install_fonts(ctx: &egui::Context) {
    let f = embedded_fonts();
    let mut fonts = egui::FontDefinitions::default();
    // egui 0.30 and older store `FontData` directly. From egui 0.31 the map
    // holds `Arc<FontData>`: wrap each value in `std::sync::Arc::new(..)`.
    fonts.font_data.insert(
        "jbm-regular".to_owned(),
        egui::FontData::from_static(f.mono_regular),
    );
    fonts.font_data.insert(
        "jbm-bold".to_owned(),
        egui::FontData::from_static(f.mono_bold),
    );
    fonts.font_data.insert(
        "inter-regular".to_owned(),
        egui::FontData::from_static(f.sans_regular),
    );
    fonts.font_data.insert(
        "inter-medium".to_owned(),
        egui::FontData::from_static(f.sans_medium),
    );
    if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        mono.insert(0, "jbm-regular".to_owned());
        mono.insert(1, "jbm-bold".to_owned());
    }
    if let Some(prop) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        prop.insert(0, "inter-regular".to_owned());
        prop.insert(1, "inter-medium".to_owned());
    }
    ctx.set_fonts(fonts);
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct UiTheme {
    pub colors: ThemeColors,
    pub spacing: ThemeSpacing,
    pub fonts: ThemeFonts,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ThemeColors {
    pub panel_fill: [u8; 4],
    pub window_fill: [u8; 4],
    pub accent: [u8; 3],
    pub warn: [u8; 3],
    pub good: [u8; 3],
    pub dim: [u8; 3],
    pub selection_bg: [u8; 3],
    pub text: [u8; 3],
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ThemeSpacing {
    pub item_spacing: [f32; 2],
    pub button_padding: [f32; 2],
    pub window_margin: f32,
    pub window_rounding: f32,
    pub panel_margin: f32,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ThemeFonts {
    pub body_size: f32,
    pub small_size: f32,
    pub heading_size: f32,
    pub mono_size: f32,
}

impl ThemeColors {
    pub fn panel_fill(&self) -> Color32 {
        let c = self.panel_fill;
        Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3])
    }
    pub fn window_fill(&self) -> Color32 {
        let c = self.window_fill;
        Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3])
    }
    pub fn accent(&self) -> Color32 {
        Color32::from_rgb(self.accent[0], self.accent[1], self.accent[2])
    }
    pub fn warn(&self) -> Color32 {
        Color32::from_rgb(self.warn[0], self.warn[1], self.warn[2])
    }
    pub fn good(&self) -> Color32 {
        Color32::from_rgb(self.good[0], self.good[1], self.good[2])
    }
    pub fn dim(&self) -> Color32 {
        Color32::from_rgb(self.dim[0], self.dim[1], self.dim[2])
    }
    pub fn selection_bg(&self) -> Color32 {
        let c = self.selection_bg;
        Color32::from_rgb(c[0], c[1], c[2])
    }
    pub fn text(&self) -> Color32 {
        Color32::from_rgb(self.text[0], self.text[1], self.text[2])
    }
}

/// The theme from `config/ui.json`, parsed once.
pub fn theme() -> &'static UiTheme {
    static T: std::sync::OnceLock<UiTheme> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        serde_json::from_str(include_str!("../config/ui.json"))
            .expect("config/ui.json is malformed; see the source for the schema")
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn theme_json_parses() {
        let t = crate::theme::theme();
        assert!(t.fonts.body_size >= 10.0 && t.fonts.body_size <= 20.0);
        assert!(t.colors.panel_fill[3] > 0, "panel_fill must be visible");
        assert!(t.spacing.item_spacing[0] > 0.0);
    }

    #[test]
    fn fonts_are_nonempty() {
        let f = crate::theme::embedded_fonts();
        assert!(f.mono_regular.len() > 10_000);
        assert!(f.mono_bold.len() > 10_000);
        assert!(f.sans_regular.len() > 10_000);
        assert!(f.sans_medium.len() > 10_000);
    }
}
