# ============================================================
# Coefficient profile plot
# General version: any number of selected rows
# At most 3 panels per row; last row centered, same panel size
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(patchwork)

setwd("U:/UNIQUE/Case study")

coef_file <- "Output/UNIQUE_coeff_scaled_matrix_set_2.csv"
x_file    <- "PPMI data/X_set_2_final.csv"
y_file    <- "PPMI data/Y_set_2.csv"

coef_df <- read_csv(coef_file, show_col_types = FALSE)
X <- read_csv(x_file, show_col_types = FALSE)
Y <- read_csv(y_file, show_col_types = FALSE)

sel_thr <- 0.1

tau_cols <- setdiff(names(coef_df), "Variable")
tau_vals <- str_extract(tau_cols, "[0-9.]+") |> as.numeric()

coef_sel <- coef_df %>%
  mutate(mean_abs = rowMeans(abs(across(all_of(tau_cols))), na.rm = TRUE)) %>%
  filter(mean_abs > sel_thr) %>%
  mutate(order_id = row_number()) %>%
  arrange(order_id)

coef_long <- coef_sel %>%
  select(Variable, all_of(tau_cols)) %>%
  pivot_longer(
    cols = all_of(tau_cols),
    names_to = "tau_name",
    values_to = "coef"
  ) %>%
  mutate(
    tau = str_extract(tau_name, "[0-9.]+") |> as.numeric(),
    Variable = factor(Variable, levels = coef_sel$Variable)
  )

pretty_label <- function(x) {
  dplyr::case_when(
    x %in% c("intercept", "(Intercept)", "Intercept") ~ "Intercept",
    x == "HVLTRDLY" ~ "HVLT delayed recall",
    x == "hvlt_discrimination" ~ "HVLT discrimination",
    x == "HVLTREC" ~ "HVLT recognition",
    x == "VLTANIM" ~ "Semantic fluency (animal)",
    x == "SDMTOTAL" ~ "SDMT",
    x == "bjlot" ~ "BJLOT",
    x == "lns" ~ "LNS",
    TRUE ~ str_replace_all(x, "_", " ")
  )
}

is_intercept_name <- function(x) {
  str_to_lower(x) %in% c("intercept", "(intercept)")
}

slope_rng <- coef_long %>%
  filter(!is_intercept_name(as.character(Variable))) %>%
  pull(coef) %>%
  range(na.rm = TRUE)

slope_pad <- 0.06 * diff(slope_rng)
if (!is.finite(slope_pad) || slope_pad < 1e-8) slope_pad <- 0.1
slope_limits <- c(slope_rng[1] - slope_pad, slope_rng[2] + slope_pad)

int_rng <- coef_long %>%
  filter(is_intercept_name(as.character(Variable))) %>%
  pull(coef) %>%
  range(na.rm = TRUE)

if (all(is.finite(int_rng))) {
  int_pad <- 0.06 * diff(int_rng)
  if (!is.finite(int_pad) || int_pad < 1e-8) int_pad <- 0.1
  int_limits <- c(int_rng[1] - int_pad, int_rng[2] + int_pad)
} else {
  int_limits <- slope_limits
}

paper_theme <- theme_minimal(base_size = 12, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.25, colour = "grey88"),
    panel.border = element_rect(colour = "grey25", fill = NA, linewidth = 0.55),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 16, face = "bold", margin = margin(t = 6)),
    axis.title.y = element_text(size = 17, margin = margin(r = 6)),
    axis.text = element_text(size = 14.5, colour = "grey15"),
    axis.ticks = element_line(linewidth = 0.35, colour = "grey25"),
    plot.margin = margin(7, 7, 7, 7)
  )

make_panel <- function(var_name) {
  
  dd <- coef_long %>% filter(as.character(Variable) == var_name)
  is_int <- is_intercept_name(var_name)
  
  line_col <- if (is_int) "#8C2D04" else "#08519C"
  fill_col <- if (is_int) "#FEE6CE" else "#DEEBF7"
  
  ggplot(dd, aes(x = tau, y = coef)) +
    annotate(
      "rect",
      xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
      fill = fill_col, alpha = 0.28
    ) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.35,
      colour = "grey45"
    ) +
    geom_line(
      linewidth = 1.05,
      lineend = "round",
      linejoin = "round",
      colour = line_col
    ) +
    scale_x_continuous(
      breaks = seq(0.1, 0.9, by = 0.1),
      labels = sprintf("%.1f", seq(0.1, 0.9, by = 0.1)),
      limits = c(0.1, 0.9),
      expand = expansion(mult = c(0.015, 0.015))
    ) +
    coord_cartesian(ylim = if (is_int) int_limits else slope_limits) +
    labs(
      title = pretty_label(var_name),
      x = expression(bold(tau)),
      y = expression("Est. coef. (scaled "*italic(x)*")")
    ) +
    paper_theme
}

vars <- as.character(coef_sel$Variable)
plot_list <- lapply(vars, make_panel)

n <- length(plot_list)
ncol_max <- 3
nrow_needed <- ceiling(n / ncol_max)

empty_panel <- ggplot() + theme_void()

rows <- vector("list", nrow_needed)

for (r in seq_len(nrow_needed)) {
  
  idx_start <- (r - 1) * ncol_max + 1
  idx_end   <- min(r * ncol_max, n)
  
  row_plots <- plot_list[idx_start:idx_end]
  n_in_row  <- length(row_plots)
  
  if (n_in_row == 3) {
    
    rows[[r]] <- wrap_plots(
      row_plots,
      nrow = 1,
      widths = c(1, 1, 1)
    )
    
  } else if (n_in_row == 2) {
    
    rows[[r]] <- wrap_plots(
      list(empty_panel, row_plots[[1]], row_plots[[2]], empty_panel),
      nrow = 1,
      widths = c(0.45, 1, 1, 0.45)
    )
    
  } else if (n_in_row == 1) {
    
    rows[[r]] <- wrap_plots(
      list(empty_panel, row_plots[[1]], empty_panel),
      nrow = 1,
      widths = c(1, 1, 1)
    )
  }
}

final_plot <- wrap_plots(
  rows,
  ncol = 1
)

out_pdf <- "Output plots/Plot_UNIQUE_case_study_selected_coeff_profiles.pdf"
out_png <- "Output plots/Plot_UNIQUE_case_study_selected_coeff_profiles.png"

ggsave(
  filename = out_pdf,
  plot = final_plot,
  width = 13.2,
  height = 3.8 * nrow_needed,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = out_png,
  plot = final_plot,
  width = 13.2,
  height = 3.8 * nrow_needed,
  units = "in",
  dpi = 600,
  bg = "white"
)

print(final_plot)

# ============================================================
# Overlayed heterogeneity plot
# beta_j(tau) - beta_j(0.5), excluding intercept
# ============================================================

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

coef_long_full <- coef_df %>%
  filter(!is_intercept_name(Variable)) %>%
  select(Variable, all_of(tau_cols)) %>%
  pivot_longer(
    cols = all_of(tau_cols),
    names_to = "tau_name",
    values_to = "coef"
  ) %>%
  mutate(
    tau = str_extract(tau_name, "[0-9.]+") |> as.numeric()
  )

coef_mid <- coef_long_full %>%
  filter(abs(tau - 0.5) < 1e-8) %>%
  select(Variable, coef_mid = coef)

coef_het <- coef_long_full %>%
  left_join(coef_mid, by = "Variable") %>%
  mutate(
    delta = coef - coef_mid,
    Variable_lab = pretty_label(Variable)
  ) %>%
  filter(Variable %in% coef_sel$Variable) %>%
  filter(!is_intercept_name(as.character(Variable))) %>%
  mutate(
    Variable_lab = factor(
      Variable_lab,
      levels = pretty_label(as.character(coef_sel$Variable[
        !is_intercept_name(as.character(coef_sel$Variable))
      ]))
    )
  )

het_palette <- c(
  "HVLT delayed recall"       = "#0072B2",  # blue
  "HVLT discrimination"       = "#E69F00",  # orange
  "HVLT recognition"          = "#009E73",  # green
  "Semantic fluency (animal)" = "#D55E00",  # vermillion
  "SDMT"                      = "#CC79A7",  # magenta
  "BJLOT"                     = "#56B4E9",  # sky blue
  "LNS"                       = "#000000"   # black (strong anchor)
)
het_overlay_plot <- ggplot(
  coef_het,
  aes(x = tau, y = delta, colour = Variable_lab)
) +
  annotate(
    "rect",
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
    fill = "#F7F7F7", alpha = 0.65
  ) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.45,
    colour = "grey35"
  ) +
  geom_line(
    linewidth = 1.25,
    lineend = "round",
    linejoin = "round"
  ) +
  scale_x_continuous(
    breaks = seq(0.1, 0.9, by = 0.1),
    labels = sprintf("%.1f", seq(0.1, 0.9, by = 0.1)),
    limits = c(0.1, 0.9),
    expand = expansion(mult = c(0.015, 0.015))
  ) +
  scale_colour_manual(values = het_palette, drop = FALSE) +
  labs(
    x = expression(bold(tau)),
    y = expression(beta[j](tau) - beta[j](0.5)),
    colour = NULL
  ) +
  theme_minimal(base_size = 13, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.20, colour = "grey90"),
    panel.grid.major.y = element_line(linewidth = 0.25, colour = "grey86"),
    panel.border = element_rect(colour = "grey25", fill = NA, linewidth = 0.60),
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 8)),
    axis.title.y = element_text(size = 20, face = "plain", margin = margin(r = 2)),
    axis.text = element_text(size = 13, colour = "grey12"),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.text = element_text(size = 12.5, face = "bold"),
    legend.key.width = unit(1.45, "cm"),
    legend.spacing.x = unit(0.25, "cm"),
    legend.background = element_rect(fill = "white", colour = "grey75", linewidth = 0.35),
    plot.margin = margin(8, 10, 8, 10)
  ) +
  guides(
    colour = guide_legend(
      nrow = 2,
      byrow = TRUE,
      override.aes = list(linewidth = 1.8)
    )
  )

out_pdf <- "Output plots/Plot_UNIQUE_case_study_heterogeneity_overlay.pdf"
out_png <- "Output plots/Plot_UNIQUE_case_study_heterogeneity_overlay.png"

ggsave(
  filename = out_pdf,
  plot = het_overlay_plot,
  width = 13.2,
  height = 4,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = out_png,
  plot = het_overlay_plot,
  width = 13.2,
  height = 4.2,
  units = "in",
  dpi = 600,
  bg = "white"
)

print(het_overlay_plot)


# ============================================================
# Grouped bar plot: selected slope coefficients at tau = 0.2, 0.5, 0.8
# No intercept; wide paper-ready figure
# ============================================================

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

pretty_label_2 <- function(x) {
  dplyr::case_when(
    x %in% c("intercept", "(Intercept)", "Intercept") ~ "Intercept",
    x == "HVLTRDLY" ~ "HVLT\ndelayed\nrecall",
    x == "hvlt_discrimination" ~ "HVLT\ndiscrimination",
    x == "HVLTREC" ~ "HVLT\nrecognition",
    x == "VLTANIM" ~ "Semantic\nfluency\n(animal)",
    x == "SDMTOTAL" ~ "SDMT",
    x == "bjlot" ~ "BJLOT",
    x == "lns" ~ "LNS",
    TRUE ~ str_replace_all(x, "_", "\n")
  )
}

tau_show <- c(0.2, 0.5, 0.8)

slope_vars <- as.character(coef_sel$Variable[
  !is_intercept_name(as.character(coef_sel$Variable))
])

bar_df <- coef_long %>%
  filter(!is_intercept_name(as.character(Variable))) %>%
  filter(tau %in% tau_show) %>%
  mutate(
    tau_lab = factor(
      paste0("\u03c4 = ", tau),
      levels = paste0("\u03c4 = ", tau_show)
    ),
    Variable_lab = pretty_label_2(as.character(Variable)),
    Variable_lab = factor(
      Variable_lab,
      levels = pretty_label_2(slope_vars)
    )
  )

bar_theme <- theme_minimal(base_size = 13, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.25, colour = "grey86"),
    panel.border = element_rect(colour = "grey25", fill = NA, linewidth = 0.55),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 18, margin = margin(r = 2)),
    axis.text.x = element_text(
      size = 15,
      angle = 0,
      face = "bold",
      hjust = 0.5,
      vjust = 0.5,
      lineheight = 0.9
    ),
    axis.text.y = element_text(size = 12, colour = "grey15"),
    legend.title = element_blank(),
    legend.text = element_text(size = 16),
    legend.position = c(0.50, 0.85),
    legend.background = element_rect(fill = "white", colour = "grey75", linewidth = 0.55),
    plot.margin = margin(8, 10, 8, 10)
  )

bar_plot <- ggplot(bar_df, aes(x = Variable_lab, y = coef, fill = tau_lab)) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.68,
    colour = "white",
    linewidth = 0.25
  ) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.35,
    colour = "grey45"
  ) +
  scale_fill_manual(values = c("#2166AC", "#67A9CF", "#D6604D")) +
  labs(
    x = NULL,
    y = expression("Est. coef. (scaled "*italic(x)*")")
  ) +
  bar_theme

out_pdf_bar <- "Output plots/Plot_UNIQUE_case_study_bar_tau_020508.pdf"
out_png_bar <- "Output plots/Plot_UNIQUE_case_study_bar_tau_020508.png"

ggsave(
  filename = out_pdf_bar,
  plot = bar_plot,
  width = 13.2,
  height = 4.2,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = out_png_bar,
  plot = bar_plot,
  width = 13.2,
  height = 4.2,
  units = "in",
  dpi = 600,
  bg = "white"
)

print(bar_plot)

# ============================================================
# Bootstrap CI forest plot at representative quantiles
# File: Output/UNIQUE_selected_BOOT_CI_<confidence_level>_summary_original_set_2.csv
# User input: confidence_level
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

confidence_level <- 95   # <-- change this, e.g., 90, 95

boot_file <- paste0(
  "Output/UNIQUE_selected_BOOT_CI_",
  confidence_level,
  "_summary_original_set_2.csv"
)

boot_df <- read_csv(boot_file, show_col_types = FALSE)

pretty_label_3 <- function(x) {
  dplyr::case_when(
    x %in% c("intercept", "(Intercept)", "Intercept") ~ "Intercept",
    x == "HVLTRDLY" ~ "HVLT\ndelayed\nrecall",
    x == "hvlt_discrimination" ~ "HVLT\ndiscrim.",
    x == "HVLTREC" ~ "HVLT\nrecog.",
    x == "VLTANIM" ~ "Semantic\nfluency\n(animal)",
    x == "SDMTOTAL" ~ "SDMT",
    x == "bjlot" ~ "BJLOT",
    x == "lns" ~ "LNS",
    TRUE ~ str_replace_all(x, "_", "\n")
  )
}

is_intercept_name <- function(x) {
  str_to_lower(x) %in% c("intercept", "(intercept)")
}

tau_show <- c(0.2, 0.5, 0.8)
tau_cols_show <- paste0("tau=", tau_show)

slope_vars <- boot_df$Variable[
  !is_intercept_name(boot_df$Variable)
]

forest_df <- boot_df %>%
  filter(!is_intercept_name(Variable)) %>%
  select(Variable, all_of(tau_cols_show)) %>%
  pivot_longer(
    cols = all_of(tau_cols_show),
    names_to = "tau_name",
    values_to = "summary"
  ) %>%
  mutate(
    tau = as.numeric(str_extract(tau_name, "[0-9.]+")),
    estimate = as.numeric(str_match(summary, "^\\s*([-0-9.]+)\\s*\\(")[, 2]),
    lower = as.numeric(str_match(summary, "\\(([-0-9.]+),")[, 2]),
    upper = as.numeric(str_match(summary, ",\\s*([-0-9.]+)\\)")[, 2]),
    Variable_lab = pretty_label_3(Variable),
    Variable_lab = factor(
      Variable_lab,
      levels = rev(pretty_label_3(slope_vars))
    ),
    tau_lab = factor(
      paste0("\u03c4 = ", tau),
      levels = paste0("\u03c4 = ", tau_show)
    )
  )

forest_theme <- theme_minimal(base_size = 13, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.25, colour = "grey86"),
    panel.border = element_rect(colour = "grey25", fill = NA, linewidth = 0.55),
    strip.text = element_text(size = 15, face = "bold"),
    axis.title.x = element_text(size = 19, margin = margin(t = 8)),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 12, colour = "grey15"),
    axis.text.y = element_text(size = 12, colour = "grey25", lineheight = 0.9, face = "bold"),
    legend.position = "none",
    plot.margin = margin(8, 10, 8, 10)
  )

forest_plot <- ggplot(
  forest_df,
  aes(x = estimate, y = Variable_lab, colour = tau_lab)
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.4,
    colour = "grey45"
  ) +
  geom_errorbarh(
    aes(xmin = lower, xmax = upper),
    height = 0,
    linewidth = 0.95
  ) +
  geom_point(size = 3.1) +
  facet_wrap(~ tau_lab, nrow = 1) +
  scale_colour_manual(values = c("#2166AC", "#67A9CF", "#D6604D")) +
  scale_x_continuous(
    breaks = seq(0, 0.6, by = 0.1),
    limits = c(0, 0.6)
  ) +
  labs(
    x = "Coefficient estimated (original scale)",
    y = NULL
  ) +
  forest_theme

out_pdf_forest <- paste0(
  "Output plots/Plot_UNIQUE_bootstrap_CI_",
  confidence_level,
  "_forest_tau_020508.pdf"
)

out_png_forest <- paste0(
  "Output plots/Plot_UNIQUE_bootstrap_CI_",
  confidence_level,
  "_forest_tau_020508.png"
)

ggsave(
  filename = out_pdf_forest,
  plot = forest_plot,
  width = 13.2,
  height = 4.2,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = out_png_forest,
  plot = forest_plot,
  width = 13.2,
  height = 4.2,
  units = "in",
  dpi = 600,
  bg = "white"
)

print(forest_plot)

# ============================================================
# FINAL COMBINED FIGURE
# Combine directly from already-saved PNG files
# with true vertical white spacing between panels
# ============================================================

library(png)
library(grid)
library(gridExtra)

# ------------------------------------------------------------
# Read saved PNG files
# ------------------------------------------------------------

img_forest <- rasterGrob(
  readPNG(
    sprintf(
      "Output plots/Plot_UNIQUE_bootstrap_CI_%d_forest_tau_020508.png",
      confidence_level
    )
  ),
  interpolate = TRUE
)

img_bar <- rasterGrob(
  readPNG("Output plots/Plot_UNIQUE_case_study_bar_tau_020508.png"),
  interpolate = TRUE
)

img_het <- rasterGrob(
  readPNG("Output plots/Plot_UNIQUE_case_study_heterogeneity_overlay.png"),
  interpolate = TRUE
)

# ------------------------------------------------------------
# Spacer rows = actual vertical white space
# (increase values for more separation)
# ------------------------------------------------------------

final_jasa_combined <- grid.arrange(
  img_forest,
  nullGrob(),
  img_bar,
  nullGrob(),
  img_het,
  
  ncol = 1,
  
  heights = unit.c(
    unit(4.2, "null"),   # forest
    unit(0.15, "null"),  # gap
    unit(4.2, "null"),   # bar
    unit(0.15, "null"),  # gap
    unit(4.2, "null"),    # heterogeneity
    unit(0.27, "null")  # gap
  )
)

# ------------------------------------------------------------
# Save final combined figure
# ------------------------------------------------------------

ggsave(
  filename = "Output plots/Plot_UNIQUE_combined_main_realdata_figure.pdf",
  plot = final_jasa_combined,
  width = 13.2,
  height = 14.4,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = "Output plots/Plot_UNIQUE_combined_main_realdata_figure.png",
  plot = final_jasa_combined,
  width = 13.2,
  height = 14.4,
  units = "in",
  dpi = 600,
  bg = "white"
)