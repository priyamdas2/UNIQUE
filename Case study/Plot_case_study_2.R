# ============================================================
# Conditional quantile curves by selected predictor
# Single shared legend placed in row 3, column 3
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(cowplot)

setwd("U:/UNIQUE/Case study")

coef_file <- "Output/UNIQUE_coeff_original_matrix_set_2.csv"
x_file    <- "PPMI data/X_set_2_final.csv"
y_file    <- "PPMI data/Y_set_2.csv"

coef_df <- read_csv(coef_file, show_col_types = FALSE)
X <- read_csv(x_file, show_col_types = FALSE)
Y <- read_csv(y_file, show_col_types = FALSE)

if (!"Y" %in% names(Y)) names(Y)[1] <- "Y"

tau_show <- c(0.2, 0.5, 0.8)

selected_vars <- c(
  "HVLTRDLY", "hvlt_discrimination", "HVLTREC",
  "VLTANIM", "SDMTOTAL", "bjlot", "lns"
)

selected_vars <- selected_vars[selected_vars %in% names(X)]

pretty_label <- function(x) {
  dplyr::case_when(
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

get_coef <- function(var, tau_col) {
  out <- coef_df %>% filter(Variable == var) %>% pull(all_of(tau_col))
  if (length(out) == 0) return(0)
  ifelse(is.na(out), 0, out)
}

x_medians <- X %>%
  summarise(across(everything(), ~ median(.x, na.rm = TRUE))) %>%
  as.list()

all_x_vars <- intersect(coef_df$Variable, names(X))

curve_df <- lapply(selected_vars, function(v) {
  
  x_grid <- seq(
    min(X[[v]], na.rm = TRUE),
    max(X[[v]], na.rm = TRUE),
    length.out = 120
  )
  
  lapply(tau_show, function(tt) {
    
    tau_col <- paste0("tau=", tt)
    
    intercept_tau <- get_coef("intercept", tau_col)
    beta_v_tau <- get_coef(v, tau_col)
    
    other_base <- sum(sapply(setdiff(all_x_vars, v), function(z) {
      as.numeric(x_medians[[z]]) * get_coef(z, tau_col)
    }), na.rm = TRUE)
    
    data.frame(
      Variable = v,
      Variable_lab = pretty_label(v),
      x_value = x_grid,
      y_hat = intercept_tau + other_base + beta_v_tau * x_grid,
      tau = tt,
      tau_lab = factor(
        paste0("\u03c4 = ", tt),
        levels = paste0("\u03c4 = ", tau_show)
      )
    )
  }) %>% bind_rows()
}) %>% bind_rows()

point_df <- bind_cols(X, Y) %>%
  select(all_of(selected_vars), Y) %>%
  pivot_longer(
    cols = all_of(selected_vars),
    names_to = "Variable",
    values_to = "x_value"
  ) %>%
  mutate(
    Variable_lab = pretty_label(Variable),
    Variable_lab = factor(Variable_lab, levels = pretty_label(selected_vars))
  )

curve_df <- curve_df %>%
  mutate(
    Variable_lab = factor(Variable_lab, levels = pretty_label(selected_vars))
  )

paper_theme <- theme_minimal(base_size = 12, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.25, colour = "grey88"),
    panel.border = element_rect(colour = "grey25", fill = NA, linewidth = 0.55),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 14.5, face = "plain", margin = margin(t = 7)),
    axis.title.y = element_text(size = 14.5, face = "plain", margin = margin(r = 7)),
    axis.text = element_text(size = 10.5, colour = "grey15"),
    legend.title = element_blank(),
    legend.text = element_text(size = 12, face = "bold"),,
    legend.key.width = unit(1.4, "cm"),
    legend.spacing.x = unit(0.1, "cm"),
    legend.background = element_rect(fill = "white", colour = "grey75", linewidth = 0.7),
    plot.margin = margin(7, 7, 7, 7)
  )

make_panel <- function(v, show_legend = FALSE) {
  
  pp <- point_df %>% filter(Variable == v)
  cc <- curve_df %>% filter(Variable == v)
  
  ggplot() +
    geom_point(
      data = pp,
      aes(x = x_value, y = Y),
      colour = "grey55", #"#6BAED6",
      alpha = 0.22,
      size = 1.7
    ) +
    geom_line(
      data = cc,
      aes(x = x_value, y = y_hat, colour = tau_lab),
      linewidth = 1.05,
      lineend = "round"
    ) +
    scale_colour_manual(values = c("#2166AC", "#67A9CF", "#D6604D")) +
    labs(
      title = pretty_label(v),
      x = paste0(pretty_label(v), " (original scale)"),
      y = "Cognitive score"
    ) +
    paper_theme +
    theme(legend.position = if (show_legend) "bottom" else "none")
}

legend_grob <- cowplot::get_legend(make_panel(selected_vars[1], show_legend = TRUE))
legend_panel <- patchwork::wrap_elements(legend_grob) +
  theme(
    legend.margin = margin(2, 2, 2, 2),
    legend.key.width = unit(1.2, "cm")
  )

plot_list <- lapply(selected_vars, function(v) make_panel(v, show_legend = FALSE))

empty_panel <- ggplot() + theme_void()

row1 <- wrap_plots(plot_list[1:3], nrow = 1)
row2 <- wrap_plots(plot_list[4:6], nrow = 1)

row3 <- wrap_plots(
  list(empty_panel, plot_list[[7]], legend_panel),
  nrow = 1,
  widths = c(1, 1, 1)
)

final_plot <- wrap_plots(row1, row2, row3, ncol = 1) 

out_pdf <- "Output plots/Plot_UNIQUE_conditional_quantile_curves_original.pdf"
out_png <- "Output plots/Plot_UNIQUE_conditional_quantile_curves_original.png"

ggsave(
  filename = out_pdf,
  plot = final_plot,
  width = 13.2,
  height = 10.8,
  units = "in",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  filename = out_png,
  plot = final_plot,
  width = 13.2,
  height = 10.8,
  units = "in",
  dpi = 600,
  bg = "white"
)

print(final_plot)