# ============================================================
# UNIQUE motivation plots: JASA/Nature style
# 1. PD-stage stacked proportion plot
# 2. Ordinal symptom heatmaps: 2 x 3 subplots
# 3. Binary clinical burden heatmap
# 4. Continuous burden violin plots
# ============================================================

rm(list = ls())
setwd("U:/UNIQUE/Case study")

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(stringr)

dat <- read_csv("PPMI data/X_set_2_for_exploratory_plots.csv",
                show_col_types = FALSE)

out_dir <- "Output plots motivation"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

dat <- dat %>%
  filter(!is.na(moca)) %>%
  mutate(
    moca_decile = ntile(moca, 10),
    moca_decile_lab = factor(
      moca_decile,
      levels = 1:10,
      labels = c("D1\nlowest", "D2", "D3", "D4", "D5",
                 "D6", "D7", "D8", "D9", "D10\nhighest")
    )
  )

theme_pub <- theme_classic(base_size = 13, base_family = "sans") +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 13, colour = "black"),
    axis.text = element_text(size = 11.5, colour = "black"),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10.5),
    panel.border = element_rect(fill = NA, colour = "grey45", linewidth = 0.45),
    plot.margin = margin(8, 10, 8, 10)
  )

# ============================================================
# 1. PD / HY stage stacked proportion plot
# ============================================================

hy_levels <- dat %>%
  filter(!is.na(hy), hy != 0) %>%
  pull(hy) %>%
  unique() %>%
  sort()

hy_cols <- c(
  "1" = "#FEE08B",
  "2" = "#F46D43",
  "3" = "#A50026"
)

hy_df <- dat %>%
  filter(!is.na(hy), hy != 0, !is.na(moca_decile_lab)) %>%
  mutate(
    hy_stage = factor(
      hy,
      levels = rev(hy_levels),
      labels = c("Advanced\n(HY \u22653)", "Bilateral\n(HY 2)", "Unilateral\n(HY 1)")
    )
  ) %>%
  count(moca_decile_lab, hy_stage) %>%
  group_by(moca_decile_lab) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

fig_hy_stack <- ggplot(
  hy_df,
  aes(x = moca_decile_lab, y = prop, fill = hy_stage)
) +
  geom_col(
    width = 0.82,
    colour = "white",
    linewidth = 0.25
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.01))
  ) +
  scale_fill_manual(
    values = c(
      "Unilateral\n(HY 1)" = "#FEE08B",
      "Bilateral\n(HY 2)" = "#F46D43",
      "Advanced\n(HY \u22653)" = "#A50026"
    ),
    breaks = c("Unilateral\n(HY 1)", "Bilateral\n(HY 2)", "Advanced\n(HY \u22653)"),
    drop = FALSE
  ) +
  guides(
    fill = guide_legend(
      title.position = "left",
      nrow = 1,
      byrow = TRUE
    )
  ) +
  labs(
    x = "Cognitive score decile",
    y = NULL,
    fill = "HY motor stage"
  ) +
  theme_pub +
  theme(
    panel.grid.major.y = element_line(
      colour = "grey85",
      linewidth = 0.25
    ),
    plot.margin = margin(
      t = 2, r = 2, b = 35, l = 2
    ),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.background = element_rect(
      fill = "white",
      colour = "grey45",
      linewidth = 0.45
    ),
    legend.box.background = element_rect(
      fill = NA,
      colour = "grey45",
      linewidth = 0.45
    )
  )

ggsave(file.path(out_dir, "Plot_UNIQUE_PD_stage_stacked_dynamic.png"),
       fig_hy_stack, width = 5.2, height = 5.2, dpi = 600, bg = "white")

ggsave(file.path(out_dir, "Plot_UNIQUE_PD_stage_stacked_dynamic.pdf"),
       fig_hy_stack, width = 5.2, height = 5.2, device = cairo_pdf, bg = "white")

print(fig_hy_stack)

# ============================================================
# 2. Ordinal symptom heatmaps
#    2 x 3 panels, dynamic levels per variable
#    Percentages sum to 100% within each decile
#    "None" excluded from display only
#    (percentages still use full denominator)
# ============================================================

ordinal_vars <- c("NP1COG", "NP1DPRS", "NP1ANXS", "NP1HALL", "NP1FATG")
ordinal_vars <- intersect(ordinal_vars, names(dat))

ordinal_labels <- c(
  NP1COG  = "Cognitive complaint",
  NP1DPRS = "Depression",
  NP1ANXS = "Anxiety",
  NP1HALL = "Hallucinations",
  NP1FATG = "Fatigue"
)

make_level_label <- function(x) {
  x_chr <- as.character(x)
  
  dplyr::case_when(
    x_chr == "0" ~ "None",
    x_chr == "1" ~ "Mild",
    x_chr == "2" ~ "Moderate",
    x_chr == "3" ~ "Severe",
    x_chr == "4" ~ "Very severe",
    TRUE ~ x_chr
  )
}

# ------------------------------------------------------------
# Long format
# ------------------------------------------------------------

ordinal_long <- dat %>%
  select(moca_decile_lab, all_of(ordinal_vars)) %>%
  pivot_longer(
    cols = all_of(ordinal_vars),
    names_to = "variable",
    values_to = "level_raw"
  ) %>%
  filter(
    !is.na(level_raw),
    !is.na(moca_decile_lab)
  ) %>%
  mutate(
    variable_lab = factor(
      ordinal_labels[variable],
      levels = ordinal_labels[ordinal_vars]
    )
  )

# ------------------------------------------------------------
# Compute proportions using FULL denominator
# ------------------------------------------------------------

ordinal_df <- ordinal_long %>%
  count(variable_lab, moca_decile_lab, level_raw) %>%
  group_by(variable_lab, moca_decile_lab) %>%
  mutate(
    prop = n / sum(n)
  ) %>%
  ungroup() %>%
  group_by(variable_lab) %>%
  mutate(
    level_num = suppressWarnings(
      as.numeric(as.character(level_raw))
    ),
    
    level_order = ifelse(
      is.na(level_num),
      dense_rank(as.character(level_raw)),
      level_num
    ),
    
    level_lab_chr = make_level_label(level_raw),
    
    # level 0 starts at bottom
    level_lab = factor(
      level_lab_chr,
      levels = unique(level_lab_chr[order(level_order)])
    )
  ) %>%
  ungroup()

# ------------------------------------------------------------
# Keep "None" category in plotting
# ------------------------------------------------------------

ordinal_plot_df <- ordinal_df

# dynamic legend upper bound
max_prop <- max(ordinal_plot_df$prop, na.rm = TRUE)
legend_upper <- ceiling(max_prop * 10) / 10
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig_ord_heat <- ggplot(
  ordinal_plot_df,
  aes(
    x = moca_decile_lab,
    y = level_lab,
    fill = prop
  )
) +
  geom_tile(
    colour = "white",
    linewidth = 0.35
  ) +
  geom_text(
    aes(
      label = percent(prop, accuracy = 1)
    ),
    size = 3.0,
    colour = "black"
  ) +
  facet_wrap(
    ~ variable_lab,
    nrow = 2,
    ncol = 3,
    scales = "free_y",
    strip.position = "top",
    axes = "all_x",
    axis.labels = "all_x"
  ) +
  scale_fill_gradientn(
    colours = c(
      "#2171B5",  # 0%
      "#6BAED6",  # 1%
      "#C6DBEF",  # 3%
      "#F7FBFF",  # 8%
      "#FFF7BC",  # 20%
      "#FEE391",  # 35%
      "#FEC44F",  # 50%
      "#FB6A4A",  # 70%
      "#B10026"   # 100%
    ),
    values = scales::rescale(c(
      0.00,
      0.01,
      0.03,
      0.08,
      0.20,
      0.50,
      legend_upper
    )),
    limits = c(0, legend_upper),
    labels = percent_format(accuracy = 1),
    name = "Prop. of patients\n(within each decile)"
  ) +
  labs(
    x = "Cognitive score decile",
    y = NULL
  ) +
  theme_classic(
    base_size = 13,
    base_family = "sans"
  ) +
  theme(
    plot.title = element_blank(),
    
    strip.background = element_rect(
      fill = "#F7E3D4",
      colour = "grey35",
      linewidth = 0.45
    ),
    
    strip.text = element_text(
      face = "bold",
      size = 12
    ),
    
    axis.text.x = element_text(
      size = 9
    ),
    
    axis.text.y = element_text(
      size = 10,
      colour = "black"
    ),
    
    axis.title = element_text(
      size = 12
    ),
    
    panel.spacing.y = unit(1.9, "lines"),
    panel.spacing.x = unit(0.8, "lines"),
    
    # prevent clipping/overlap
    plot.margin = margin(
      t = 2, r = 10, b = 2, l = 10
    ),
    
    panel.border = element_rect(
      fill = NA,
      colour = "grey45",
      linewidth = 0.4
    ),
    
    # legend inside empty facet block
    legend.position = c(0.85, 0.25),
    legend.justification = c(0.5, 0.5),
    legend.direction = "vertical",
    
    legend.title = element_text(
      size = 11,
      face = "bold",
      hjust = 0.5
    ),
    
    legend.text = element_text(
      size = 10
    ),
    
    legend.background = element_rect(
      fill = "white",
      colour = "grey45",
      linewidth = 0.45
    ),
    
    legend.key.height = unit(
      1.0,
      "cm"
    ),
    
    legend.key.width = unit(
      0.55,
      "cm"
    )
  ) +
  guides(
    fill = guide_colorbar(
      title.position = "top",
      title.hjust = 0.5,
      frame.colour = NA,
      ticks.colour = "grey50"
    )
  )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

ggsave(
  file.path(
    out_dir,
    "Plot_UNIQUE_ordinal_heatmap_2x3_dynamic.png"
  ),
  fig_ord_heat,
  width = 12.8,
  height = 7.2,
  dpi = 600,
  bg = "white"
)

ggsave(
  file.path(
    out_dir,
    "Plot_UNIQUE_ordinal_heatmap_2x3_dynamic.pdf"
  ),
  fig_ord_heat,
  width = 12.8,
  height = 7.2,
  device = cairo_pdf,
  bg = "white"
)

print(fig_ord_heat)
# ============================================================
# 3. Binary clinical burden heatmap
#    Five binary variables; MCI excluded
# ============================================================

binary_vars <- c(
  "orthostasis",
  "quip_any",
  "pm_wb_any",
  "pm_auto_any",
  "pm_cog_any"
)

binary_vars <- intersect(binary_vars, names(dat))

binary_labels <- c(
  orthostasis = "Orthostasis",
  quip_any    = "Impulse-control\nsymptoms",
  pm_wb_any   = "Well-being\nimpairment",
  pm_auto_any = "Autonomic\nsymptoms",
  pm_cog_any  = "Cognitive\nimpairment"
)

binary_long <- dat %>%
  select(moca_decile_lab, all_of(binary_vars)) %>%
  pivot_longer(
    cols = all_of(binary_vars),
    names_to = "variable",
    values_to = "value"
  ) %>%
  filter(!is.na(value), !is.na(moca_decile_lab)) %>%
  mutate(
    variable_lab = factor(
      binary_labels[variable],
      levels = rev(binary_labels[binary_vars])
    ),
    value01 = as.numeric(value)
  )

binary_df <- binary_long %>%
  group_by(variable_lab, moca_decile_lab) %>%
  summarise(
    prevalence = mean(value01 == 1, na.rm = TRUE),
    .groups = "drop"
  )

binary_upper <- ceiling(max(binary_df$prevalence, na.rm = TRUE) * 20) / 20

fig_binary_heat <- ggplot(
  binary_df,
  aes(x = moca_decile_lab, y = variable_lab, fill = prevalence)
) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_text(
    aes(label = round(100 * prevalence)),
    size = 4.1,
    colour = "black"
  ) +
  scale_fill_gradientn(
    colours = c(
      "white",
      "#FFF7BC",
      "#FEC44F",
      "#FDAE61",
      "#F46D43",
      "#A50026"
    ),
    values = scales::rescale(c(
      0.00,
      0.02,
      0.05,
      0.10,
      0.20,
      binary_upper
    )),
    limits = c(0, binary_upper),
    labels = percent_format(accuracy = 1),
    name = "Prevalence  "
  ) +
  guides(
    fill = guide_colorbar(
      title.position = "left",   # title on left side
      title.hjust = 0.5,
      direction = "horizontal",
      barwidth = unit(4.2, "cm"),
      barheight = unit(0.55, "cm")
    )
  ) +
  scale_y_discrete(
    expand = expansion(mult = c(0.03, 0.08))
  ) +
  labs(
    x = "Cognitive score decile",
    y = NULL
  ) +
  theme_pub +
  theme(
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 11),
    plot.margin = margin(
      t = 4, r = 4, b = 35, l = 2
    ),
    
    legend.position = "top",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.margin = margin(
      t = 4, r = 10, b = 4, l = 10
    ),
    
    # increase spacing between title and colorbar
    legend.box.spacing = unit(8, "pt"),
    
    panel.border = element_rect(
      fill = NA,
      colour = "grey45",
      linewidth = 0.45
    ),
    
    legend.background = element_rect(
      fill = "white",
      colour = "grey45",
      linewidth = 0.45
    ),
    
    legend.box.background = element_rect(
      fill = NA,
      colour = "grey45",
      linewidth = 0.45
    )
  )
ggsave(file.path(out_dir, "Plot_UNIQUE_binary_prevalence_heatmap.png"),
       fig_binary_heat, width = 5.2, height = 5.2, dpi = 600, bg = "white")

ggsave(file.path(out_dir, "Plot_UNIQUE_binary_prevalence_heatmap.pdf"),
       fig_binary_heat, width = 5.2, height = 5.2,
       device = cairo_pdf, bg = "white")

print(fig_binary_heat)

# ============================================================
# 4. Continuous variables: pure violin plots
#    No boxplot, no n labels
#    REM treated continuous; MCI excluded
# ============================================================

continuous_vars <- c("updrs_totscore", "rem")
continuous_vars <- intersect(continuous_vars, names(dat))

continuous_labels <- c(
  updrs_totscore = "UPDRS total score",
  rem = "REM sleep behavior score"
)

cont_long <- dat %>%
  select(moca_decile_lab, all_of(continuous_vars)) %>%
  pivot_longer(
    cols = all_of(continuous_vars),
    names_to = "variable",
    values_to = "value"
  ) %>%
  filter(!is.na(value), !is.na(moca_decile_lab)) %>%
  mutate(
    variable_lab = factor(
      continuous_labels[variable],
      levels = continuous_labels[continuous_vars]
    )
  )

fig_cont_violin <- ggplot(cont_long, aes(x = moca_decile_lab, y = value)) +
  geom_violin(
    aes(fill = variable_lab),
    trim = FALSE,
    alpha = 0.36,
    colour = "#4B2E83",
    linewidth = 0.35
  ) +
  geom_jitter(
    aes(colour = variable_lab),
    width = 0.10,
    height = 0,
    alpha = 0.42,
    size = 0.85
  ) +
  facet_wrap(~ variable_lab, scales = "free_y", nrow = 1) +
  labs(
    x = "Cognitive score decile",
    y = "Clinical burden"
  ) +
  theme_pub +
  theme(
    strip.background = element_rect(fill = "#E7D8F1", colour = "grey35", linewidth = 0.45),
    strip.text = element_text(face = "bold", size = 12),
    legend.position = "none",
    axis.text.x = element_text(size = 9)
  )

ggsave(file.path(out_dir, "Plot_UNIQUE_continuous_violin_dynamic.png"),
       fig_cont_violin, width = 12, height = 4.9, dpi = 600, bg = "white")
ggsave(file.path(out_dir, "Plot_UNIQUE_continuous_violin_dynamic.pdf"),
       fig_cont_violin, width = 12, height = 4.9, device = cairo_pdf, bg = "white")


print(fig_cont_violin)

# ============================================================
# 5. Histogram of MoCA score
#    Read response file directly
# ============================================================

y_dat <- read_csv("PPMI data/Y_set_2.csv", show_col_types = FALSE)

if ("moca" %in% names(y_dat)) {
  y_dat <- y_dat %>% rename(MoCA = moca)
} else {
  names(y_dat)[1] <- "MoCA"
}

moca_median <- median(y_dat$MoCA, na.rm = TRUE)
moca_mean   <- mean(y_dat$MoCA, na.rm = TRUE)

fig_moca_hist <- y_dat %>%
  filter(!is.na(MoCA)) %>%
  ggplot(aes(x = MoCA)) +
  geom_histogram(
    aes(y = after_stat(count)),
    bins = 18,
    fill = "#6B8EAA",
    colour = "white",
    linewidth = 0.35,
    alpha = 0.88
  ) +
  labs(
    x = "MoCA score",
    y = "Number of patients"
  ) +
  theme_pub +
  theme(
    panel.grid.major.y = element_line(
      colour = "grey85",
      linewidth = 0.25
    ),
    axis.title.x = element_text(
      margin = margin(t = 12, b = 0)
    ),
    plot.margin = margin(
      t = 40, r = 2, b = 35, l = 2
    )
  )

ggsave(file.path(out_dir, "Plot_UNIQUE_MoCA_histogram.png"),
       fig_moca_hist, width = 5.2, height = 4.8,
       dpi = 600, bg = "white")

ggsave(file.path(out_dir, "Plot_UNIQUE_MoCA_histogram.pdf"),
       fig_moca_hist, width = 5.2, height = 5.2,
       device = cairo_pdf, bg = "white")

print(fig_moca_hist)

# ============================================================
# Combine saved motivation plots horizontally
# Reads PNG files from Output plots motivation folder
# Order: MoCA histogram -> PD stage -> binary heatmap
# ============================================================

library(magick)

plot_dir <- "Output plots motivation"

img_moca <- image_read(file.path(plot_dir, "Plot_UNIQUE_MoCA_histogram.png"))
img_hy   <- image_read(file.path(plot_dir, "Plot_UNIQUE_PD_stage_stacked_dynamic.png"))
img_bin  <- image_read(file.path(plot_dir, "Plot_UNIQUE_binary_prevalence_heatmap.png"))

# make same height
target_height <- max(
  image_info(img_moca)$height,
  image_info(img_hy)$height,
  image_info(img_bin)$height
)

img_moca <- image_resize(img_moca, paste0("x", target_height))
img_hy   <- image_resize(img_hy,   paste0("x", target_height))
img_bin  <- image_resize(img_bin,  paste0("x", target_height))

# small white spacer
spacer <- image_blank(
  width = 45,
  height = target_height,
  color = "white"
)

combined <- image_append(
  c(img_moca, spacer, img_hy, spacer, img_bin),
  stack = FALSE
)

image_write(
  combined,
  path = file.path(plot_dir, "Plot_UNIQUE_motivation_three_panel_combined.png"),
  format = "png",
  density = 600
)

combined