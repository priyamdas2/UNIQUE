# ============================================================
# Univariable QR vs QR-LASSO coefficient table
# PPMI curated baseline data, multiple quantile levels
# ============================================================
setwd("U:/UNIQUE/Motivating study")
rm(list = ls())

library(dplyr)
library(tidyr)
library(readxl)
library(readr)
library(purrr)
library(tibble)
library(quantreg)
library(rqPen)
library(broom)

set.seed(1)
# ------------------------------------------------------------
# User-editable inputs
# ------------------------------------------------------------

# Put this R script in the same folder as:
#   PPMI_Curated_Data_Cut_Public_20250321.xlsx
# Output CSV will also be saved in this same folder.

tau_grid <- c(0.25, 0.5, 0.75)

# If ALL_vars = 1: QR-LASSO regression uses all variables in data extraction below
# If ALL_vars = 0: QR-LASSO regression uses only selected_vars below
# In both cases, saved coefficient table keeps only selected_vars
ALL_vars <- 1

excel_file <- "PPMI_Curated_Data_Cut_Public_20250321.xlsx"
excel_sheet <- "20250310"

selected_vars <- c(
  "hvlt_immediaterecall", "HVLTRDLY", "hvlt_discrimination", "HVLTFPRL",
  "HVLTREC", "hvlt_retention", "VLTANIM", "SDMTOTAL", "bjlot", "lns",
  "gds", "upsit", "pigd", "rem", "scopa",
  "updrs2_score", "updrs3_score", "MCI_testscores", "age"
)

run_tag <- ifelse(ALL_vars == 1, "reg_all_vars", "reg_selected_vars")

# ------------------------------------------------------------
# Read and prepare data using curated PPMI baseline extraction
# ------------------------------------------------------------

Curated_data <- read_excel(excel_file, sheet = excel_sheet)

# Select baseline data
Curated_data_BL <- Curated_data[Curated_data$EVENT_ID == "BL", ]

# Make ID numeric
Curated_data_BL$PATNO <- as.numeric(Curated_data_BL$PATNO)

# Extract outcome and candidate predictors
data <- Curated_data_BL %>%
  dplyr::select(
    PATNO,
    Y = moca,
    upsit, quip, ess, gds, scopa, lns,
    abeta, tau, ptau, asyn,
    mean_putamen, mean_caudate, mean_striatum,
    hvlt_discrimination, hvlt_immediaterecall, hvlt_retention,
    HVLTFPRL, HVLTRDLY, HVLTREC,
    VLTANIM, SDMTOTAL, bjlot, stai, rem,
    updrs1_score, updrs2_score, updrs3_score,
    nfl_serum, urate,
    age, EDUCYRS, APOE_e4, pigd
  ) %>%
  dplyr::select(-PATNO) %>%
  as.data.frame()

# Complete-case data, same as attached workflow
data <- tidyr::drop_na(data)

# ------------------------------------------------------------
# Remove all-NA and zero-variance predictors
# ------------------------------------------------------------

data <- data %>%
  dplyr::select(where(~ !all(is.na(.x))))

predictors_all <- setdiff(names(data), "Y")

nonzero_var <- predictors_all[
  sapply(data[predictors_all], function(z) {
    if (is.numeric(z)) {
      sd(z, na.rm = TRUE) > 0
    } else {
      length(unique(z[!is.na(z)])) > 1
    }
  })
]

data <- data %>%
  dplyr::select(Y, all_of(nonzero_var))

# ------------------------------------------------------------
# Choose variables used in regression
# ------------------------------------------------------------

available_selected_vars <- intersect(selected_vars, names(data))

missing_selected_vars <- setdiff(selected_vars, names(data))
if (length(missing_selected_vars) > 0) {
  warning(
    "These selected variables are not available after preprocessing: ",
    paste(missing_selected_vars, collapse = ", ")
  )
}

# ------------------------------------------------------------
# Save processed dataset for downstream methods
# ------------------------------------------------------------

# Save exactly the dataset after preprocessing
# (after complete-case filtering + zero-variance removal)

write_csv(
  data,"Processed_dataset.csv")


if (ALL_vars == 1) {
  predictors <- setdiff(names(data), "Y")
} else {
  predictors <- available_selected_vars
  data <- data %>%
    dplyr::select(Y, all_of(predictors))
}

# Save only selected variables in either mode
save_vars <- intersect(selected_vars, names(data))

cat("\nRegression mode:", run_tag, "\n")
cat("Complete-case sample size:", nrow(data), "\n")
cat("Number of predictors used in QR-LASSO:", length(predictors), "\n")
cat("Number of variables saved:", length(save_vars), "\n")

# ------------------------------------------------------------
# Function for one quantile
# ------------------------------------------------------------

fit_one_tau <- function(tau_val) {
  
  # ==========================================================
  # 1. Univariable quantile regression
  #    Fit one predictor at a time using the selected regression dataset
  # ==========================================================
  
  uni_results <- map_dfr(predictors, function(var) {
    
    formula_j <- as.formula(paste("Y ~", var))
    
    fit_j <- tryCatch(
      rq(formula_j, tau = tau_val, data = data),
      error = function(e) NULL
    )
    
    if (is.null(fit_j)) {
      return(tibble(predictor = var, uni_coef = NA_real_))
    }
    
    coef_j <- coef(fit_j)
    beta_j <- coef_j[names(coef_j) != "(Intercept)"]
    
    tibble(
      predictor = var,
      uni_coef = as.numeric(beta_j[1])
    )
  })
  
  # ==========================================================
  # 2. QR-LASSO using all or selected variables, depending on ALL_vars
  # ==========================================================
  
  x <- model.matrix(
    as.formula(paste("Y ~", paste(predictors, collapse = " + "))),
    data = data
  )[, -1, drop = FALSE]
  
  y <- data$Y
  
  set.seed(123)
  
  cv_q_lasso <- rq.pen.cv(
    x = x,
    y = y,
    tau = tau_val,
    penalty = "LASSO",
    nfolds = 10
  )
  
  idx_min <- which.min(cv_q_lasso$cverr)
  lasso_coef <- coef(cv_q_lasso$fit)[, idx_min]
  
  lasso_coef <- lasso_coef[
    !names(lasso_coef) %in% c("(Intercept)", "Intercept", "intercept")
  ]
  
  lasso_table <- tibble(
    predictor = names(lasso_coef),
    lasso_coef = as.numeric(lasso_coef)
  ) %>%
    filter(
      !is.na(predictor),
      predictor != "",
      !tolower(predictor) %in% c("(intercept)", "intercept")
    )
  
  # ==========================================================
  # 3. Merge; save filtering happens after all tau values are run
  # ==========================================================
  
  uni_results %>%
    full_join(lasso_table, by = "predictor") %>%
    mutate(
      tau = tau_val,
      univariate_QR_slope = round(uni_coef, 3),
      lasso_QR_slope = round(lasso_coef, 3)
    ) %>%
    select(
      tau,
      variable = predictor,
      univariate_QR_slope,
      lasso_QR_slope
    )
}

# ------------------------------------------------------------
# Run for all quantiles and save only selected variables
# ------------------------------------------------------------

qr_table_all <- map_dfr(tau_grid, fit_one_tau) %>%
  filter(!tolower(variable) %in% c("(intercept)", "intercept"))

qr_table_save <- qr_table_all %>%
  filter(variable %in% save_vars)

# ------------------------------------------------------------
# Save one combined CSV in the same folder as this script/data
# ------------------------------------------------------------

out_file <- paste0(
  "Output/QR_univariate_vs_lasso_all_tau_",
  run_tag,
  "_save_selected_vars.csv"
)

write_csv(qr_table_save, out_file)

print(qr_table_save, n = Inf)

cat("\nSaved output file:\n", normalizePath(out_file), "\n")