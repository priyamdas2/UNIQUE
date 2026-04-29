rm(list = ls())
setwd("U:/UNIQUE/Case study/PPMI data")

library(dplyr)
library(tidyr)
library(readxl)
library(readr)
library(reshape2)
library(broom)
library(purrr)
library(glmnet)
library(quantreg)
library(rqPen)

drop_option <- 2  # use dataset 2

# Read the data
Curated_data <- read_excel(
  "PPMI_Curated_Data_Cut_Public_20250321.xlsx",
  sheet = "20250310"
)

# Select only baseline data
Curated_data_BL <- Curated_data[Curated_data$EVENT_ID == "BL", ]


# Keep only Idiopathic PD
Curated_data_BL <- Curated_data_BL %>%
  dplyr::filter(PRIMDIAG == 1)

table(Curated_data_BL$PRIMDIAG)

# Make the id numeric
Curated_data_BL$PATNO <- as.numeric(Curated_data_BL$PATNO)

# Variables that may be dropped later due to heavy missingness / design choice
to_be_dropped_1 <- c(
  "MSEADLG",
  "urate",
  "mean_caudate",
  "mean_putamen",
  "mean_striatum"
)

to_be_dropped_2 <- c(to_be_dropped_1,
  "abeta",
  "asyn",
  "ptau",
  "tau"
)

to_be_dropped_3 <- c(to_be_dropped_2,
  "hvlt_immediaterecall",
  "HVLTRDLY",
  "hvlt_discrimination",
  "HVLTFPRL",
  "HVLTREC",
  "hvlt_retention",
  "VLTANIM",
  "SDMTOTAL",
  "bjlot",
  "lns",
  "MCI_testscores"
)




# Choose drop set based on option
if (drop_option == 1) {
  to_be_dropped <- to_be_dropped_1
} else if (drop_option == 2) {
  to_be_dropped <- to_be_dropped_2
} else if (drop_option == 3) {
  to_be_dropped <- to_be_dropped_3
} else {
  stop("drop_option must be either 1, 2 or 3.")
}

clinical_covariates <- c(
  "hvlt_immediaterecall",
  "HVLTRDLY",
  "hvlt_discrimination",
  "HVLTFPRL",
  "HVLTREC",
  "hvlt_retention",
  "VLTANIM",
  "SDMTOTAL",
  "bjlot",
  "lns",
  "ess",
  "gds",
  "upsit",
  "stai",
  "pigd",
  "quip",
  "MSEADLG",
  "rem",
  "scopa",
  "updrs1_score",
  "updrs2_score",
  "updrs3_score",
  "MCI_testscores"
)

biologic_covariates <- c(
  "abeta",
  "asyn",
  "ptau",
  "tau",
  "urate"
)

datscan_covariates <- c(
  "mean_caudate",
  "mean_putamen",
  "mean_striatum"
)

genetic_covariates <- c(
  "APOE_e4"
)

demographic_covariates <- c(
  "age",
  "race",
  "SEX",
  "fampd_bin",
  "handed"
)

##############################################################
# Combine all predictor names

all_covariates <- c(
  clinical_covariates,
  biologic_covariates,
  datscan_covariates,
  genetic_covariates,
  demographic_covariates
)

# Keep only variables that actually exist in the dataset
available_covariates <- intersect(all_covariates, names(Curated_data_BL))

# Optional: check if any requested variables are missing from dataset
missing_covariates <- setdiff(all_covariates, names(Curated_data_BL))
missing_covariates

##############################################################
# Create analysis dataset with all available selected variables

data_full <- Curated_data_BL %>%
  dplyr::select(PATNO, Y = moca, all_of(available_covariates))

# Remove ID
data_full <- data_full %>%
  dplyr::select(-PATNO)

data_full <- data.frame(data_full)

summary(data_full$Y)
dim(data_full)
head(data_full)

############################################
# Check: Non-missing, Missing, Total

get_summary_table <- function(data) {
  data.frame(
    Variable    = colnames(data),
    Non_Missing = colSums(!is.na(data)),
    Missing     = colSums(is.na(data)),
    Total       = nrow(data)
  )
}

summary_full <- get_summary_table(data_full)

summary_full
summary_full[, -1]

##############################################################
# Complete-case data using all selected variables "as is"

data_full_cc <- drop_na(data_full)   # take complete data using all variables

cat("\nDimension of full selected data (before drop_na):\n")
print(dim(data_full))

cat("\nDimension after drop_na on full selected data:\n")
print(dim(data_full_cc))

head(data_full_cc)
str(data_full_cc)

##############################################################
# Remove selected drop-set variables first, then remove NA entries

# Keep only those drop-candidates that are actually present
to_be_dropped_available <- intersect(to_be_dropped, names(data_full))

# Data after removing selected variables
data_reduced <- data_full %>%
  dplyr::select(-all_of(to_be_dropped_available))

data_reduced <- data.frame(data_reduced)

# Missingness summary after dropping selected variables
summary_reduced <- get_summary_table(data_reduced)

summary_reduced
summary_reduced[, -1]

retained_row_numbers <- which(stats::complete.cases(data_reduced))

# Complete-case data after removing the selected variables
data_reduced_cc <- drop_na(data_reduced)

cat("\nDrop option selected:\n")
print(drop_option)

cat("\nVariables removed before complete-case filtering:\n")
print(to_be_dropped_available)

cat("\nDimension of reduced data (before drop_na):\n")
print(dim(data_reduced))

cat("\nDimension after dropping selected variables and then drop_na:\n")
print(dim(data_reduced_cc))

head(data_reduced_cc)
str(data_reduced_cc)

##############################################################
# Final modeling data
# Choose which version you want to use:
#   1. data_full_cc    = complete-case data using all selected variables
#   2. data_reduced_cc = complete-case data after dropping selected vars

data <- data_reduced_cc

X <- data[, -1]
Y <- data[,  1]

cat("\nFinal modeling data dimension:\n")
print(dim(data))

head(data)
str(data)

##############################################################
# Save final Y and X based on selected drop_option

# Save Y
write.csv(
  data.frame(Y = Y),
  paste0("Y_set_", drop_option, ".csv"),
  row.names = FALSE
)

# Save original X (keep variable names)
write.csv(
  X,
  paste0("X_set_", drop_option, ".csv"),
  row.names = FALSE
)

##############################################################
# Create modified X
# (i) replace race with race_modified:
#     1 stays 1, all other values become 0
# (ii) replace handed with two binary columns:
#      handed_right = 1 if handed == 1, else 0
#      handed_left  = 1 if handed == 2, else 0

X_final <- X

# Modify race
if ("race" %in% names(X_final)) {
  X_final$race_modified <- ifelse(X_final$race == 1, 1, 0)
  X_final$race <- NULL
}

# Modify handed
if ("handed" %in% names(X_final)) {
  X_final$handed_right <- ifelse(X_final$handed == 1, 1, 0)
  X_final$handed_left  <- ifelse(X_final$handed == 2, 1, 0)
  X_final$handed <- NULL
}

# Save final modified X
write.csv(
  X_final,
  paste0("X_set_", drop_option, "_final.csv"),
  row.names = FALSE
)

##############################################################
# Create variable index mapping for final X

# Number of predictors
p <- ncol(X_final)

# Create serial number + variable name mapping
X_var_map <- data.frame(
  serial_num = 1:p,
  variable_name = names(X_final)
)

# Save mapping
write.csv(
  X_var_map,
  paste0("X_variable_serial_num_set_", drop_option, "_final.csv"),
  row.names = FALSE
)

##############################################################
# Sanity check

cat("\nVariable mapping file saved:\n")
cat(paste0("X_variable_serial_num_set_", drop_option, "_final.csv\n"))

cat("\nFirst few entries of variable mapping:\n")
head(X_var_map)

##############################################################
# Check dimensions / names

cat("\nSaved files:\n")
cat(paste0("Y_set_", drop_option, ".csv\n"))
cat(paste0("X_set_", drop_option, ".csv\n"))
cat(paste0("X_set_", drop_option, "_final.csv\n"))

cat("\nDimension of Y:\n")
print(dim(data.frame(Y = Y)))

cat("\nDimension of original X:\n")
print(dim(X))

cat("\nDimension of final modified X:\n")
print(dim(X_final))

cat("\nColumn names of final modified X:\n")
print(names(X_final))

##############################################################
# Save retained rows for Section 2 exploratory / motivation plots

motivation_variables <- c(
  # outcome / ID
  "PATNO",
  "moca",
  
  # Figure M1 candidates
  "MCI_testscores",
  "hy",
  "updrs_totscore",
  "NP1COG",
  "NP1DPRS",
  "NP1ANXS",
  
  # Figure M2 candidates
  "NP1HALL",
  "NP1FATG",
  "orthostasis",
  "pm_cog_any",
  "pm_auto_any",
  "pm_wb_any",
  
  # Figure M3 candidates
  "quip_any",
  "rem"
)

motivation_variables_available <- intersect(
  motivation_variables,
  names(Curated_data_BL)
)

motivation_data <- Curated_data_BL[retained_row_numbers, motivation_variables_available]

write.csv(
  motivation_data,
  paste0("X_set_", drop_option, "_for_exploratory_plots.csv"),
  row.names = FALSE
)

cat("\nExploratory motivation data saved:\n")
cat(paste0("X_set_", drop_option, "_for_exploratory_plots.csv\n"))

cat("\nDimension of exploratory motivation data:\n")
print(dim(motivation_data))

cat("\nVariables saved for exploratory plots:\n")
print(names(motivation_data))