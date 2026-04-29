library(dplyr)
library(tidyr)
library(readxl)
library(readr)
library(reshape2)
library(broom)
library(dplyr)
library(purrr)
library(glmnet)
library(quantreg)
library(rqPen)



setwd("U:/UNIQUE/Case study/PPMI data/Table uniQR vs LassoQR")
#Read the data
Curated_data<-read_excel("PPMI_Curated_Data_Cut_Public_20250321.xlsx",sheet = "20250310")

#select only baseline data
Curated_data_BL<-Curated_data[Curated_data$EVENT_ID=="BL",]

#make the id numeric
Curated_data_BL$PATNO<-as.numeric(Curated_data_BL$PATNO)




##############################################################
#Full Variable names

# hvlt_immediaterecall=HVLT Immediate/Total Recall
# HVLTRDLY=HVLT Delayed Recall
# hvlt_discrimination=HVLT Discrimination Recognition Index
# HVLTFPRL=HVLT False Alarms
# HVLTREC=HVLT Delayed Recognition
# hvlt_retention=HVLT Retention
# moca=Montreal Cognitive Assessment (MoCA) Score (adjusted for education)
# VLTANIM=Semantic Fluency (Animal) Score
# SDMTOTAL=Symbol Digit Modalities Score
# bjlot=Benton Judgement of Line Orientation Score
# lns=Letter Number Sequencing Score
# ess=Epworth Sleepiness Scale Score
# gds= geriatic depression scale score
# upsit = UPSIT raw score
# stai=State-Trait Anxiety Index (STAI) Total Score
# MSEADLG=Modified Schwab & England Activities of Daily Living Score (ADL)
# rem= REM Sleep Behavior Disorder Screening Questionnaire (RBDSQ) total score
# nfl_serum=Serum Neurofilament Light
# urate=Serum Uric Acid (mg/dL)
# scopa= SCOPA-AUT Total Score



# DatScan variable
# mean_caudate,mean_putamen,mean_striatum

#Biologics variable "abeta","asyn","ptau","tau"


# updrs4_score: missing 3755
# MSEADLG: missing 201 but it reduce the obs from 456 to 331
# nfl_serum (Serum Neurofilament Light): missing 2906. if we remove this the obs from 456 to 542
# urate: missing 1354. if we remove this then obs from 542 to 555 
####################################################################


data<-Curated_data_BL%>%
  dplyr::select(PATNO, Y = moca,
                upsit, quip, ess, gds, scopa,lns,
                abeta, tau, ptau, asyn,
                mean_putamen, mean_caudate, mean_striatum,
                hvlt_discrimination,hvlt_immediaterecall,hvlt_retention,HVLTFPRL,HVLTRDLY,HVLTREC,
                VLTANIM,SDMTOTAL,bjlot,lns,stai,rem,
                updrs1_score,updrs2_score,updrs3_score,
                nfl_serum,
                urate,
                age,EDUCYRS,APOE_e4,pigd
  )


#remove the id
data<-data%>%
  dplyr::select(-PATNO)

summary(data$Y)

data<-data.frame(data)


head(data)




############################################
#Check: Non_Missing, Missing, Total

get_summary_table <- function(data) {
  data.frame(
    Variable = colnames(data),
    Non_Missing = colSums(!is.na(data)),
    Missing = colSums(is.na(data)),
    Total = nrow(data)
  )
}


summary_data <- get_summary_table(data)

summary_data[,-1]


data<-drop_na(data) # take complete data


dim(data)


head(data)

X<-data[,-1]
Y<-data[,1]

str(data)

dim(data)
##################################







##########################################################
#Variable section LASSO

# install.packages("glmnet")  # if needed
# library(glmnet)

y <- data$Y

# model.matrix handles factors if any; drop intercept column
X <- model.matrix(Y ~ . , data = data)[, -1]

n <- nrow(X)

# LOOCV to choose lambda
set.seed(12345)
cvfit <- cv.glmnet(
  x = X,
  y = y,
  alpha = 1,          # LASSO
  nfolds = n,         # LOOCV (leave one out CV)
  grouped = FALSE,   # explicitly set
  standardize = TRUE,
  family = "gaussian"
)

lambda_min <- cvfit$lambda.min
lambda_min

# Fit final LASSO model at chosen lambda on the full dataset
fit_min <- glmnet(X, y, alpha = 1, standardize = TRUE, family = "gaussian", lambda = lambda_min)

# Predict on the full dataset (in-sample predictions)
pred_min <- as.numeric(predict(fit_min, newx = X, s = lambda_min))

# MAD (median absolute deviation of prediction errors) 
mad_min <- median(abs(y - pred_min))

mad_min


#  which variables are selected
coef_min <- coef(fit_min)

selected_min <- rownames(coef_min)[as.vector(coef_min != 0)]
selected_min

######################################################







###################################
# Univariate and LASSO model
###################################

outcome <- "Y"

predictors <- setdiff(
  names(data),
  outcome
)




uni_results <- map_dfr(predictors, function(var) {
  
  formula <- as.formula(paste(outcome, "~", var))
  
  fit <- lm(formula, data = data)
  
  tidy(fit, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(predictor = var)
})



uni_table <- uni_results %>%
  select(
    predictor
    , estimate
    # ,std.error
    # ,conf.low
    # ,conf.high
    # ,p.value
   ) #%>%
  # arrange(p.value)


print(data.frame(uni_table))



Uni_table<-print(
  data.frame(
    uni_table %>%
      mutate(across(where(is.numeric), ~ round(.x, 3)))
  ),
  row.names = FALSE
)





x <- model.matrix(Y ~ . , data = data)[, -1]  # remove intercept
y <- data$Y
n<-length(y)

set.seed(123)

cv_lasso <- cv.glmnet(
  x,
  y,
  alpha = 1,        # LASSO
  # nfolds = n,         # LOOCV (leave one out CV)
  grouped = FALSE,   # explicitly set
  standardize = TRUE,
  family = "gaussian"
)

lasso_coef <- coef(cv_lasso, s = "lambda.min")



lasso_table <- data.frame(
  predictor = rownames(lasso_coef),
  estimate  = as.numeric(lasso_coef)
) %>%
  filter(predictor != "(Intercept)") %>%
  mutate(estimate = round(estimate, 3)) %>%
  arrange(desc(abs(estimate)))





Uni_table <- Uni_table %>%
  rename(uni_coef = estimate)

lasso_table <- lasso_table %>%
  rename(lasso_coef = estimate)


coef_compare <- Uni_table %>%
  full_join(lasso_table, by = "predictor") #%>%
  # arrange(desc(abs(lasso_coef)))


coef_compare

##############################################


## To get the latex table
# library(knitr)
# library(kableExtra)
# 
# kable(
#   coef_compare,
#   format = "latex",
#   booktabs = TRUE,
#   digits = 3,
#   col.names = c("Predictor", "Univariate Coefficient", "LASSO Coefficient"),
#   caption = "Comparison of Univariate and LASSO Regression Coefficients"
# ) %>%
#   kable_styling(
#     latex_options = c("hold_position", "scale_down"),
#     font_size = 9
#   )














###################################
# univariate quantile regression
###################################
# library(quantreg)


quantile<-.75   # .25, .5, .75 ### user input ###
  
outcome <- "Y"

predictors <- setdiff(
  names(data),
  outcome
)



uni_results <- map_dfr(predictors, function(var) {
  
  formula <- as.formula(paste(outcome, "~", var))
  
  fit <- rq(formula,tau = quantile, data = data)
  
  tidy(fit, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(predictor = var)
})



uni_table <- uni_results %>%
  select(
    predictor
    , estimate
  )

print(data.frame(uni_table))



Uni_table<-print(
  data.frame(
    uni_table %>%
      mutate(across(where(is.numeric), ~ round(.x, 3)))
  ),
  row.names = FALSE
)






x <- model.matrix(Y ~ ., data = data)[, -1]  # remove intercept
y <- data$Y


# Fit quantile LASSO
set.seed(123)

cv_q_lasso <- rq.pen.cv(
  x = x,
  y = y,
  tau = quantile, #Quantiles 
  penalty = "LASSO",
  nfolds = 10
)


idx_min <- which.min(cv_q_lasso$cverr)
lasso_coef <- coef(cv_q_lasso$fit)[, idx_min][-1]



lasso_table <- data.frame(
  predictor = names(lasso_coef),
  estimate  = as.numeric(lasso_coef)
) %>%
  filter(predictor != "(Intercept)") %>%
  mutate(estimate = round(estimate, 3)) %>%
  arrange(desc(abs(estimate)))



Uni_table <- Uni_table %>%
  rename(uni_coef = estimate)

lasso_table <- lasso_table %>%
  rename(lasso_coef = estimate)


coef_compare <- Uni_table %>%
  full_join(lasso_table, by = "predictor") #%>%
  # arrange(desc(abs(lasso_coef)))


coef_compare



# library(knitr)
# library(kableExtra)
# 
# kable(
#   coef_compare,
#   format = "latex",
#   booktabs = TRUE,
#   digits = 3,
#   col.names = c("Predictor", "Univariate Coefficient", "LASSO Coefficient"),
#   caption = "Comparison of Univariate and LASSO Regression Coefficients"
# ) %>%
#   kable_styling(
#     latex_options = c("hold_position", "scale_down"),
#     font_size = 9
#   )


