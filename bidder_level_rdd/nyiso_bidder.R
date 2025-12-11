
# Here we compute the bidder-level analysis for NYISO data w/o double ML.


library(dplyr)
library(ggplot2)
library(matrixcalc)
library(corpcor)
library(plm)
library(lmtest)
library(multiwayvcov)
library(numDeriv)
library(sandwich)
library(arrow)
library(pacman)
library(openxlsx)

# Some more packages
p_load(tictoc,tidyverse,broom,data.table,matrixStats, MASS, foreign, stargazer, reshape2, reticulate)

# Estimation packages
p_load(rdd,lfe,estimatr,boot, bootstrap, fixest)

# Data import (change the path accordingly)
source_python("amp_tests\\utils.py")
setwd("data")
data <- read_parquet("2025-08-12_nyiso_dataset.parquet")
attach(data)

# Rename bidder and unit columns
data <- data %>% rename(bidder = "Masked Lead Participant ID", unit = "Masked Asset ID")
data$unit <- as.factor(data$unit)

# Set parameters
threshold <- 0.04
bandwidth <- c(3, 20) # choose bandwidths for the RDD c(0.2, 3, 20)
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices")
multicovs <- c("ref_level", "gas_prices", "load_fcst", "temperature")

### SCORE ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
data$score <- data$avg_cong_1h_lag - threshold
data <- data[year(data$DateTime) == 2019, ] # filter for the year 2019

### TREATMENT ###
data$treatment <- ifelse(data$score <= 0, 0, 1) # compute sharp treatment variable
data$treat_fuzzy_cont <- fuzzy_prob(data$score, std=std) # calculate the probability of treatment


### RDD ###
# Estimate the sharp RDD model with medium bandwidths
main <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + ")))
fuzzy <- as.formula(paste("max_bid ~ treat_fuzzy_cont + score + treat_fuzzy_cont:score +", paste(covs, collapse = " + ")))
unit <- as.formula(paste("max_bid ~ treatment + score + treatment:score + unit +", paste(covs, collapse = " + ")))
multi <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(multicovs, collapse = " + ")))

### GROUP SPLIT ###
bidder_groups <- data %>% group_by(bidder)
bidders_df <- group_split(bidder_groups)
bidders <- group_keys(bidder_groups)
n <- nrow(bidders)

models <- list(
  local = main,
  wide = main, # same specification but broader bandwidth
  fuzzy = fuzzy,
  unit = unit,
  multi = multi
)

all_results <- setNames(bidders, c("bidder_id"))

### ITERATIVELY COMPUTE RDD FOR EACH MODEL SPECIFICATION AND BIDDER ##

for (model_name in names(models)) {
    
    model_spec <- models[[model_name]]
    col_names <- c("bidder_id", paste0("coef_", model_name), paste0("t_stat_", model_name), paste0("p_val_", model_name), paste0("num_bids_", model_name))
    results <- data.frame(
                rep(NA_real_, n),
                rep(NA_real_, n),
                rep(NA_real_, n),
                rep(NA_real_, n),
                rep(NA_real_, n),
                stringsAsFactors = FALSE
                ) |> setNames(col_names)

    for (i in 1:n) {
        bidder_id <- bidders$bidder[i]
        bidder_data <- bidders_df[[i]]
        bw <- bandwidth[1]
        treat_var <- "treatment"

        # TREATMENT VARIABLE CHANGES NAME IN FUZZY SPECIFICATION
        if (model_name == "fuzzy") {
            treat_var <- "treat_fuzzy_cont"
        
        # NO NEED FOR DUMMY IF BIDDER ONLY OWNS ONE UNIT
        } else if (model_name == "unit" & length(unique(bidder_data$unit)) == 1) {
            treat_var <- "treatment"
            model_spec <- main
        
        } else if (model_name == "wide") {
            bw <- bandwidth[2]
        } 
        
        tryCatch(
            {    
            subset_data <- bidder_data[bidder_data$score > - bw & bidder_data$score < bw, ]
            model <- lm(model_spec, data = subset_data)
            res <-summary(model)
            results$bidder_id[i] <- bidder_id
            results[[paste0("coef_", model_name)]][i] <- res$coefficients[treat_var, "Estimate"]
            results[[paste0("t_stat_", model_name)]][i] <- res$coefficients[treat_var, "t value"]
            results[[paste0("p_val_", model_name)]][i] <- res$coefficients[treat_var, "Pr(>|t|)"]
            results[[paste0("num_bids_", model_name)]][i] <- nrow(subset_data)
            },
            error = function(e) {
                print(paste("Error for bidder", bidder_id, ":", e$message))
            }
        )

    }
    all_results <- full_join(all_results, results, by="bidder_id")
    print(paste("Completed model", model_name))

}

#dropna
all_results <- na.omit(all_results)
#write.xlsx(all_results, "nyiso_results.xlsx")