

# Here we compute the market-level analysis for NYISO data w/o double ML.



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

# Some more packages
p_load(tictoc,tidyverse,broom,data.table,matrixStats, MASS, foreign, stargazer, reshape2, reticulate)

# Estimation packages
p_load(rdd,lfe,estimatr,boot, bootstrap, fixest)

# Data import (change the path accordingly)
source_python("amp_tests\\utils.py")
setwd("data")
data <- read_parquet("2025-08-12_nyiso_dataset.parquet")
attach(data)
# Add bidder fixed effects
data$bidder <- as.factor(data$"Masked Lead Participant ID")

# Set parameters
threshold <- 0.04
bandwidth <- c(3, 20) # choose bandwidths for the RDD # c(0.2, 3, 20) 
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices")

### SCORE AND TREATMENT ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
## main specification
data$score <- data$avg_cong_1h_lag - threshold
data$treatment <- ifelse(data$score <= 0, 0, 1) # compute sharp treatment variable
data$treat_fuzzy <- fuzzy_prob(data$score, std=std) # calculate the probability of treatment
data$score_no_lag <- data$avg_cong - threshold
data$treat_no_lag <- ifelse(data$score_no_lag <= 0, 0, 1) # compute sharp treatment variable
data$score_max <- data$max_cong_1h_lag - threshold
data$treat_max <- ifelse(data$score_max <= 0, 0, 1) # compute sharp treatment variable

data <- data[year(data$DateTime) == 2019, ] # filter for the year 2019

### BANDWIDTH ###
# choose only data within a certain range from the threshold to estimate the local treatment effect
subset1 <- data[data$score > - bandwidth[1] & data$score < bandwidth[1], ]
subset2 <- data[data$score > - bandwidth[2] & data$score < bandwidth[2], ]

### RDD ###
# Estimate the sharp RDD model with fixed effects with narrow and medium bandwidths
rdd_sharp <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + "), paste("| bidder")))
local <- feols(rdd_sharp, data = subset1)
wide <- feols(rdd_sharp, data = subset2)

# Estimate the fuzzy RDD models with fixed effects with medium bandwidth
rdd_fuzzy <- as.formula(paste("max_bid ~ treat_fuzzy + score + treat_fuzzy:score +", paste(covs, collapse = " + "), paste("| bidder")))
fuzzy <- feols(rdd_fuzzy, data = subset1)

etable(local, fuzzy, wide)

### SENSITIVITY ANALYSES ###
rdd_no_lag <- as.formula(paste("max_bid ~ treat_no_lag + score_no_lag + treat_no_lag:score +", paste(covs, collapse = " + "), paste("| bidder")))
no_lag <- feols(rdd_no_lag, data = subset1)
# test impact of spatial error
rdd_max <- as.formula(paste("max_bid ~ treat_max + score_max + treat_max:score +", paste(covs, collapse = " + "), paste("| bidder")))
max <- feols(rdd_max, data = subset1)
# test adding square term to regression
rdd_squared <- as.formula(paste("max_bid ~ treatment + score + treatment:score + I(score^2) + treatment:I(score^2) +", paste(covs, collapse = " + "), paste("| bidder")))
squared <- feols(rdd_squared, data = subset1)


sensitivity_bandwidth <- c(1.5, 4.5)
subset3 <- data[data$score > - sensitivity_bandwidth[1] & data$score < sensitivity_bandwidth[1], ]
subset4 <- data[data$score > - sensitivity_bandwidth[2] & data$score < sensitivity_bandwidth[2], ]
robust_narrow <- feols(rdd_sharp, data = subset3)
robust_wide <- feols(rdd_sharp, data = subset4)

etable(no_lag, max, squared)