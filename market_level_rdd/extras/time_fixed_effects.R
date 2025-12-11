

# Here we compute the market-level analysis for ISO-NE isone_data w/o double ML.



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
p_load(tictoc,tidyverse,broom,data.table, matrixStats, MASS, foreign, stargazer, reshape2, reticulate)

# Estimation packages
p_load(rdd,lfe,estimatr,boot, bootstrap, fixest)

# isone_data import (change the path accordingly)
source_python("amp_tests\\utils.py")
setwd("data")
isone_data <- read_parquet("2025-08-12_iso-ne_dataset.parquet")
attach(isone_data)
# Add bidder fixed effects
isone_data$bidder <- as.factor(isone_data$"Masked Lead Participant ID")
isone_data$month <- as.factor(month(isone_data$DateTime))
isone_data$hour  <- as.factor(hour(isone_data$DateTime))


# Set parameters
threshold <- 1
bandwidth <- c(0.2, 0.5) #c(0.1, 0.2, 0.5)
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices", "month")

### SCORE ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
isone_data$score <- threshold - isone_data$rsi
isone_data <- isone_data[year(isone_data$DateTime) == 2019, ] # filter for the year 2019

### TREATMENT ###
isone_data$treatment <- ifelse(isone_data$score <= 0, 0, 1) # compute sharp treatment variable
isone_data$treat_fuzzy <- fuzzy_prob(isone_data$score, std=std) # calculate the probability of treatment

### BANDWIDTH ###
# choose only isone_data within a certain range from the threshold to estimate the local treatment effect
subset1 <- isone_data[isone_data$score > - bandwidth[1] & isone_data$score < bandwidth[1], ]
subset2 <- isone_data[isone_data$score > - bandwidth[2] & isone_data$score < bandwidth[2], ]

### RDD ###
# Estimate the sharp RDD model with fixed effects with narrow and medium bandwidths
rdd_sharp <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + "), paste("| bidder")))
isone_local <- feols(rdd_sharp, data = subset1)
isone_wide <- feols(rdd_sharp, data = subset2)

# Estimate the fuzzy RDD models with fixed effects with medium bandwidth
rdd_fuzzy <- as.formula(paste("max_bid ~ treat_fuzzy + score + treat_fuzzy:score +", paste(covs, collapse = " + "), paste("| bidder")))
isone_fuzzy <- feols(rdd_fuzzy, data = subset1)

etable(isone_local, isone_fuzzy, isone_wide)


nyiso_data <- read_parquet("2025-08-12_nyiso_dataset.parquet")
attach(nyiso_data)
# Add bidder fixed effects
nyiso_data$bidder <- as.factor(nyiso_data$"Masked Lead Participant ID")
nyiso_data$month <- as.factor(month(nyiso_data$DateTime))
nyiso_data$hour  <- as.factor(hour(nyiso_data$DateTime))

# Set parameters
threshold <- 0.04
bandwidth <- c(3, 20) # choose bandwidths for the RDD # c(0.2, 3, 20) 
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices", "month")

### SCORE AND TREATMENT ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
## main specification
nyiso_data$score <- nyiso_data$avg_cong_1h_lag - threshold
nyiso_data$treatment <- ifelse(nyiso_data$score <= 0, 0, 1) # compute sharp treatment variable
nyiso_data$treat_fuzzy <- fuzzy_prob(nyiso_data$score, std=std) # calculate the probability of treatment

nyiso_data <- nyiso_data[year(nyiso_data$DateTime) == 2019, ] # filter for the year 2019

### BANDWIDTH ###
# choose only data within a certain range from the threshold to estimate the local treatment effect
subset1 <- nyiso_data[nyiso_data$score > - bandwidth[1] & nyiso_data$score < bandwidth[1], ]
subset2 <- nyiso_data[nyiso_data$score > - bandwidth[2] & nyiso_data$score < bandwidth[2], ]

### RDD ###
# Estimate the sharp RDD model with fixed effects with narrow and medium bandwidths
rdd_sharp <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + "), paste("| bidder")))
nyiso_local <- feols(rdd_sharp, data = subset1)
nyiso_wide <- feols(rdd_sharp, data = subset2)
# Estimate the fuzzy RDD models with fixed effects with medium bandwidth
rdd_fuzzy <- as.formula(paste("max_bid ~ treat_fuzzy + score + treat_fuzzy:score +", paste(covs, collapse = " + "), paste("| bidder")))
nyiso_fuzzy <- feols(rdd_fuzzy, data = subset1)

etable(nyiso_local, nyiso_fuzzy, nyiso_wide)