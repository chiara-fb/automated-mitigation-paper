

# Here we compute the market-level analysis for ISO-NE data w/o double ML.



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
setwd("C:\\Users\\c.fusarbassini\\Desktop\\automated_mitigation_paper")
source_python("utils.py")
setwd("C:\\Users\\c.fusarbassini\\OneDrive - Hertie School\\25 ML-Strom\\2 Literatur & Research ideas\\AP 3\\data")
data <- read_parquet("2025-08-12_iso-ne_dataset.parquet")
attach(data)
# Add bidder fixed effects
data$bidder <- as.factor(data$"Masked Lead Participant ID")


# Set parameters
threshold <- 1
bandwidth <- c(0.2, 0.5) #c(0.1, 0.2, 0.5)
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices")

### SCORE ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
data$score <- threshold - data$rsi
data <- data[year(data$DateTime) == 2019, ] # filter for the year 2019

### TREATMENT ###
data$treatment <- ifelse(data$score <= 0, 0, 1) # compute sharp treatment variable
data$treat_fuzzy <- fuzzy_prob(data$score, std=std) # calculate the probability of treatment

### BANDWIDTH ###
# choose only data within a certain range from the threshold to estimate the local treatment effect
subset1 <- data[data$score > - bandwidth[1] & data$score < bandwidth[1], ]
subset2 <- data[data$score > - bandwidth[2] & data$score < bandwidth[2], ]

### RDD ###
# Estimate the sharp RDD model with fixed effects with narrow and medium bandwidths
sharp_rdd <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + "), paste("| bidder")))
local <- feols(sharp_rdd, data = subset1)
wide <- feols(sharp_rdd, data = subset2)

# Estimate the fuzzy RDD models with fixed effects with medium bandwidth
fuzzy_rdd <- as.formula(paste("max_bid ~ treat_fuzzy + score + treat_fuzzy:score +", paste(covs, collapse = " + "), paste("| bidder")))
fuzzy <- feols(fuzzy_rdd, data = subset1)

etable(local, fuzzy, wide)




