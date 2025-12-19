
# Here we compute the bidder-level analysis for ISO-NE data w/o double ML.

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
library(svglite)

# Some more packages
p_load(tictoc,tidyverse,broom,data.table,matrixStats, MASS, foreign, stargazer, reshape2, reticulate)

# Estimation packages
p_load(rdd,lfe,estimatr,boot, bootstrap, fixest)

tryCatch({
    setwd("data")
},  error = function(e) {
                 print("Directory already updated.")
            }
)

data <- read_parquet("2025-08-12_iso-ne_dataset.parquet")


# Rename bidder and unit columns
data <- data %>% rename(bidder = "Masked Lead Participant ID", unit = "Masked Asset ID")
data$unit <- as.factor(data$unit)


# Set parameters
threshold <- 1
bw <- 0.2
std <- 0.01 # 0.05
covs <- c("ref_level", "gas_prices")

### SCORE ###
#implement a score variable that is centered around 0: left is positive, to the right is negative
data$score <- threshold - data$rsi
data <- data[year(data$DateTime) == 2019, ] # filter for the year 2019
### TREATMENT ###

data$treatment <- ifelse(data$score <= 0, 0, 1) # compute sharp treatment variable
subset_data <- data[data$score > - bw & data$score < bw, ]


### RDD ###
# Estimate the sharp RDD model with medium bandwidths
main <- as.formula(paste("max_bid ~ treatment + score + treatment:score +", paste(covs, collapse = " + ")))

### GROUP SPLIT ###
bidder_groups <- subset_data %>% group_by(bidder)
bidders_df <- group_split(bidder_groups)
bidders <- group_keys(bidder_groups)
n <- nrow(bidders)
line <- y ~ poly(x, 1, raw = TRUE)

### ITERATIVELY COMPUTE RDD FOR EACH MODEL SPECIFICATION AND BIDDER ##
    
    for (i in 1:n) {
        bidder_id <- bidders$bidder[i]
        bidder_data <- bidders_df[[i]]
        ref <- max(bidder_data$ref_level)
        threshold <- min(ref + 100, 3*ref)

        print(paste("Bidder", bidder_id, "Reference:", ref, "Threshold:", threshold))
        
        tryCatch(
            {    
            model <- lm(main, data = bidder_data)
            p <- ggplot(bidder_data, aes(x = rsi, y = max_bid, group = as.factor(treatment))) +
              theme(
                text = element_text(size = 12),        # base font size
                title = element_text(hjust=0.5, size=24),
              ) + 
            geom_point(size = 2) + 
            geom_smooth(method = "lm", se = TRUE, formula = line, linewidth = 2) +
            geom_vline(xintercept = 1, linewidth = 1) + 
            labs(x = "RSI", y = "Maximum bid") +
            geom_segment(
            x = 0, xend = 1,
            y = ref, yend = ref, colour = "#2ca02c", linewidth = 2) + 
            geom_segment(
            x = 0, xend = 1,
            y = threshold, yend = threshold, colour = "#d62728", linewidth = 2, linetype = "dotdash") + 
            scale_x_continuous(breaks = seq(0.7, 1.3, by = 0.2)) +
            ggtitle("Parametric fit of RDD for a bidder")  
            ggsave(paste0(bidder_id, ".svg"), plot=p, height=6, width=12)
            print(paste("Saved bidder:", bidder_id))
            },
            error = function(e) {
                print(paste("Error for bidder", bidder_id, ":", e$message))
            }
        )

    }


