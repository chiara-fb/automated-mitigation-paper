library(arrow)
library(tidyverse)


nbins <- 15
# setwd("data")
# data <- read_parquet("2025-08-12_iso-ne_dataset_with_lags.parquet")
# threshold <- 1
# bandwidth <- c(0.1, 0.2, 0.3, 0.5)

data <- read_parquet("2025-08-12_nyiso_dataset.parquet")
threshold <- 0.04
bandwidth <- c(1.5, 3, 4.5, 20) 
attach(data)

# Run McCrary density test

#implement a score variable that is centered around 0: left is positive, to the right is negative
data <- data[year(data$DateTime) == 2019, ] # filter for the year 2019
#score <- threshold - data$rsi
score <- data$avg_cong_1h_lag - threshold

for (b in bandwidth){
    s <- score[score > -  b & score < b]
    n <- length(s)


    # -------------------------------
    # Construct bins
    # -------------------------------
    binwidth = 2*b / nbins
    breaks <- seq(min(s), max(s) + binwidth, by = binwidth)
    mids <- breaks[-length(breaks)] + binwidth/2
    counts <- hist(s, breaks = breaks, plot = FALSE)$counts

    # Remove zero-count bins
    mids <- mids[counts > 0]
    counts <- counts[counts > 0]

    # -------------------------------
    # Log density
    # -------------------------------
    density_hat <- counts / (n * binwidth)
    log_density <- log(density_hat)

    # -------------------------------
    # Local linear = OLS inside window
    # -------------------------------

    fitL <- lm(log_density ~ I(mids), 
            weights = 1 * (mids < 0))

    fitR <- lm(log_density ~ I(mids), 
            weights = 1 * (mids >= 0))

    alpha_L <- coef(fitL)[1]
    alpha_R <- coef(fitR)[1]

    # -------------------------------
    # 7. Discontinuity estimate
    # -------------------------------
    theta_hat <- alpha_R - alpha_L

    se_theta <- sqrt(vcov(fitL)[1,1] +
                    vcov(fitR)[1,1])

    t_stat <- theta_hat / se_theta
    p_value <- 2 * (1 - pnorm(abs(t_stat)))

    print(paste("Bandwidth:", b, "Binwidth: ", binwidth, "No. of bins: ", length(counts), "P-value: ", p_value))

}

