library(ggplot2)

data <- read.csv("../outputs/exhaustive_sketch_sampling.csv")

ggplot(data, aes(x=size, y = samples)) +
    geom_col(data = data, fill = "deepskyblue2") +
    labs(y = "Number of samples", x = "Number of sketches as factor of CCSketch size") +
    scale_x_continuous(breaks = seq(0, max(data$size), by = 0.1)) +
    theme_minimal() +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank()) +
    ggsave("exhaustive_sketch_sampling.pdf", width = 6, height = 2)