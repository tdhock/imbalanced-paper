library(ggplot2)
library(data.table)
result.dt <- nc::capture_first_df(fread('results.csv'), learner_id=list(
  arch=".*",
  "_",
  loss=".*",
  "_",
  lr=".*", as.numeric
))[, let(
  Data=paste0("\n", task_id),
  hours=difftime(end.time, start.time, units="hours"),
  start.hours=difftime(start.time, min(start.time), units="hours"),
  end.hours=difftime(end.time, min(start.time), units="hours")
)][]
gg <- ggplot()+
  geom_segment(aes(
    start.hours, process,
    xend=end.hours, yend=process,
    color=loss),
    data=result.dt)+
  geom_point(aes(
    start.hours, process,
    size=arch,
    fill=lr,
    color=loss),
    shape=21,
    data=result.dt)+
  facet_grid(test.subset + train.subsets ~ Data, labeller=label_both)+
  scale_size_manual(values=c(conv=2,linear=1))+
  scale_fill_gradient(low="white", high="black", trans="log10")+
  scale_x_continuous("Hours from start of computation")+
  theme_bw()+
  ggtitle("Each segment shows time for training 10000 epochs (full batch size), then re-training with best number of epochs, 543/864 jobs finished in 20 hours on 200 CPUs")
png("results-timings.png", width=15, height=8, res=200, units="in")
print(gg)
dev.off()
