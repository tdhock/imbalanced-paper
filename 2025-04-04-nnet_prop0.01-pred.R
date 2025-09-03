(objs=load("2025-04-04-nnet_prop0.01.RData"))#from beluga scratch
score_dt <- bench.result$score()#mlr3resampling::score(bench.result)

linear_dt <- score_dt[grepl("linear", learner_id) & grepl("EMNIST_seed1", task_id) & iteration==1]
library(data.table)
melt_dt_list <- list()
for(row_i in 1:nrow(linear_dt)){
  ptest <- linear_dt$prediction_test[[row_i]]
  melt_dt_list[[row_i]] <- melt(
    data.table(prob=ptest$prob, truth=ptest$truth, linear_dt[row_i, .(learner_id, iteration)]),
    measure.vars=measure(pred_class, pattern="prob[.]([0-1])"))
}
(melt_dt <- rbindlist(melt_dt_list))
library(ggplot2)

gg <- ggplot()+
  geom_histogram(aes(
    value, fill=pred_class, group=pred_class),
    bins=100,
    data=melt_dt)+
  scale_x_continuous("Predicted probability for pred_class")+
  facet_grid(truth ~ learner_id, scales="free", labeller=label_both)
png("2025-04-04-nnet_prop0.01-pred.png", width=6, height=4, units="in", res=200)
print(gg)
dev.off()
