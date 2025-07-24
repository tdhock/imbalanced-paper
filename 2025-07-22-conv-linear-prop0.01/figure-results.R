library(data.table)
## data files from /scratch/thocking/2025-07-22-conv-linear-prop0.01
score_dt <- fread("results.csv")
plist <- mlr3resampling::pvalue(score_dt)
plot(plist)
score_dt[learner_id=="conv_AUM" & task_id=="FashionMNIST_seed1_prop0.01" & test.subset=="balanced"]# Why are there only 3 rows for same? there should be six rows, 3 same and 3 other.
score_dt[learner_id=="conv_AUM" & task_id=="EMNIST_seed1_prop0.01" & test.subset=="balanced"]

score_not_fashion <- score_dt[!grepl("Fashion",task_id)]
p_not_fashion <- mlr3resampling::pvalue(score_not_fashion)
png("2025-07-22-conv-linear-prop0.01.png", width=17, height=7, units="in", res=200)
gg <- plot(p_not_fashion)
print(gg)
dev.off()

history_dt <- fread("learners.csv.gz")[
  test.fold==1 & test.subset=="balanced"
][
, test.subset := sub("un", "im", test.subset)
][]
for(set_name in c("train","valid")){
  auc_name <- paste0(set_name, ".classif.auc")
  inv_value <- 1-history_dt[[auc_name]]
  inv_name <- paste0(set_name, ".invAUC")
  set(history_dt, j=inv_name, value=inv_value)
}
melt_history <- function(DT)nc::capture_melt_single(
  DT,
  set=nc::alevels(train="subtrain", valid="validation"),
  "[.]",
  measure=nc::alevels(
    classif.acc="accuracy_prop",
    classif.ce="error_prop",
    classif.auc="AUC",
    classif.logloss="logloss",
    "invAUC",
    ROC_AUM="AUM"))
other <- c(imbalanced="balanced", balanced="imbalanced")
history_long <- melt_history(history_dt)[, let(
  train.subset = ifelse(train.subsets=="same",test.subset,other[test.subset]),
  loss=ifelse(grepl("AUM", learner_id), "AUM", "logloss"),
  Data = gsub("_","\n", task_id),
  Learner = paste0("\n", learner_id)
)][, let(
  Measure = fcase(
    measure==loss, "loss",
    measure=="invAUC", "invAUC",
    default=NA)
)][]
history_long[, table(measure, Measure)]
## plot loss.
(history_show <- history_long[!is.na(Measure)])
min_dt <- history_show[set=="validation", .SD[which.min(value)], by=.(train.subset, Data, learner_id, Measure)]
library(ggplot2)
gg <- ggplot()+
  ggtitle("Test subset balanced, test fold 1")+
  theme_bw()+
  geom_line(aes(
    epoch, value, color=set),
    data=history_show)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=data.frame(min_dt, point="min"))+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(Learner + Measure ~ train.subset + Data, labeller=label_both, scales="free")+
  scale_y_log10(
    "Objective value (AUM or logistic or AUC)",
    breaks=c(0.5, 10^seq(-5, -1)))
png("2025-07-22-conv-linear-prop0.01-subtrain-validation.png", width=20, height=10, units="in", res=200)
print(gg)
dev.off()
