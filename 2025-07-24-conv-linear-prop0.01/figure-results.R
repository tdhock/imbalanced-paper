library(data.table)
## data files from /scratch/thocking/2025-07-24-conv-linear-prop0.01

history_dt <- fread("learners.csv.gz")[
  test.fold==1 
][, test.subset := sub("un", "im", test.subset)]
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

under2newline <- function(x)paste0("\n", gsub("_", "\n", x))
history_long <- melt_history(history_dt)[, let(
  train.subset = ifelse(train.subsets=="same",test.subset,other[test.subset]),
  loss=ifelse(grepl("AUM", learner_id), "AUM", "logloss"),
  Data = under2newline(task_id),
  Learner = under2newline(learner_id)
)][, let(
  Measure = fcase(
    measure==loss, "\nloss",
    measure=="invAUC", "\ninvAUC",
    default=NA)
)][]
history_long[, table(measure, Measure)]
history_long[train.subset=="imbalanced", table(Learner, test.fold)]
## plot loss.
(history_show <- history_long[!is.na(Measure)][, let(
  Test_subset = paste0("\n", test.subset),
  Train_subset= paste0("\n", train.subset)
)][])

history_AUM <- history_show[grepl("AUM", Learner)& task_id=="MNIST_seed1_prop0.01"] 
min_dt <- history_AUM[set=="validation", .SD[which.min(value)], by=.(test.subset, train.subset, learner_id, Measure)]
library(ggplot2)
gg <- ggplot()+
  ggtitle("Test fold 1, Data MNIST, seed1, prop0.01, AUM")+
  theme_bw()+
  geom_line(aes(
    epoch, value, color=set),
    data=history_AUM)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=data.frame(min_dt, point="min"))+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(Learner + Measure ~ test.subset + train.subset, labeller=label_both, scales="free")+
  scale_y_log10(
    "Objective value (AUM or AUC)",
    breaks=c(0.5, 10^seq(-5, -1)))
png("2025-07-24-conv-linear-prop0.01-AUM.png", width=20, height=10, units="in", res=200)
print(gg)
dev.off()

history_log <- history_show[!grepl("AUM", Learner)]
min_dt <- history_log[set=="validation", .SD[which.min(value)], by=.(Data, test.subset, train.subset, learner_id, Measure)]
library(ggplot2)
gg <- ggplot()+
  ggtitle("Train balanced, log loss")+
  theme_bw()+
  geom_line(aes(
    epoch, value, color=set),
    data=history_log)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=data.frame(min_dt, point="min"))+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(Learner + Measure ~ Data + test.subset, labeller=label_both, scales="free")+
  scale_y_log10(
    "Objective value (AUM or AUC)",
    breaks=c(0.5, 10^seq(-5, -1)))
png("2025-07-24-conv-linear-prop0.01-logistic.png", width=20, height=10, units="in", res=200)
print(gg)
dev.off()

history_loss <- history_show[!grepl("AUM", Learner) & measure=="logloss"]
nc::capture_first_df(
  history_loss,
  learner_id=list(
    model=".*?",
    "_",
    Loss=".*?",
    "_",
    lr="[0-9.]+", as.numeric))
min_dt <- history_loss[set=="validation", .SD[which.min(value)], by=.(Data, test.subset, train.subset, model)]
hline_dt <- min_dt[, .(Test_subset, Data, value)]
library(ggplot2)
gg <- ggplot()+
  ggtitle("Log loss, compare step sizes")+
  theme_bw()+
  geom_hline(aes(
    yintercept=value),
    data=hline_dt,
    color="grey50")+
  geom_line(aes(
    epoch, value, color=set),
    data=history_loss)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=data.frame(min_dt, point="min"))+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(Data + Test_subset ~ model + lr, labeller=label_both, scales="free")+
  scale_y_log10(
    "Objective value (log loss)",
    breaks=c(0.5, 10^seq(-5, -1)))
png("2025-07-24-conv-linear-prop0.01-logistic-loss-MNIST.png", width=20, height=12, units="in", res=200)
print(gg)
dev.off()

history_loss <- history_show[grepl("AUM", Learner) & measure=="AUM"]
nc::capture_first_df(
  history_loss,
  learner_id=list(
    model=".*?",
    "_",
    Loss=".*?",
    "_",
    lr="[0-9.]+", as.numeric))
min_dt <- history_loss[set=="validation", .SD[which.min(value)], by=.(Data, Test_subset, Train_subset, model)]
hline_dt <- min_dt[, .(Test_subset, Train_subset, Data, value)]
library(ggplot2)
gg <- ggplot()+
  ggtitle("AUM loss, compare step sizes")+
  theme_bw()+
  geom_hline(aes(
    yintercept=value),
    data=hline_dt,
    color="grey50")+
  geom_line(aes(
    epoch, value, color=set),
    data=history_loss)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=data.frame(min_dt, point="min"))+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(Data + Test_subset + Train_subset ~ model + lr, labeller=label_both, scales="free")+
  scale_y_log10(
    "Objective value (AUM loss)",
    breaks=c(0.5, 10^seq(-5, -1)))
png("2025-07-24-conv-linear-prop0.01-AUM-compare-lr.png", width=20, height=15, units="in", res=200)
print(gg)
dev.off()
