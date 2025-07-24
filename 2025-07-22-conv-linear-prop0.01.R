library(data.table)
requireNamespace("mlr3resampling")
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/*csv")
##unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/*csv")[1]
task.list <- list()
mytab <- function(TSK){
  sname <- TSK$col_roles$subset
  dt <- TSK$data(col=c(sname, "y"))
  table(dt[[sname]], dt$y)
}
for(unb.csv in unb.csv.vec){
  data.csv <- sub("_unbalanced", "", unb.csv)
  MNIST_dt <- fread(file=data.csv)
  subset_dt <- fread(unb.csv) 
  subset_dt[, identical(
    seed1_prop0.1=="balanced",
    seed1_prop0.05=="balanced")]
  subset_dt[, identical(
    seed2_prop0.005=="balanced",
    seed2_prop0.001=="balanced")]
  subset_dt[, all(which(
    seed1_prop0.05=="unbalanced"
  ) %in% which(
    seed1_prop0.1=="unbalanced"
  ))]
  subset_dt[, all(which(
    seed2_prop0.001=="unbalanced"
  ) %in% which(
    seed2_prop0.005=="unbalanced"
  ))]
  ## the first factor level in R is considered the positive class
  ## in torch, which has the float value 1. The negative class is
  ## the second factor level, which gets converted to the float
  ## value 0.
  task_dt <- data.table(
    subset_dt, MNIST_dt
  )[
  , odd := factor(
    ifelse(y %% 2, "odd", "even"),
    c("odd", "even") # odd includes 1/positive, even includes 0/negative
  )]
  feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
  subset.name.vec <- names(subset_dt)
  subset.name.vec <- c("seed1_prop0.01","seed2_prop0.01")
  (data.name <- gsub(".*/|[.]csv$", "", unb.csv))
  for(subset.name in subset.name.vec){
    subset_vec <- task_dt[[subset.name]]
    ##print(table(subset_vec, task_dt$y))
    task_id <- paste0(data.name,"_",subset.name)
    itask <- mlr3::TaskClassif$new(
      task_id, task_dt[subset_vec != ""], target="odd")
    itask$col_roles$stratum <- c("y", subset.name)
    itask$col_roles$subset <- subset.name
    itask$col_roles$feature <- feature.names
    task.list[[task_id]] <- itask
  }
}
if(FALSE){#verify odd and y distributions.
  SOAK$instantiate(itask)
  SOAK_row <- SOAK$instance$iteration.dt[
    train.subsets=="same" & test.subset=="balanced" & test.fold==1]
  itask$backend$data(SOAK_row$train[[1]], c("odd","y"))[, table(odd, y)]
}
Proposed_AUM <- function(pred_tensor, label_tensor){
  is_positive = label_tensor$flatten() == 1
  is_negative = !is_positive
  if(all(as.logical(is_positive)) || all(as.logical(is_negative))){
    return(torch::torch_sum(pred_tensor*0))
  }
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
  roc = list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=torch::torch_cat(c(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(Inf))))
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  torch::torch_sum(min_FPR_FNR * constant_diff)
}
if(FALSE){
  w <- torch::nn_linear(2, 1)
  x <- torch::torch_randn(c(3,2))
  pred <- w(x)
  l <- torch::torch_tensor(c(1,1,1))
  aum <- Proposed_AUM(pred,l)
  aum$backward()
}
nn_AUM_loss <- torch::nn_module(
  "nn_AUM_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = Proposed_AUM
)
make_torch_learner <- function(id,...){
  po_list <- list(
    mlr3pipelines::po(
      "select",
      selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
    mlr3torch::PipeOpTorchIngressNumeric$new(),
    ...,
    mlr3pipelines::po("nn_head"),
    mlr3pipelines::po(
      "torch_loss",
      loss_fun),
    mlr3pipelines::po(
      "torch_optimizer",
      mlr3torch::t_opt("sgd", lr=0.05)),
    mlr3pipelines::po(
      "torch_callbacks",
      mlr3torch::t_clbk("history")),
    mlr3pipelines::po(
      "torch_model_classif",
      batch_size = 100000,
      patience=n.epochs,
      measures_valid=measure_list,
      measures_train=measure_list,
      predict_type="prob",
      epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)))
  graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
  glearner <- mlr3::as_learner(graph)
  mlr3::set_validate(glearner, validate = 0.5)
  mlr3tuning::auto_tuner(
    learner = glearner,
    tuner = mlr3tuning::tnr("internal"),
    resampling = mlr3::rsmp("insample"),
    measure = mlr3::msr("internal_valid_score", minimize = TRUE),
    term_evals = 1,
    id=paste0(id,"_",train_loss),
    store_models = TRUE)
}
n.pixels <- 28
n.epochs <- 400
## from https://github.com/OGuenoun/R-AUM_Multiclass/blob/main/AUM_comparison.r
ROC_AUM = R6::R6Class(
  "ROC_AUM",
  inherit = mlr3::MeasureClassif,
  public = list(
    AUM=Proposed_AUM,
    initialize = function() { 
      super$initialize(
        id = "ROC_AUM",
        packages = "torch",
        properties = character(),
        task_properties = "twoclass",
        predict_type = "prob",
        range = c(0, Inf),
        minimize = TRUE
      )
    }
  ),
  private = list(
    # define score as private method
    .score = function(prediction, ...) {
      pred_tensor <- torch::torch_tensor(prediction$prob[,1])
      label_tensor <- torch::torch_tensor(prediction$truth)
      loss_tensor <- self$AUM(pred_tensor, label_tensor)
      torch::as_array(loss_tensor)
    }
  )
)

measure_list <- c(
  mlr3::msrs(c("classif.auc", "classif.acc", "classif.logloss")),
  ROC_AUM$new())
mlr3torch_loss_list <- list(
  AUM=nn_AUM_loss,
  logistic=torch::nn_bce_with_logits_loss)
learner_list <- list()
for(train_loss in names(mlr3torch_loss_list)){
  loss_fun <- mlr3torch_loss_list[[train_loss]]
  arch.list <- list(
    make_torch_learner(
      "conv",
      mlr3pipelines::po(
        "nn_reshape",
        shape=c(-1,1,n.pixels,n.pixels)),
      mlr3pipelines::po(
        "nn_conv2d_1",
        out_channels = 20,
        kernel_size = 6),
      mlr3pipelines::po("nn_relu_1", inplace = TRUE),
      mlr3pipelines::po(
        "nn_max_pool2d_1",
        kernel_size = 4),
      mlr3pipelines::po("nn_flatten"),
      mlr3pipelines::po(
        "nn_linear",
        out_features = 50),
      mlr3pipelines::po("nn_relu_2", inplace = TRUE)
    ),
    make_torch_learner("linear")
  )
  for(arch.i in seq_along(arch.list)){
    learner_auto <- arch.list[[arch.i]]
    learner_list[[learner_auto$id]] <- learner_auto
  }
}
get_history_graph <- function(x){
  x$archive$learners(1)[[1]]$model$torch_model_classif$model$callbacks$history
}
sapply(learner_list, "[[", "predict_type")
for(learner_i in seq_along(learner_list)){
  learner_list[[learner_i]]$predict_type <- "prob"
}
sapply(learner_list, "[[", "predict_type")

proj.dir <- "/scratch/thocking/2025-07-22-conv-linear-prop0.01-new"
unlink(proj.dir, recursive=TRUE)
SOK <- mlr3resampling::ResamplingSameOtherSizesCV$new()
SOK$param_set$values$subsets <- "SO"
pgrid <- mlr3resampling::proj_grid(
  proj.dir,
  task.list,
  learner_list,
  SOK,
  save_learner=get_history_graph,
  score_args = measure_list)
mlr3resampling::proj_test(proj.dir, max_jobs=2, min_samples_per_stratum = 10)
proj.grid <- readRDS(file.path(proj.dir, "test", "grid.rds"))
mytab(task.list[[1]])
mytab(proj.grid$tasks[[1]])

mlr3resampling::proj_submit(proj.dir, tasks=5, hours=48, gigabytes=16)
gjobs <- fread(file.path(proj.dir, "grid_jobs.csv"))
slurm::sjob()

gres <- readRDS(file.path(proj.dir, "results.rds"))
cres <- fread(file.path(proj.dir, "results.csv"))
