# Measurement invariance between human and synthetic records (Reviewer 1, comment 1).
#
# For every dataset, model family, arm and scale, a one-factor model is fitted to the
# human answers of a set of respondents and to the synthetic answers generated for the
# same respondents. Two sequences are run, because neither is sufficient alone.
#
# ordinal     Items treated as ordered categories (WLSMV, delta parameterization), with
#             the constraints that are identified for ordinal indicators (Wu and
#             Estabrook, 2016, written by semTools::measEq.syntax): configural, then
#             equal thresholds, then equal thresholds and loadings. A category that one
#             group never uses has no threshold in that group, so it is merged with its
#             neighbour in both groups. The number of merges is recorded. When many
#             categories are merged the sequence loses its power to find threshold
#             differences, since the merging removes them.
# continuous  Items treated as continuous (robust ML), no category merged: configural,
#             equal loadings (metric), equal loadings and intercepts (scalar). The
#             intercept step is where a difference in how the response scale is used
#             shows up, which is what the ordinal sequence cannot test once categories
#             are missing.
#
# Each step is compared with the previous one by a scaled chi-square difference test and
# by the change in CFI and RMSEA. The same two sequences are run on random halves of the
# human respondents. Those halves are invariant by construction, so they give the
# rejection rate and the change in fit to expect when invariance holds.
#
# Input : results/analysis/r2_item_records_<dataset>.csv (repeated-splits source, one
#         synthetic record per respondent), written by r2_psychometric_properties.py
# Output: results/analysis/r2_invariance_fits.csv, r2_invariance_loadings.csv,
#         r2_invariance_control.csv
#
# Usage : Rscript scripts/r2_invariance.R        (R2_TEST=1 for a two-cell check)

suppressMessages({library(lavaan); library(semTools)})
set.seed(20261112)
root <- "results/analysis"
datasets <- c("anket_tr", "bangladesh_sns")
ordinal_steps <- list(configural = character(0), thresholds = "thresholds",
                      loadings = c("thresholds", "loadings"))
continuous_steps <- list(configural = "", metric = "loadings", scalar = c("loadings", "intercepts"))
control_reps <- 20
test_mode <- Sys.getenv("R2_TEST") == "1"
if (test_mode) control_reps <- 2

collapse_categories <- function(d, items) {
  merged <- 0; dropped <- character(0)
  for (it in items) {
    repeat {
      tab <- table(factor(d[[it]], levels = sort(unique(na.omit(d[[it]])))), d$group)
      if (nrow(tab) < 2) { dropped <- c(dropped, it); break }
      empty <- which(apply(tab, 1, min) == 0)
      if (length(empty) == 0) break
      lev <- as.numeric(rownames(tab)); k <- empty[1]
      target <- if (k > 1) lev[k - 1] else lev[k + 1]
      d[[it]][which(d[[it]] == lev[k])] <- target
      merged <- merged + 1
    }
    d[[it]] <- as.integer(factor(d[[it]]))
  }
  list(data = d, items = setdiff(items, dropped), merged = merged, dropped = length(dropped))
}

row_of <- function(sequence, step, base, fit, previous) {
  ok <- !is.null(fit) && isTRUE(lavInspect(fit, "converged"))
  out <- cbind(sequence = sequence, step = step, base, converged = ok, admissible = NA, chisq = NA, df = NA,
               cfi = NA, rmsea = NA, srmr = NA, p_difference = NA)
  if (!ok) return(out)
  # FALSE when the solution has a negative variance or a non-positive-definite matrix
  out$admissible <- suppressWarnings(isTRUE(lavInspect(fit, "post.check")))
  names_fit <- if (sequence == "ordinal") c("chisq.scaled", "df.scaled", "cfi.scaled", "rmsea.scaled", "srmr")
               else c("chisq.scaled", "df.scaled", "cfi.robust", "rmsea.robust", "srmr")
  fm <- tryCatch(fitMeasures(fit, names_fit), error = function(e) rep(NA, 5))
  out$chisq <- fm[[1]]; out$df <- fm[[2]]; out$cfi <- fm[[3]]; out$rmsea <- fm[[4]]; out$srmr <- fm[[5]]
  if (!is.null(previous) && isTRUE(lavInspect(previous, "converged"))) {
    p <- suppressWarnings(tryCatch(lavTestLRT(previous, fit)[2, "Pr(>Chisq)"], error = function(e) NA))
    out$p_difference <- p
  }
  out
}

fit_sequences <- function(d, items) {
  rows <- list(); loadings <- NULL

  # ordinal
  prep <- collapse_categories(d, items)
  base <- data.frame(items_used = length(prep$items), merged_categories = prep$merged, items_dropped = prep$dropped)
  previous <- NULL
  if (length(prep$items) >= 3) {
    model <- paste("f =~", paste(prep$items, collapse = " + "))
    for (s in names(ordinal_steps)) {
      # warnings are kept out of the control flow: a warning handler in tryCatch would
      # discard a fit that converged, and whether a solution is admissible is recorded below
      fit <- suppressWarnings(tryCatch(measEq.syntax(configural.model = model, data = prep$data, ordered = prep$items,
                                    parameterization = "delta", ID.fac = "std.lv",
                                    ID.cat = "Wu.Estabrook.2016", group = "group",
                                    group.equal = ordinal_steps[[s]], return.fit = TRUE, estimator = "WLSMV"),
                      error = function(e) NULL))
      rows[[length(rows) + 1]] <- row_of("ordinal", s, base, fit, previous)
      if (s == "configural" && !is.null(fit) && isTRUE(lavInspect(fit, "converged"))) {
        ss <- standardizedSolution(fit); ss <- ss[ss$op == "=~", ]
        loadings <- reshape(ss[, c("rhs", "group", "est.std")], idvar = "rhs", timevar = "group", direction = "wide")
      }
      previous <- fit
    }
  }

  # continuous: no merging, but an item with no variance in a group cannot be modelled
  flat <- sapply(items, function(it) min(tapply(d[[it]], d$group, function(v) var(v, na.rm = TRUE))) == 0)
  use <- items[!flat]
  base <- data.frame(items_used = length(use), merged_categories = 0, items_dropped = sum(flat))
  previous <- NULL
  if (length(use) >= 3) {
    model <- paste("f =~", paste(use, collapse = " + "))
    for (s in names(continuous_steps)) {
      eq <- continuous_steps[[s]]
      fit <- suppressWarnings(tryCatch(if (identical(eq, "")) cfa(model, data = d, group = "group", estimator = "MLR", std.lv = TRUE)
                      else cfa(model, data = d, group = "group", estimator = "MLR", std.lv = TRUE, group.equal = eq),
                      error = function(e) NULL))
      rows[[length(rows) + 1]] <- row_of("continuous", s, base, fit, previous)
      previous <- fit
    }
  }
  list(fits = do.call(rbind, rows), loadings = loadings)
}

fits <- list(); loads <- list(); control <- list()
for (ds in datasets) {
  rec <- read.csv(file.path(root, paste0("r2_item_records_", ds, ".csv")), check.names = FALSE)
  rec <- rec[rec$source == "repeated_splits", ]
  items <- grep("_item_", names(rec), value = TRUE)
  scales <- split(items, sub("_item_.*", "", items))
  human <- rec[rec$model == "human", ]
  cells <- unique(rec[rec$model != "human", c("model", "arm")])
  if (test_mode) cells <- cells[cells$arm %in% c("SADP", "HC") & cells$model == "GPT-5.4", ]
  for (i in seq_len(nrow(cells))) {
    syn <- rec[rec$model == cells$model[i] & rec$arm == cells$arm[i], ]
    hum <- human[match(syn$respondent_id, human$respondent_id), ]
    for (sc in names(scales)) {
      d <- rbind(cbind(group = "human", hum[, scales[[sc]]]), cbind(group = "synthetic", syn[, scales[[sc]]]))
      r <- fit_sequences(d, scales[[sc]])
      key <- data.frame(dataset = ds, model = cells$model[i], arm = cells$arm[i], scale = sc, n = nrow(syn))
      if (!is.null(r$fits)) fits[[length(fits) + 1]] <- cbind(key, r$fits)
      if (!is.null(r$loadings) && ncol(r$loadings) == 3) {
        lh <- r$loadings[[2]]; ls <- r$loadings[[3]]
        loads[[length(loads) + 1]] <- cbind(key, items = length(lh),
          tucker_phi = sum(lh * ls) / sqrt(sum(lh^2) * sum(ls^2)),
          loading_mean_human = mean(lh), loading_mean_synthetic = mean(ls),
          loading_mad = mean(abs(lh - ls)), loading_r = suppressWarnings(cor(lh, ls)))
      }
    }
    cat(ds, cells$model[i], cells$arm[i], "done\n")
  }
  for (sc in names(scales)) for (rep in seq_len(control_reps)) {
    g <- sample(rep(c("half_a", "half_b"), length.out = nrow(human)))
    r <- fit_sequences(cbind(group = g, human[, scales[[sc]]]), scales[[sc]])
    if (!is.null(r$fits)) control[[length(control) + 1]] <- cbind(dataset = ds, scale = sc, rep = rep, n = nrow(human), r$fits)
  }
  cat(ds, "control done\n")
}
fits <- do.call(rbind, fits); loads <- do.call(rbind, loads); control <- do.call(rbind, control)
if (test_mode) {
  root <- tempdir()
  print(fits[, c("arm", "scale", "sequence", "step", "items_used", "merged_categories", "converged", "admissible", "cfi", "rmsea", "p_difference")], digits = 3)
  print(aggregate(cbind(cfi, p_difference) ~ dataset + sequence + step, control, mean))
}
write.csv(fits, file.path(root, "r2_invariance_fits.csv"), row.names = FALSE)
write.csv(loads, file.path(root, "r2_invariance_loadings.csv"), row.names = FALSE)
write.csv(control, file.path(root, "r2_invariance_control.csv"), row.names = FALSE)
cat("fits:", nrow(fits), "| converged:", sum(fits$converged), "| loadings rows:", nrow(loads),
    "| control fits:", nrow(control), "| control converged:", sum(control$converged), "\n")
