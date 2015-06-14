# run the script with:
# rscript test\test_confirmation.R

if (!require("rjson")) {
  install.packages("rjson", repos="http://cran.rstudio.com/")
  library("rjson")
}


json_data <- fromJSON(file="test/data/test_cases.json")


sigmoid <- function(x){
    1 / (1 + exp(-x))
}

split_by_columns <- function(arr, column_count, as_lists=FALSE){
    if((length(arr) %% column_count) != 0){
        stop(sprintf("Error: Tried to divide array of size(%d) into %d columns", length(arr), column_count))
    }

    result <- c()
    for (i in 1:column_count) {
        a <- (i) %% column_count
        sub_arr <- split(arr, 1:length(arr) %% column_count == a)$`TRUE`
        result <- c(result, sub_arr)
    }

    if(length(arr) != length(result)) {
        stop(sprintf("Error: Length of provided array(%d) != length of result(%d)", length(arr), length(result)))
     }

    if(as_lists){
        result <- matrix(result, ncol=column_count)
    }

    result
}


test_layer <- function(data, preproces_mean=FALSE, result_multiplier=0, decimal_places=2){
    cat("\n", data$name, ":\n")

    n_prev_filter_cnt <- data$n_prev_filter_cnt
    current_filter_count <- data$current_filter_count
    f_spatial_size <- data$f_spatial_size
    input_size <- c(data$input_w, data$input_h)

    input_raw <- data$input
    output_raw <- data$output
    weights_raw <- data$weights
    bias <- data$bias

    out_size <- input_size - c(f_spatial_size, f_spatial_size) + c(1,1)
    out_dims <- c(out_size[2], out_size[1], current_filter_count)
    input_modifier <- if(preproces_mean) mean(input_raw) else 0
    # print(out_size)

    # preprocess data so that we can use native * operator for element-wise multiplication
    # (in json we have format that is suitable to be dumped into kernel indexing)
    input_vec <- split_by_columns(input_raw, n_prev_filter_cnt) - input_modifier
    input <- array(input_vec, c(input_size[1], input_size[2], n_prev_filter_cnt))
    # print(round(input, 3))

    # create submatrices of size f_spatial_size^2 * n_prev_filter_cnt
    sub_views <- list()
    for (dy in 1:out_size[2]) {
    for (dx in 1:out_size[1]) {
        end_dx <- dx + f_spatial_size - 1
        end_dy <- dy + f_spatial_size - 1
        sub_view <- input[dx:end_dx, dy:end_dy,]
        sub_views[[length(sub_views)+1]] <- sub_view
        # cat("SUBVIEW: ", dx, ":", end_dx, ", ", dy, ":",end_dy, "\n")
        # print(round(sub_view, 3))
    }
    }

    # weights
    weights_vec <- c()
    weights_by_filter <- split_by_columns(weights_raw, current_filter_count, as_lists=TRUE)
    for (filter_id in 1:current_filter_count) {
        ws <- weights_by_filter[,filter_id]
        # print(sprintf("Weights for filter %d (len=%d): %s", filter_id, length(ws), paste(ws, collapse=" ")))
        for(i in 1:length(ws)){
            # a <- (i-1) %/% (f_spatial_size*f_spatial_size)
            b <- (i-1) %/% f_spatial_size
            c <- (i-1) %% f_spatial_size
            d <- filter_id-1
            idx <- c * f_spatial_size * n_prev_filter_cnt * current_filter_count +
                #    a * n_prev_filter_cnt * current_filter_count +
                   b * current_filter_count +
                   d

            weights_vec[idx+1] = ws[i]
        }
    }
    weights <- array(weights_vec, c(current_filter_count, f_spatial_size, f_spatial_size, n_prev_filter_cnt))

    # weights - debug print
    # for (filter_id in 1:current_filter_count) {
        # cat("Weights for filter", filter_id, ":\n")
        # print(weights[filter_id,,,])
    # }

    # execute
    result <- c()
    for (filter_id in 1:current_filter_count) {
        B <- bias[filter_id]
        filter_weight <- weights[filter_id,,,]
        # print(filter_weight)

        for (sub_view in sub_views) {
            # print(round(sub_view,3))
            res <- sum(sub_view * filter_weight) + B
            res <- if(result_multiplier != 0) res * result_multiplier
                   else sigmoid(res)
            result <- c(result, res)
        }
    }
    res_arr <- array(round(result, decimal_places), out_dims)

    # print status
    output_vec <- split_by_columns(output_raw, current_filter_count)
    output <- array(output_vec, c(out_dims[1], out_dims[2], current_filter_count))
    exp_arr <- array(round(output, decimal_places), out_dims)

    cat("DIFFERENCE - calculated result vs JSON output field (should be ~0 across the board):\n")
    print(round(result-output,2))
    # cat("RESULT:\n")
    # print(round(res_arr,2))
    # cat("EXPECTED:\n")
    # print(round(exp_arr,2))

    result
}

help_text <- "How to interpret results:\nResults have OUT_W*OUT_H*CURRENT_FILTER_COUNT numbers printed as OUT_W*OUT_H matrices. With the convention that data (in JSON) for each filter is in the respective column write content of each matrix (column-by-column) into single column (in JSON)."

cat("\n\n", help_text, "\n")

l1  <- test_layer(json_data$layer_1, preproces_mean = TRUE, decimal_places=3)
l21 <- test_layer(json_data$layer_2_data_set_1, decimal_places=3)
l22 <- test_layer(json_data$layer_2_data_set_2)
l3  <- test_layer(json_data$layer_3, result_multiplier=256, decimal_places=1)
