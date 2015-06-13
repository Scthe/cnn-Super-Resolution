# run through:
# rscript test\test_confirmation.R

if (!require("rjson")) {
  install.packages("rjson", repos="http://cran.rstudio.com/")
  library("rjson")
}


json_data <- fromJSON(file="test/data/test_cases.json")


sigmoid <- function(x){
    1 / (1 + exp(-x))
}

test_layer <- function(data, preproces_mean=FALSE, result_multiplier=0, decimal_places=2){
    cat("\n", data$name, ":\n")

    n_prev_filter_cnt <- data$n_prev_filter_cnt
    current_filter_count <- data$current_filter_count
    f_spatial_size <- data$f_spatial_size
    input_size <- c(data$input_w, data$input_h)
    out_size <- input_size - c(f_spatial_size, f_spatial_size) + c(1,1)
    out_dims <- c(out_size[2], out_size[1], current_filter_count)

    input_raw <- data$input
    output_raw <- data$output
    weights_raw <- data$weights
    bias <- data$bias
    input_modifier <- if(preproces_mean) mean(input_raw) else 0

    # preprocess data so that we can use native * operator for element-wise multiplication
    # (in json we have format that is suitable to be dumped into kernel indexing)
    input_vec <- c()
    for (filter_id in 1:n_prev_filter_cnt) {
        a <- (filter_id) %% n_prev_filter_cnt
        filter_data <- split(input_raw, 1:length(input_raw) %% n_prev_filter_cnt == a)$`TRUE`
        filter_data <- filter_data - input_modifier
        input_vec <- c(input_vec, filter_data)
    }
    input <- array(input_vec, c(data$input_w, data$input_h, n_prev_filter_cnt))
    # print(round(input, 3))

    # create submatrices of size f_spatial_size^2 * n_prev_filter_cnt
    # print(out_size)
    sub_views <- list()
    for (dy in 1:out_size[2]) {
    for (dx in 1:out_size[1]) {
        # if (dx != 1 || dy != 1) next
        end_dx <- dx + f_spatial_size - 1
        end_dy <- dy + f_spatial_size - 1
        sub_view <- input[dx:end_dx, dy:end_dy,]
        sub_views[[length(sub_views)+1]] <- sub_view
        # print(round(sub_view, 3))
        # cat("\nSUBVIEW: ", dx, ":", end_dx, ", ", dy, ":",end_dy, "\n")
    }
    }

    # weights_raw_ <- c(11,21,31, 12,22,32, 13,23,33, 14,24,34, 15,25,35, 16,26,36, 17,27,37, 18,28,38, 19,29,39)
    # weights_raw <- c()
    # for(x in weights_raw_){weights_raw <- c(weights_raw, x+100,x+200)}
    # print(weights_raw)

    # weights
    weights_vec <- c()
    idxs <- c()
    for (filter_id in 1:current_filter_count) {
        a <- filter_id %% current_filter_count
        weights_for_filter = split(weights_raw, 1:length(weights_raw) %% current_filter_count == a)$`TRUE`
        # cat("weights_for_filter",filter_id,":\n",weights_for_filter,"\n")

        for(i in 1:length(weights_for_filter)){
            # a <- (i-1) %/% (f_spatial_size*f_spatial_size)
            b <- (i-1) %/% f_spatial_size
            c <- (i-1) %% f_spatial_size
            d <- filter_id-1
            idx <- c * f_spatial_size * n_prev_filter_cnt * current_filter_count +
                #    a * n_prev_filter_cnt * current_filter_count +
                   b * current_filter_count +
                   d

            idxs = c(idxs, filter_id*1000+a*100+b*10+c)
            # weights_vec[idx+1] = idx
            weights_vec[idx+1] = weights_for_filter[i]
        }
    }
    # cat("\nfinal weights_vec\n", weights_vec, "\n\n")
    # cat("idx \n", idxs, "\n\n")
    # weights_vec[54] <- 555
    weights <- array(weights_vec, c(current_filter_count, f_spatial_size, f_spatial_size, n_prev_filter_cnt))
    # weights <- array(weights_raw, c(current_filter_count, f_spatial_size, f_spatial_size, n_prev_filter_cnt))

    # for (filter_id in 1:current_filter_count) {
        # cat("##############################################\n")
        # print(filter_id)
        # filter_weight <- weights[filter_id,,,]
        # print(filter_weight)
    # }

    # return(0)

    # output
    output_vec <- c()
    for (filter_id in 1:current_filter_count) {
        a <- filter_id %% current_filter_count
        output_for_filter = split(output_raw, 1:length(output_raw) %% current_filter_count == a)$`TRUE`
        output_vec <- c(output_vec, output_for_filter)
    }
    output <- array(output_vec, c(out_dims[1], out_dims[2], current_filter_count))


    #
    print("***********")
    result <- c()
    for (filter_id in 1:current_filter_count) {
        # if (filter_id != 3) next

        filter_weight <- weights[filter_id,,,]
        # print(filter_weight)

        B <- bias[filter_id]
        for (sub_view in sub_views) {
            # print(round(sub_view,3))
            res <- sum(sub_view * filter_weight) + B
            if(result_multiplier != 0){
                res <- res * result_multiplier
            }else{
                res <- sigmoid(res) # NOT HERE ?
            }
            result <- c(result, res)
            # print(res)
        }
    }



    # cat("RES: ", round(result, decimal_places), "\n")
    # cat("EXP: ", round(data$output, decimal_places), "\n")


    # cat("RES:\n")
    # print(result)
    # print(array(round(result, decimal_places), out_dims))
    cat("EXP:\n")
    print(array(round(output, decimal_places), out_dims))

    # cat("obsolete data: ", sum(result-data$output) == 0) # abs, etc.
    # cat("exp:  0.097259 0.12784 0.131725 0.062449 0.151116 0.080854\n")

    cat("**********\nDIFF\n")
    print(round(result-output,2))

    result
}

# cat("--------\nlayer 2 data set 1\n")
# l2_ds1 <- test_layer_2(json_data$layer_1, json_data$layer_2$data_set_1)

# cat("\n--------\nlayer 2 data set 1\n")
# l2_ds2 <- test_layer_2(json_data$layer_1, json_data$layer_2$data_set_2) # TODO submartices
# cat("layer 2 data set 1: ", l2_ds1[1], ", ", l2_ds1[2], "\n")
# cat("layer 2 data set 1: ", l2_ds1, "\n")
# cat("layer 2 data set 2: ", l2_ds2, "\n")

l1  <- test_layer(json_data$layer_1, preproces_mean = TRUE, decimal_places=3)
l21 <- test_layer(json_data$layer_2_data_set_1, decimal_places=3)
l22 <- test_layer(json_data$layer_2_data_set_2)
# l3  <- test_layer(json_data$layer_3, decimal_places=1) # raw without *255
l3  <- test_layer(json_data$layer_3, result_multiplier=256, decimal_places=1)
