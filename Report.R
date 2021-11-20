# Creation of the dataset and preprocessing

## Libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gridExtra)

## Download of the MovieLens 10M dataset
## https://grouplens.org/datasets/movielens/10m/
## http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

## Creation of training and test set. The validation will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

## Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

## Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
save.image("Movies_workspace.RData")



# Data Analysis

load("Movies_workspace.RData")

## Dimension, labels, structure and look on the first 6 entries
dim(edx)
names(edx)
str(edx)
head(edx)

## Search for NA values
edx %>%
  filter(is.na(rating)) %>%
  summarize(count = n())

## Transform the data into a matrix to study its distribution
## We show only a sample of 100 movies and users
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

## Graphics of the distribution of movies and users
plot_movies <- edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

plot_users <- edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

grid.arrange(plot_movies,
             plot_users,
             ncol = 2)



# Modeling

## Definition of the residual mean squared error (RMSE) function:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Model 1: Regularized Movie and User effects
### Lambda parameter tuning with 10-fold cross validation
lambdas <- seq(0, 10, 0.25)
folds <- createFolds(edx$rating, k = 10, list = TRUE, returnTrain = FALSE)

rmses <- sapply(lambdas, function(l){
  rmse <- sapply(folds, function(i){
    train_set_star <- edx[i,]
    test_set_star <- edx[-i,] %>%
      semi_join(train_set_star, by="movieId") %>%
      semi_join(train_set_star, by="userId")
    
    mu <- mean(train_set_star$rating)
    b_i <- train_set_star %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_set_star %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - mu - b_i)/(n()+l))
    
    predicted_ratings <- test_set_star %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
    return(RMSE(test_set_star$rating, predicted_ratings))
  })
  return(mean(rmse))
})

### RMSEs of each lambda and the best one, with the lowest RMSE
lambdas_1 <- qplot(lambdas, rmses)
min_lambdas_1 <- lambdas[which.min(rmses)]

### With the best lambda, we train our model using the entire training set.
lambda <- lambdas[which.min(rmses)]
mu_hat <- mean(edx$rating)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))

### Using the fitted model, we predict on the validation set
predicted_ratings <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

### Calculating the RMSE and saving it
model_1_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- tibble(method = "Model 1: Regularized Movie + User Effect",
                       RMSE = model_1_rmse)


## Model 2: Regularized Movie, User and Genre effects
### Lambda parameter tuning with 10-fold cross validation
rmses <- sapply(lambdas, function(l){
  rmse <- sapply(folds, function(i){
    train_set_star <- edx[i,]
    test_set_star <- edx[-i,] %>%
      semi_join(train_set_star, by="movieId") %>%
      semi_join(train_set_star, by="userId")
    
    mu <- mean(train_set_star$rating)
    b_i <- train_set_star %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_set_star %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - mu - b_i)/(n()+l))
    b_g <- train_set_star %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
    
    predicted_ratings <- test_set_star %>%
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      left_join(b_g, by="genres") %>%
      mutate(pred = mu + b_i + b_u + b_g) %>%
      .$pred
    return(RMSE(test_set_star$rating, predicted_ratings))
  })
  return(mean(rmse))
})

### RMSEs of each lambda and the best one, with the lowest RMSE
lambdas_2 <- qplot(lambdas, rmses)
min_lambdas_2 <- lambdas[which.min(rmses)]

### With the best lambda, we train our model using the entire training set.
lambda <- lambdas[which.min(rmses)]
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda))
b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+lambda))

### Using the fitted model, we predict on the validation set and calculate its RMSE
predicted_ratings <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred

### Calculating the RMSE and saving it
model_2_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Model 2: Regularized Movie + User + Genre Effect",
                                 RMSE = model_2_rmse))



# Results
## Graphics of lambda tuning process
lambdas_1
min_lambdas_1

lambdas_2
min_lambdas_2

## Showing RMSE of both models
rmse_results %>% knitr::kable()
