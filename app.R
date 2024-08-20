library(shiny)
library(ggplot2)
library(caret)
library(tidyverse)
library(dplyr)
library(DT)

# Load the dataset and preprocess
# Set working directory to the directory where your CSV file is located

insurance <- read.csv("insurance.csv")
insurance$sex <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)

# Train and test split
set.seed(123)
train_index <- createDataPartition(insurance$charges, p = 0.70, list = FALSE)
train_data <- insurance[train_index, ]
test_data <- insurance[-train_index, ]

# Train the model with cross-validation
Train_Model2 <- train(
  form = charges ~ age + sex + bmi + children + smoker + region,
  data = train_data,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

# Define UI
ui <- fluidPage(
  titlePanel("Insurance Charges Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("age", "Age:", value = 40, min = 18, max = 100),
      selectInput("sex", "Sex:", choices = c("male", "female")),
      numericInput("bmi", "BMI:", value = 25, min = 10, max = 50),
      numericInput("children", "Children:", value = 0, min = 0, max = 10),
      selectInput("smoker", "Smoker:", choices = c("yes", "no")),
      selectInput("region", "Region:", choices = c("northwest", "northeast", "southwest", "southeast")),
      actionButton("predict_btn", "Predict Charges")
    ),
    
    mainPanel(
      plotOutput("scatter_plot"),
      DTOutput("results_table"),
      textOutput("mae"),
      textOutput("rmse"),
      textOutput("r_squared")
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Reactive expression to make predictions based on user input
  prediction <- eventReactive(input$predict_btn, {
    new_data <- data.frame(
      age = input$age,
      sex = as.factor(input$sex),
      bmi = input$bmi,
      children = input$children,
      smoker = as.factor(input$smoker),
      region = as.factor(input$region)
    )
    predict(Train_Model2, new_data)
  })
  
  # Reactive expression to create scatter plot with prediction
  output$scatter_plot <- renderPlot({
    ggplot(test_data, aes(x = bmi, y = charges, color = smoker)) +
      geom_point() +
      geom_smooth(method = "lm", se = FALSE, color = "blue") +
      geom_hline(yintercept = prediction(), linetype = "dashed", color = "red") +
      labs(title = "Scatter Plot of BMI vs Charges with Prediction",
           x = "BMI",
           y = "Charges") +
      theme_minimal() +
      theme(legend.position = "bottom")
  })
  
  # Display results in a table
  output$results_table <- renderDT({
    df <- data.frame(
      Age = input$age,
      Sex = input$sex,
      BMI = input$bmi,
      Children = input$children,
      Smoker = input$smoker,
      Region = input$region,
      Predicted_Charges = prediction()
    )
    datatable(df, options = list(
      pageLength = 5,
      autoWidth = TRUE,
      dom = 't'
    )) %>%
      formatStyle('Predicted_Charges', backgroundColor = styleInterval(c(5000, 10000), c('lightblue', 'lightgreen', 'lightcoral')))
  })
  
  # Display model evaluation metrics
  output$mae <- renderText({
    pred <- predict(Train_Model2, test_data)
    actual <- test_data$charges
    mae <- mean(abs(actual - pred))
    paste("Mean Absolute Error (MAE):", round(mae, 2))
  })
  
  output$rmse <- renderText({
    pred <- predict(Train_Model2, test_data)
    actual <- test_data$charges
    rmse <- sqrt(mean((actual - pred)^2))
    paste("Root Mean Squared Error (RMSE):", round(rmse, 2))
  })
  
  output$r_squared <- renderText({
    pred <- predict(Train_Model2, test_data)
    actual <- test_data$charges
    rss <- sum((pred - actual)^2)
    tss <- sum((actual - mean(actual))^2)
    r_squared <- 1 - (rss / tss)
    paste("R-squared (R²):", round(r_squared, 2))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
