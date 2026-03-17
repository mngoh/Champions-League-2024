# Champions League Team Goals Predictions
**Author:** Martin Ngoh

## Background
This self-guided project uses Champions League team data from the 2008/09 to 2022/23 seasons to 
generate a machine learning model that can accurately predict the total goals scored by teams competing 
in the Champions League.


## Results
- **Best Model**: The XGB Regressor model performed the best on the training data with an RMSE of 
0.647 (logged target).
- **Testing on Historical Data**: The model was tested on the 2007/08 Champions League data, 
and the results were less impressive, with an RMSE of 0.854. This indicates that the model is 
overfitted and that more hyperparameter tuning should be explored.

## Next Steps
- **Feature Engineering**: Consider expanding the feature set to improve the model's generalization.
- **Model Testing**: Explore and test new models, such as neural networks, to see if they perform better.
- **Continued Refinement**: Despite being a side project, the current results are satisfactory. 
Further refinements and experiments will continue to improve the model.

## References
- **FBREF**: Sports statistics website

## Project Structure
- `0_data_scrape_08_22.ipynb`: Notebook for data scraping.
- `1_eda.ipynb`: Notebook for exploratory data analysis.
- `2_predictions.ipynb`: Notebook for making predictions.
- `3_score_model.ipynb`: Notebook for scoring the model.
- `data/`: Directory containing raw and processed data files.
- `models/`: Directory containing saved models.
- `plots/`: Directory containing generated plots.
- `helper.py`: Script containing utility functions.
- `README.txt`: Project documentation.
