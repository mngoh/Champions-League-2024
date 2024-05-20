# Helper Functions 

# Import Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Frequency Plot 
def freq_plot(x='None', df=None):
    sns.displot(df, x=x, kde=True, palette='cool')
    plt.xlabel(x)
    plt.title(f'Distribution of {x}')
    plt.show()

# Bar Plot
def bar_plot(x, y, df):
    plt.figure(figsize=(7, 6))  
    sns.barplot(data=df, x=x, y=y, palette='cool')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} by {x}')
    plt.xticks(rotation=45)  
    plt.show()    

# Scatter Plot
def scatter_plot(x, y, df):
    plt.figure(figsize=(7, 6))  
    sns.scatterplot(data=df, x=x, y=y, palette='cool')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x}')
    plt.show()

# Box Plot
def box_plot(x, df, save_out = False, title = 'None'):
    plt.figure(figsize=(7,6))
    sns.boxplot(y=x, data=df, palette='cool')
    plt.ylabel(x)
    plt.title(f'Distribution of {x}')
    if save_out: 
        plt.savefig(f'plots/{title}')
        plt.show()
    else:
        plt.show()


# Heat Map
def heat_map(corr):
    plt.figure(figsize=(10,10))
    sns.heatmap(data=corr, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
    plt.title('Correlation Matrix')
    plt.show()

# Accuracy Metrics 
def regression_acc(actuals, preds, model = 'None'):
    if model == 'None':
        print('Add Model Type as: model = XXX')
    else:
        mae = mean_absolute_error(y_true = actuals, y_pred = preds)
        mse = mean_squared_error(y_true = actuals, y_pred = preds)
        rmse = np.sqrt(mse)
        metrics = pd.DataFrame({
            'MAE':[mae],
            'MSE':[mse],
            'RMSE':[rmse],
            'Model':model
        })
        metrics = metrics.round(3)
        print(metrics)
        return metrics
    

# Residuals Plot 
def residuals_plot(actuals, preds, model_name='Model'):
    residuals = actuals - preds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=preds, y=residuals, palette='coolwarm', hue=residuals, ax=axes[0], legend=False)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals Plot for {model_name}')
    sns.scatterplot(x=actuals, y=preds, ax=axes[1], palette='cool')
    axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')  
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title(f'Actual vs Predicted Plot for {model_name}')
    plt.tight_layout()
    plt.savefig(f'plots/residual_plot_for_{model_name}')
    plt.show()
