import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.metrics import recall_score, precision_score

def show_hist(col,title=None):
  text =  'Stats:' + '\n\n' #הוספת טקסט שמתאר את כל הנתונים הסטטיים על העמודה
  text +=  'Mean: ' + str(round(col.mean(), 2)) + '\n'
  text += 'Median: ' + str(round(col.median(), 2)) + '\n'
  text += 'Mode: ' + str(list(col.mode().values)[0]) + '\n'
  text += 'Std dev: ' + str(round(col.std(), 2)) + '\n'
  text += 'Skew: ' + str(round(col.skew(), 2)) + '\n'

  bn = round(col.count() ** (1/3)) *2 #חישוב הבינים לפי נוסחא שבחרנו 

  col.plot(kind='hist', bins = bn)
  plt.axvline(col.mean(), color='k', linestyle='dashed', linewidth=1)#הצגת קו הממוצע בצבע שחור
  plt.axvline(col.median(), color='red', linestyle='dashed', linewidth=1) #הצגת קו החציון בצבע אדום
  plt.text(0.95, 0.45, text, fontsize=12, transform=plt.gcf().transFigure) #העיצוב של המקרא
  plt.title(title, fontsize=16, fontweight="bold"); #העיצוב של הכותרת
  #plt.xlabel(col) #למה אני לא מצליח להוסיף את תיאור ציר האיקס?

def show_box(col):  
    text =  'Stats:' + '\n\n' #שם המקרא
    text +=  'quantile 25%: ' + str(round(col.quantile(.25), 2)) + '\n'
    text += 'quantile 50%: ' + str(round(col.quantile(.50), 2)) + '\n'
    text += 'quantile 75%: ' + str(round(col.quantile(.75), 2)) + '\n'
    text += 'iqr: ' + str(round(col.quantile(.75)-col.quantile(.25), 2)) + '\n'#חישוב הטווח בין אחוזון 25 ל-75
    text += 'Median: ' + str(round(col.median(), 2)) + '\n'

    plt.text(0.95, 0.55, text, fontsize=12, transform=plt.gcf().transFigure)
    col.plot(kind='box', vert=False);

def show_counts(column1, column2=None):
    ax = sns.countplot(x = column1, hue=column2);
    for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.15, p.get_height()), ha='center', va='top', color='white', size=12)

def show_bar(df):
    ax = df.plot(kind='bar');
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(),2), (p.get_x()+0.15, p.get_height()), ha='center', va='top', color='white', size=10)

def get_numeric_details(df, sort_column='mean', sort_order=False):
    res = pd.DataFrame()
    numeric_columns = df.select_dtypes(include='number').columns

    for column in numeric_columns:
        data = pd.DataFrame({'min':[df[column].min()],
                             'quantile 25':df[column].quantile(.25),
                             'quantile 50':df[column].quantile(.50),
                             'quantile 75':df[column].quantile(.75),
                             'max':df[column].max(),
                             'mean':df[column].mean(),
                             'median':df[column].median(),
                             'mode': ','.join(str(obj) for obj in list(df[column].mode().values)),
                             'std':df[column].std(),
                             'count':df[column].count(),
                             'nunique':df[column].nunique(),
                             'skew':df[column].skew()
                            },index=[column])
        res = res.append(data)
    return res

def show_distribution(df, column):
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    
    text  = 'Std dev: ' + str(round(df[column].std(), 2)) + '\n'
    text += 'Mean: ' + str(round(df[column].mean(), 2)) + '\n'
    text += 'Median: ' + str(round(df[column].median(), 2)) + '\n'

    fig.suptitle(column)
    
    # Histogram
    num_bins = int(round(df[column].count()**(1/3)*2, 0))
    sns.histplot(df[column], bins=num_bins, ax=axes[0])
    axes[0].set_title('Distribution I')
    axes[0].text(0.35, 0.5, text, fontsize=10, transform=plt.gcf().transFigure)
    
    # Box Plot
    sns.boxplot(x=column,
                data=df, 
                showmeans=True,
                meanline=True,
                meanprops={'color':'white'},
                ax=axes[1])
    axes[1].set_title('Distribution II')

    plt.show()

# If I want to see the distribution on all the columns i can take this furmole
#for column in numeric_columns:
#helpers.show_distribution(df,column)

def cat_count (df,column) :
  df1= df[column].value_counts().to_frame().rename(columns={column: 'count'})
  df2= df[column].value_counts(normalize=True).to_frame().rename(columns={column: 'pct'})
  
  output= df1.merge(df2,how='inner',left_index=True,right_index=True).style.format({'pct':'{:,.2%}'.format})
  return output

def show_counts(df,column,title=None):
  plt.figure(figsize=(15,8))
  plt.title(title, fontsize=16, fontweight="bold")
  ax = sns.countplot(x = column,
                     data=df,
                     order= df[column].value_counts().index)
  for p in ax.patches: 
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()+100))
  
  def ci_unknown_std(sample, alpha):
    sample_size = sample.size
    sample_mean = sample.mean()
    t_critical  = stats.t.ppf(q = 1-alpha/2, df=sample.size-1)  
    sample_stdev = sample.std(ddof=1) 
    sigma = sample_stdev/math.sqrt(sample.size) 
    margin_of_error = t_critical * sigma
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)  
    return confidence_interval

def ci_known_std(sample, pop_stdev, alpha):
    sample_mean = sample.mean()
    sample_size = sample.size
    z_critical  = stats.norm.ppf(q = 1-alpha/2)
    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean,
                           sample_mean + margin_of_error)
    return confidence_interval

def find_categorical_numeric_columns(df):
    # categorical_columns
    categorical_columns = df.select_dtypes(include='object').columns

    # numeric_columns
    numeric_columns = df.select_dtypes(include='number').columns

    # Print categorical columns
    print("Categorical columns:")
    for column in categorical_columns:
        print(f"- {column}")

    print("\nNumeric columns:")
    for column in numeric_columns:
        print(f"- {column}")

def calc_anova(df, group_column, values_column):
    # Get list of unique group values in provided column
    unique_group_values = df[group_column].drop_duplicates().to_list()

    # Iterate through each unique group value and filter the dataframe to get the values
    # of the provided column for that group, then store them in a list
    values_by_group = []
    for group_value in unique_group_values:
        group_filter = df[group_column] == group_value
        values_by_group.append(df[values_column][group_filter])

    # Perform ANOVA test on the list of value arrays using the `f_oneway` function from the `scipy.stats` module
    return f_oneway(*values_by_group)


def calculate_scores(models, model_names, X_test, y_test):
    scores = {}
    for name, model in zip(model_names, models):
        y_pred = model.predict(X_test)
        accuracy = round(model.score(X_test, y_test) * 100, 2)
        recall = round(recall_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred) * 100, 2)
        scores[name] = {"Accuracy": f"{accuracy}%","Recall": f"{recall}%","Precision": f"{precision}%"}
    return scores

def create_score_dataframe(scores):
    df = pd.DataFrame(scores).style.highlight_max(color="green", axis=1).highlight_min(color="red", axis=1)
    return df

# example usage
#model_names = ["Decision Tree", "Random Forest", "XGBoost", "Stacked Model"]
#models = [dt, rf, xgb, stcked_model]

#scores = helpers.calculate_scores(models, model_names, X_test, y_test)
#df_scores = helpers.create_score_dataframe(scores)
#df_scores
