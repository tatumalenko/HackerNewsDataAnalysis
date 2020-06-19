# HackerNewsDataAnalysis

A machine learning text classification app using categorical [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Bag-of-Words](https://en.wikipedia.org/wiki/Bag-of-words_model) model of HackerNews posts according to their post types, see [./res/a2.pdf](./res/a2.pdf) for more details.

## Dependencies
This app uses Python 3.7+, `scikit-learn`, `nltk`, `numpy`, and `matplotlib`. To see the list of all dependencies (including indirect dependencies), consult `requirements.txt`.

To install the dependencies, navigate to the root directory and run:
```shell
$ pip install -r requirements.txt
```

## Getting Started
To run the app, navigate to the root directory and run:
```shell 
$ python app.py
```

This will run all experiments specified in the [handout](./res/a2.pdf) using the provided data in [./res/hns_2018_2019.csv](hns_2018_2019.csv).

To customize the default behaviour of the app, change the configuration file values of the [./settings.ini](./settings.ini) file.

### Changing [settings.ini](./settings.ini)

There are many configuration options you can change in [settings.ini](./settings.ini). Details of each option is provided in [Configuration Options](#configuration-options). Some examples are provided below.

#### Change the train and test data used

Set `in_path_data` in `[Model]`
```ini
[Model]
in_path_data = ./res/new_data_file.csv
```

#### Change the metric averaging strategy

Set `average` in `[Metric]`
```ini
[Metric]
average = weighted
```

#### Change the plotting backend
Set `backend` in `[Plot]`
```ini
[Plot]
backend = MacOSX
```

## Configuration Options
```ini
[Model]
in_path_data = [absolute or relative path to train and test data]
out_path_model = [absolute or relative path to output baseline model]
out_path_remove_words = [absolute or relative path to output remove-words model]
out_path_vocabulary =  [absolute or relative path to output model vocabulary]
out_path_wordlength =  [absolute or relative path to output wordlength model]
in_path_stopwords =  [absolute or relative path to stop-words used in stopword model]
out_path_stopword =  [absolute or relative path to output stopword model]
labels = [post type labels used to consider for printing purposes]
word_weights = [comma separated word and weight values to consider when scoring]
data_test_year = [year used to filter for input train data]
data_train_year = [year used to filter for input test data]
index_title = [column index in input data containing the title of a post]
index_class = [column index in input data containing the class (type) of a post]
index_created_at = [column index in input data containing the created at date of a post]
wordlength_min = [inclusive min value of wordlength model]
wordlength_max = [inclusive max value of wordlength model]
smoothing = [smoothing value]

[Metric]
average = [averaging strategy to use in multiclass metrics, options are 'weighted', 'micro', 'macro', see more details at: https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification]
beta = [f-measure beta coefficient value between 0.0-1.0]
out_path_baseline = [absolute or relative path output baseline performance result]
out_path_stopword = [absolute or relative path output stopword performance result]
out_path_wordlength = [absolute or relative path output wordlength performance result]

[Plot]
dpi = [dots per inch, resolution]
linewidth = [plot linewidth value]
fontsize = [plot character fontsize value]
markersize = [plot marker size value]
backend = [graphical rendering backend, see options available at: https://matplotlib.org/faq/usage_faq.html#what-is-a-backend]
```