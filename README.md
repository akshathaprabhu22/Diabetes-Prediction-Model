# **How common is diabetes?**
<center><img src = "https://www.cifc.org/sites/g/files/vyhlif9361/files/media/patient-health-resources/image/1711/type-two-diabetes-facts-and-stats-you-need-to-know-1440x810.jpg"/></center>


Diabetes is one of the most common chronic diseases in the United States. It is reported to be the eighth leading cause of death in the United States, affecting over 38.4 million individuals annually. An individual diagnosed with diabetes can be prone to multiple other diseases. Diabetes is associated with severe complications such as heart disease, vision loss, limb amputation, and kidney disease, making prevention and effective management crucial.

In the United States, the disease also contributes singificantly to household expenditure. The average out-of-pocket cost for a 30-day supply was $58 per insulin fill in 2019. From both an economic and a healthcare perspective, the prevention of diabetes assumes paramount importance.

Diabetes prediction models not only help with providing insight into risk factors for individuals but they can also assist healthcare and public systems in predicting the demand for required drugs,therapies and patient care.

In this project, we will utilize the 2015 telephone survey data from the CDC to examine diabetes patterns in the United States and construct predictive models.


## **Data Source**

The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. For this project, a csv of the dataset available on Kaggle for the year 2015 was used.

Kaggle Source:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## **Preparing our data**

Let's begin by importing our data using the pandas module.


```python
import pandas as pd
db=pd.read_csv('/content/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
```

The following dataset was obtained from Kaggle. The data was derived from the Centers for Disease Control and Prevention (CDC) via their Behavioral Risk Factor Surveillance System (BRFSS) for the year 2015.


The variable are mostly binary due to closed ended questions resulting to yes or no responses, however, there are some quantitative and categorical variables also included. The variables and their descriptions can be seen below:



```python
db.head()
```





  <div id="df-0069f40f-8a13-470f-b9c3-e3f7e000723f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0069f40f-8a13-470f-b9c3-e3f7e000723f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0069f40f-8a13-470f-b9c3-e3f7e000723f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0069f40f-8a13-470f-b9c3-e3f7e000723f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5cf799e2-52ce-417a-bf24-b23f89f3f2b2">
  <button class="colab-df-quickchart" onclick="quickchart('df-5cf799e2-52ce-417a-bf24-b23f89f3f2b2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5cf799e2-52ce-417a-bf24-b23f89f3f2b2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>






*   Our data contains 22 columns and 70,692 data points.


```python
db.describe()
```





  <div id="df-20871208-6db2-4761-903e-3af2d6664ebf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diabetes_binary</th>
      <th>HighBP</th>
      <th>HighChol</th>
      <th>CholCheck</th>
      <th>BMI</th>
      <th>Smoker</th>
      <th>Stroke</th>
      <th>HeartDiseaseorAttack</th>
      <th>PhysActivity</th>
      <th>Fruits</th>
      <th>...</th>
      <th>AnyHealthcare</th>
      <th>NoDocbcCost</th>
      <th>GenHlth</th>
      <th>MentHlth</th>
      <th>PhysHlth</th>
      <th>DiffWalk</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>...</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
      <td>70692.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.500000</td>
      <td>0.563458</td>
      <td>0.525703</td>
      <td>0.975259</td>
      <td>29.856985</td>
      <td>0.475273</td>
      <td>0.062171</td>
      <td>0.147810</td>
      <td>0.703036</td>
      <td>0.611795</td>
      <td>...</td>
      <td>0.954960</td>
      <td>0.093914</td>
      <td>2.837082</td>
      <td>3.752037</td>
      <td>5.810417</td>
      <td>0.252730</td>
      <td>0.456997</td>
      <td>8.584055</td>
      <td>4.920953</td>
      <td>5.698311</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500004</td>
      <td>0.495960</td>
      <td>0.499342</td>
      <td>0.155336</td>
      <td>7.113954</td>
      <td>0.499392</td>
      <td>0.241468</td>
      <td>0.354914</td>
      <td>0.456924</td>
      <td>0.487345</td>
      <td>...</td>
      <td>0.207394</td>
      <td>0.291712</td>
      <td>1.113565</td>
      <td>8.155627</td>
      <td>10.062261</td>
      <td>0.434581</td>
      <td>0.498151</td>
      <td>2.852153</td>
      <td>1.029081</td>
      <td>2.175196</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>98.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-20871208-6db2-4761-903e-3af2d6664ebf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-20871208-6db2-4761-903e-3af2d6664ebf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-20871208-6db2-4761-903e-3af2d6664ebf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-cd18e11b-e31a-4451-92fd-63bc49b2a34e">
  <button class="colab-df-quickchart" onclick="quickchart('df-cd18e11b-e31a-4451-92fd-63bc49b2a34e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-cd18e11b-e31a-4451-92fd-63bc49b2a34e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>






*   There are 22 columns overall. Diabetes_binary is our predictive variable which takes a value of 1 if an individual is pre-diabetic or diabetic and 0 otherwise.


```python
db.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 70692 entries, 0 to 70691
    Data columns (total 22 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Diabetes_binary       70692 non-null  float64
     1   HighBP                70692 non-null  float64
     2   HighChol              70692 non-null  float64
     3   CholCheck             70692 non-null  float64
     4   BMI                   70692 non-null  float64
     5   Smoker                70692 non-null  float64
     6   Stroke                70692 non-null  float64
     7   HeartDiseaseorAttack  70692 non-null  float64
     8   PhysActivity          70692 non-null  float64
     9   Fruits                70692 non-null  float64
     10  Veggies               70692 non-null  float64
     11  HvyAlcoholConsump     70692 non-null  float64
     12  AnyHealthcare         70692 non-null  float64
     13  NoDocbcCost           70692 non-null  float64
     14  GenHlth               70692 non-null  float64
     15  MentHlth              70692 non-null  float64
     16  PhysHlth              70692 non-null  float64
     17  DiffWalk              70692 non-null  float64
     18  Sex                   70692 non-null  float64
     19  Age                   70692 non-null  float64
     20  Education             70692 non-null  float64
     21  Income                70692 non-null  float64
    dtypes: float64(22)
    memory usage: 11.9 MB


Our dataset is pre-cleaned hence there are no missing/null values.


```python
db.isnull().sum()
```




    Diabetes_binary         0
    HighBP                  0
    HighChol                0
    CholCheck               0
    BMI                     0
    Smoker                  0
    Stroke                  0
    HeartDiseaseorAttack    0
    PhysActivity            0
    Fruits                  0
    Veggies                 0
    HvyAlcoholConsump       0
    AnyHealthcare           0
    NoDocbcCost             0
    GenHlth                 0
    MentHlth                0
    PhysHlth                0
    DiffWalk                0
    Sex                     0
    Age                     0
    Education               0
    Income                  0
    dtype: int64




```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
```

## **Exploratory Data Analysis: Which factors impact diabetes the most?**

Given that our data is already pre-cleaned we can move on to analyzing how our variables correlate with the diabetes indicator.

We can split our data into two type of indicators: Demographic and non-demographic. Using python's seaborn tool, we will look at how the proportion of diabetes is distributed across various demographic categories.

The demographic indicators in the data:

*   Sex
*   Age
*   Income
*   AnyHealthcare
*   Education


```python
by_age = db.groupby(['Age'])['Diabetes_binary'].mean().reset_index()

by_Income = db.groupby(['Income'])['Diabetes_binary'].mean().reset_index()

by_Educ = db.groupby(['Education'])['Diabetes_binary'].mean().reset_index()
```

Trend by Age, Income and Education:


*  The bar graphs below plot the frequency of diabetes in each demographic category, which will help us visualize how prevalence differs within groups.




```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

sns.barplot(data=by_age, x='Age', y='Diabetes_binary', palette= 'Blues',ax = ax1)
ax1.set_title('Distribution of diabetes by Age')
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')

sns.barplot(data=by_Income, x='Income', y='Diabetes_binary', palette= 'Blues',ax = ax2)
ax2.set_title('Distribution of diabetes by Income')
ax2.set_xlabel('Income')
ax2.set_ylabel('Frequency')

sns.barplot(data=by_Educ, x='Education', y='Diabetes_binary', palette= 'Blues',ax = ax3)
ax3.set_title('Distribution of diabetes by Education')
ax3.set_xlabel('Education')
ax3.set_ylabel('Frequency')
```

    <ipython-input-12-23e3f8f6b29d>:7: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=by_age, x='Age', y='Diabetes_binary', palette= 'Blues',ax = ax1)
    <ipython-input-12-23e3f8f6b29d>:12: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=by_Income, x='Income', y='Diabetes_binary', palette= 'Blues',ax = ax2)
    <ipython-input-12-23e3f8f6b29d>:17: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=by_Educ, x='Education', y='Diabetes_binary', palette= 'Blues',ax = ax3)





    Text(0, 0.5, 'Frequency')




    
![png](output_16_2.png)
    



*  As expected, we see an upward trend in diabetes with age. Older participants
are more likely to have diabetes. Individuals in age brackets 9.0 and beyond seem more likely to be at risk of diabetes The 9.0 group here accounts for participants in the age bracket of 60-64, groups 9.0 and above indicates for individuals aged 60 and above.

*   On the other hand, we see a reverse trend with income. Participants from lower income groups are more prone to diabetes. This may be due to lack of accessibility to healthy eating options or good healthcare.

*   The trend with education is mixed. We see a spike around groups 2.0 and 3.0.Groups 2 and 3 contain groups that have recieved basic schooling, either only elementary or a medium level of schooling. These groups are more likely to have diabetes in contrast to groups that are more educated.




```python
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,8))
fig.subplots_adjust(hspace=1, wspace= 1)

by_sex = db.groupby(['Sex'])['Diabetes_binary'].mean().reset_index()
by_sex
colors = ['lightskyblue','cornflowerblue']

labels = by_sex['Sex']
values = by_sex['Diabetes_binary']
ax1.pie(values,labels = labels,autopct = '%1.1f%%',wedgeprops=dict(width=0.6),colors = colors)
ax1.set_title('Diabetes distribution by sex')

by_healthcare = db.groupby(['AnyHealthcare'])['Diabetes_binary'].mean().reset_index()
colors = ['lightskyblue','cornflowerblue']
labels = by_healthcare['AnyHealthcare']
values = by_healthcare['Diabetes_binary']
ax2.pie(values,labels = labels,autopct = '%1.1f%%',wedgeprops=dict(width=0.6),colors = colors)
ax2.set_title('Diabetes distribution by healthcare status')
```




    Text(0.5, 1.0, 'Diabetes distribution by healthcare status')




    
![png](output_18_1.png)
    


*   There seems to be no significant variation in diabetes across Sex. The split for sex and diabetes is almost 50-50. Hence, it may be better to drop this indicator from our dataset.
*   Similarly, Healthcare status is also not a strong indicator of diabetes. There is almost equal prevalance of diabetes amongst both groups, participants that have healthcare and participants that do not have healthcare.

How does diabetes correlate with healthcare indicators?


```python
average_bmi = db.groupby('Diabetes_binary')['BMI'].mean().reset_index()
fig, ax = plt.subplots()
BMI_Bar_Colors = ['green', 'red']

sns.boxplot(x='Diabetes_binary', y='BMI', showfliers = False, data=db, palette='Blues')
ax.set_ylabel('BMI')
ax.set_xlabel('Diabetes')
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Diabetes', 'Yes Diabetes'])
ax.set_title(['Average BMI by Diabetes'])

no_diabetes_midpoint = average_bmi.loc[average_bmi['Diabetes_binary'] == 0, 'BMI'].values[0]
yes_diabetes_midpoint = average_bmi.loc[average_bmi['Diabetes_binary'] == 1, 'BMI'].values[0]

ax.text(0, no_diabetes_midpoint, f'Mean BMI: {no_diabetes_midpoint:.2f}', ha='center', va='center', fontsize=10, color='black')
ax.text(1, yes_diabetes_midpoint, f'Mean BMI: {yes_diabetes_midpoint:.2f}', ha='center', va='center', fontsize=10, color='black')

plt.show()
```

    <ipython-input-14-5ab0ddf71164>:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(x='Diabetes_binary', y='BMI', showfliers = False, data=db, palette='Blues')



    
![png](output_21_1.png)
    



*   On average, individuals with diabetes seem to have higher BMI (31.94) compared to individuals without diabetes (27.77). Suggesting that healthcare choices also make a difference in causing diabetes.


# **Correlation Plot: How do our independent variables correlate with the diabetes indicator?**

Given that we have 21 independent variables to pick from for a prediction model, it might be more useful to visualize the correlation to understand which variables help explain our dependent variable the best. Simultaneously, we can also drop variables that are poorly correlated or inter-correlated in order to avoid bias.


```python
correlation= db.corr()
print("Correlation",correlation)
```

    Correlation                       Diabetes_binary    HighBP  HighChol  CholCheck  \
    Diabetes_binary              1.000000  0.381516  0.289213   0.115382   
    HighBP                       0.381516  1.000000  0.316515   0.103283   
    HighChol                     0.289213  0.316515  1.000000   0.085981   
    CholCheck                    0.115382  0.103283  0.085981   1.000000   
    BMI                          0.293373  0.241019  0.131309   0.045648   
    Smoker                       0.085999  0.087438  0.093398  -0.004331   
    Stroke                       0.125427  0.129060  0.099786   0.022529   
    HeartDiseaseorAttack         0.211523  0.210750  0.181187   0.043497   
    PhysActivity                -0.158666 -0.136102 -0.090453  -0.008249   
    Fruits                      -0.054077 -0.040852 -0.047384   0.017384   
    Veggies                     -0.079293 -0.066624 -0.042836   0.000349   
    HvyAlcoholConsump           -0.094853 -0.027030 -0.025443  -0.027146   
    AnyHealthcare                0.023191  0.035764  0.031532   0.106800   
    NoDocbcCost                  0.040977  0.026517  0.033199  -0.062669   
    GenHlth                      0.407612  0.320540  0.237778   0.059213   
    MentHlth                     0.087029  0.064294  0.083881  -0.010660   
    PhysHlth                     0.213081  0.173922  0.142610   0.034540   
    DiffWalk                     0.272646  0.234784  0.162043   0.044430   
    Sex                          0.044413  0.040819  0.017324  -0.007991   
    Age                          0.278738  0.338132  0.240338   0.101743   
    Education                   -0.170481 -0.141643 -0.084386  -0.008695   
    Income                      -0.224449 -0.187657 -0.107777   0.007550   
    
                               BMI    Smoker    Stroke  HeartDiseaseorAttack  \
    Diabetes_binary       0.293373  0.085999  0.125427              0.211523   
    HighBP                0.241019  0.087438  0.129060              0.210750   
    HighChol              0.131309  0.093398  0.099786              0.181187   
    CholCheck             0.045648 -0.004331  0.022529              0.043497   
    BMI                   1.000000  0.011551  0.022931              0.060355   
    Smoker                0.011551  1.000000  0.064658              0.124418   
    Stroke                0.022931  0.064658  1.000000              0.223394   
    HeartDiseaseorAttack  0.060355  0.124418  0.223394              1.000000   
    PhysActivity         -0.170936 -0.079823 -0.079985             -0.098223   
    Fruits               -0.084505 -0.074811 -0.008996             -0.019436   
    Veggies              -0.056528 -0.029926 -0.047601             -0.036315   
    HvyAlcoholConsump    -0.058232  0.077835 -0.023395             -0.037130   
    AnyHealthcare        -0.013417 -0.012939  0.006484              0.015687   
    NoDocbcCost           0.065832  0.035799  0.036198              0.036029   
    GenHlth               0.267888  0.152416  0.189447              0.275868   
    MentHlth              0.104682  0.091257  0.087303              0.075057   
    PhysHlth              0.161862  0.120698  0.164488              0.198416   
    DiffWalk              0.246094  0.119789  0.192266              0.232611   
    Sex                   0.000827  0.112125  0.003822              0.098161   
    Age                  -0.038648  0.105424  0.123879              0.221878   
    Education            -0.100233 -0.140966 -0.073926             -0.096559   
    Income               -0.124878 -0.104725 -0.136577             -0.146748   
    
                          PhysActivity    Fruits  ...  AnyHealthcare  NoDocbcCost  \
    Diabetes_binary          -0.158666 -0.054077  ...       0.023191     0.040977   
    HighBP                   -0.136102 -0.040852  ...       0.035764     0.026517   
    HighChol                 -0.090453 -0.047384  ...       0.031532     0.033199   
    CholCheck                -0.008249  0.017384  ...       0.106800    -0.062669   
    BMI                      -0.170936 -0.084505  ...      -0.013417     0.065832   
    Smoker                   -0.079823 -0.074811  ...      -0.012939     0.035799   
    Stroke                   -0.079985 -0.008996  ...       0.006484     0.036198   
    HeartDiseaseorAttack     -0.098223 -0.019436  ...       0.015687     0.036029   
    PhysActivity              1.000000  0.133813  ...       0.027089    -0.063302   
    Fruits                    0.133813  1.000000  ...       0.029385    -0.045843   
    Veggies                   0.149322  0.238605  ...       0.029152    -0.037146   
    HvyAlcoholConsump         0.019111 -0.033246  ...      -0.013484     0.009683   
    AnyHealthcare             0.027089  0.029385  ...       1.000000    -0.221658   
    NoDocbcCost              -0.063302 -0.045843  ...      -0.221658     1.000000   
    GenHlth                  -0.273548 -0.098687  ...      -0.033060     0.169515   
    MentHlth                 -0.130090 -0.062102  ...      -0.049850     0.193877   
    PhysHlth                 -0.234500 -0.048572  ...      -0.003285     0.157451   
    DiffWalk                 -0.276868 -0.050784  ...       0.008113     0.127111   
    Sex                       0.051753 -0.088723  ...      -0.006562    -0.048187   
    Age                      -0.100753  0.061096  ...       0.136975    -0.129839   
    Education                 0.190271  0.098715  ...       0.106601    -0.096989   
    Income                    0.196551  0.079009  ...       0.130492    -0.198171   
    
                           GenHlth  MentHlth  PhysHlth  DiffWalk       Sex  \
    Diabetes_binary       0.407612  0.087029  0.213081  0.272646  0.044413   
    HighBP                0.320540  0.064294  0.173922  0.234784  0.040819   
    HighChol              0.237778  0.083881  0.142610  0.162043  0.017324   
    CholCheck             0.059213 -0.010660  0.034540  0.044430 -0.007991   
    BMI                   0.267888  0.104682  0.161862  0.246094  0.000827   
    Smoker                0.152416  0.091257  0.120698  0.119789  0.112125   
    Stroke                0.189447  0.087303  0.164488  0.192266  0.003822   
    HeartDiseaseorAttack  0.275868  0.075057  0.198416  0.232611  0.098161   
    PhysActivity         -0.273548 -0.130090 -0.234500 -0.276868  0.051753   
    Fruits               -0.098687 -0.062102 -0.048572 -0.050784 -0.088723   
    Veggies              -0.115795 -0.052359 -0.066896 -0.084072 -0.052604   
    HvyAlcoholConsump    -0.058796  0.015626 -0.036257 -0.049294  0.014164   
    AnyHealthcare        -0.033060 -0.049850 -0.003285  0.008113 -0.006562   
    NoDocbcCost           0.169515  0.193877  0.157451  0.127111 -0.048187   
    GenHlth               1.000000  0.315077  0.552757  0.476639 -0.014555   
    MentHlth              0.315077  1.000000  0.380272  0.251489 -0.089204   
    PhysHlth              0.552757  0.380272  1.000000  0.487976 -0.045957   
    DiffWalk              0.476639  0.251489  0.487976  1.000000 -0.082248   
    Sex                  -0.014555 -0.089204 -0.045957 -0.082248  1.000000   
    Age                   0.155624 -0.101746  0.084852  0.195265 -0.002315   
    Education            -0.285420 -0.107005 -0.159317 -0.202590  0.043564   
    Income               -0.382969 -0.219070 -0.279326 -0.343245  0.159654   
    
                               Age  Education    Income  
    Diabetes_binary       0.278738  -0.170481 -0.224449  
    HighBP                0.338132  -0.141643 -0.187657  
    HighChol              0.240338  -0.084386 -0.107777  
    CholCheck             0.101743  -0.008695  0.007550  
    BMI                  -0.038648  -0.100233 -0.124878  
    Smoker                0.105424  -0.140966 -0.104725  
    Stroke                0.123879  -0.073926 -0.136577  
    HeartDiseaseorAttack  0.221878  -0.096559 -0.146748  
    PhysActivity         -0.100753   0.190271  0.196551  
    Fruits                0.061096   0.098715  0.079009  
    Veggies              -0.018893   0.152512  0.154899  
    HvyAlcoholConsump    -0.057705   0.036279  0.064095  
    AnyHealthcare         0.136975   0.106601  0.130492  
    NoDocbcCost          -0.129839  -0.096989 -0.198171  
    GenHlth               0.155624  -0.285420 -0.382969  
    MentHlth             -0.101746  -0.107005 -0.219070  
    PhysHlth              0.084852  -0.159317 -0.279326  
    DiffWalk              0.195265  -0.202590 -0.343245  
    Sex                  -0.002315   0.043564  0.159654  
    Age                   1.000000  -0.107127 -0.130140  
    Education            -0.107127   1.000000  0.460565  
    Income               -0.130140   0.460565  1.000000  
    
    [22 rows x 22 columns]



```python
fig, ax = plt.subplots(figsize=(30,10))
sns.heatmap(correlation,annot=True,cmap='YlOrRd')
#variables to drop: sex, AnyHealthcare, PhysHlth, DiffWalk, Fruits, Veggies, Education
```




    <Axes: >




    
![png](output_25_1.png)
    


# **Comparing models: Linear Regression vs KNN Model vs Logistic Regression**

### Removing highly correlated/poorly correlated indicators to improve model performance:



*   Multicollinearity occurs when two indicators are highly correlated. It reduces the statistical significance of our indicators and consquently makes our model less reliable.
*   In order to avoid **multicollinearity**, we need to remove indicators that are highly correlated. As seen above, the general health indicator is highly correlated with the mental health and physical health indicator. The indicators have a correlation of almost 0.5. We will also remove education as it is highly correlated with income.


Additionally, we will drop the following indicators that have a poor correlation with our dependent variable (correlation < 0.1):

'CholCheck','Stroke','Fruits','Sex','Veggies',
'AnyHealthcare', 'NoDocbcCost'and'DiffWalk'.

This also helps ensure that we are utilizing features that are redundant in our model.

## Linear Regression:
Using the **Ordinary Least Squares** model, we have created the following linear regression. Since, a majority of our indicators are categorical, we have used C() for the co-efficients of our categorical indicator.


```python
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm

db_2 =db.drop(['CholCheck','Stroke','Fruits','Sex','Veggies',
'AnyHealthcare','NoDocbcCost','MentHlth','PhysHlth','DiffWalk','Education'],axis=1)

['HighBP',
 'HighChol',
 'BMI',
 'Smoker',
 'HeartDiseaseorAttack',
 'PhysActivity',
 'HvyAlcoholConsump',
 'GenHlth',
 'Age',
 'Income']

lin_model = ols('Diabetes_binary~ C(HighChol) + C(HighBP) + BMI + C(Smoker) + C(HeartDiseaseorAttack) +  C(PhysActivity) + C(HvyAlcoholConsump) + C(GenHlth) + C(Age) + C(Income)', data=db_2).fit()

print(lin_model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        Diabetes_binary   R-squared:                       0.309
    Model:                            OLS   Adj. R-squared:                  0.309
    Method:                 Least Squares   F-statistic:                     1055.
    Date:                Thu, 23 May 2024   Prob (F-statistic):               0.00
    Time:                        19:41:41   Log-Likelihood:                -38229.
    No. Observations:               70692   AIC:                         7.652e+04
    Df Residuals:                   70661   BIC:                         7.680e+04
    Df Model:                          30                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------
    Intercept                         -0.3284      0.017    -19.476      0.000      -0.361      -0.295
    C(HighChol)[T.1.0]                 0.1057      0.003     30.985      0.000       0.099       0.112
    C(HighBP)[T.1.0]                   0.1572      0.004     42.809      0.000       0.150       0.164
    C(Smoker)[T.1.0]                   0.0004      0.003      0.131      0.896      -0.006       0.007
    C(HeartDiseaseorAttack)[T.1.0]     0.0681      0.005     14.350      0.000       0.059       0.077
    C(PhysActivity)[T.1.0]            -0.0092      0.004     -2.554      0.011      -0.016      -0.002
    C(HvyAlcoholConsump)[T.1.0]       -0.1272      0.008    -16.288      0.000      -0.142      -0.112
    C(GenHlth)[T.2.0]                  0.0883      0.006     16.023      0.000       0.077       0.099
    C(GenHlth)[T.3.0]                  0.2426      0.006     43.090      0.000       0.232       0.254
    C(GenHlth)[T.4.0]                  0.3317      0.006     51.421      0.000       0.319       0.344
    C(GenHlth)[T.5.0]                  0.3446      0.008     43.237      0.000       0.329       0.360
    C(Age)[T.2.0]                     -0.0107      0.017     -0.616      0.538      -0.045       0.023
    C(Age)[T.3.0]                      0.0087      0.016      0.536      0.592      -0.023       0.040
    C(Age)[T.4.0]                      0.0381      0.016      2.452      0.014       0.008       0.068
    C(Age)[T.5.0]                      0.0742      0.015      4.903      0.000       0.045       0.104
    C(Age)[T.6.0]                      0.1100      0.015      7.462      0.000       0.081       0.139
    C(Age)[T.7.0]                      0.1475      0.014     10.264      0.000       0.119       0.176
    C(Age)[T.8.0]                      0.1617      0.014     11.368      0.000       0.134       0.190
    C(Age)[T.9.0]                      0.2078      0.014     14.684      0.000       0.180       0.236
    C(Age)[T.10.0]                     0.2371      0.014     16.773      0.000       0.209       0.265
    C(Age)[T.11.0]                     0.2657      0.014     18.517      0.000       0.238       0.294
    C(Age)[T.12.0]                     0.2540      0.015     17.272      0.000       0.225       0.283
    C(Age)[T.13.0]                     0.2346      0.015     15.965      0.000       0.206       0.263
    C(Income)[T.2.0]                   0.0031      0.009      0.330      0.741      -0.015       0.021
    C(Income)[T.3.0]                  -0.0034      0.009     -0.382      0.703      -0.021       0.014
    C(Income)[T.4.0]                  -0.0090      0.009     -1.034      0.301      -0.026       0.008
    C(Income)[T.5.0]                  -0.0274      0.008     -3.248      0.001      -0.044      -0.011
    C(Income)[T.6.0]                  -0.0365      0.008     -4.446      0.000      -0.053      -0.020
    C(Income)[T.7.0]                  -0.0420      0.008     -5.162      0.000      -0.058      -0.026
    C(Income)[T.8.0]                  -0.0678      0.008     -8.667      0.000      -0.083      -0.052
    BMI                                0.0116      0.000     48.466      0.000       0.011       0.012
    ==============================================================================
    Omnibus:                     7005.048   Durbin-Watson:                   0.611
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2316.790
    Skew:                          -0.166   Prob(JB):                         0.00
    Kurtosis:                       2.178   Cond. No.                         958.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


As seen above, the model is not a great predictor of diabetes. The R-squared is low at 0.309. The model is also not very coherent in terms of interpretation. For example, a one unit increase in BMI leads to an increase in diabetes by 0.0116 units. However, our dependent variable is a binary variable so the correlation between the two indicators is not straightforward.  

### k-Nearest Neighbours




Moving on to the next model, we can utilize the K nearest neighbours algorithm to see whether this proves a better fit for diabetes prediction.
K Nearest Neighbours is an algorithm that uses closeness between points to predict the classification of an individual data point.

We will use a confusion matrix, accuracy score and classification report to evaluate the model


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

cols_drop = ['CholCheck','Stroke','Fruits','Sex','Veggies',
                   'AnyHealthcare','NoDocbcCost','MentHlth','PhysHlth','DiffWalk','Education']
db_test = db.drop(cols_drop, axis=1)
X_knn = db_test.drop(['Diabetes_binary'], axis=1)
y_knn = db_test['Diabetes_binary']


X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.3, random_state=100)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
```

    Accuracy: 0.643483591097699
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.60      0.81      0.69     10533
             1.0       0.72      0.47      0.57     10675
    
        accuracy                           0.64     21208
       macro avg       0.66      0.64      0.63     21208
    weighted avg       0.66      0.64      0.63     21208
    


How well did the KNN model do?

The model was split into two sets. 30% for testing and 70% training. The results below have been achieved after running this model


```python
from sklearn.neighbors import KNeighborsClassifier
knncfm = confusion_matrix(y_test,y_pred_knn)
labels = ['Positive','Negative']
sns.heatmap(knncfm/np.sum(knncfm), annot=True,
            fmt='.2%', cmap='rocket',xticklabels=labels, yticklabels=labels)
```




    <Axes: >




    
![png](output_35_1.png)
    


The KNN model has done moderately better than the linear regression model. Overall, the accuracy is around 64%. Our model does an average job in predicting the **true positives** i.e. predicting individuals who actually have diabetes. The true prediction rate is around 50%.

However, it does a poor job at predicting **true negatives**. The true negative prediction rate is only around 23.84%. This model may not be a good fit when predicting diabetes for individuals who do not have diabetes in reality.

## Logistic Regression

Given that our prediction variable is a binary indicator, logistic regression might be a better fit.



```python
from sklearn.linear_model import LogisticRegression

X=db.drop(['Diabetes_binary','CholCheck','Stroke','Fruits','Sex','Veggies',
'AnyHealthcare','NoDocbcCost','MentHlth','PhysHlth','DiffWalk','Education'],axis=1)
y=db['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=100)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
```

    Accuracy: 0.742078460958129
    
    Classification Report:
                   precision    recall  f1-score   support
    
             0.0       0.75      0.72      0.74     10533
             1.0       0.74      0.76      0.75     10675
    
        accuracy                           0.74     21208
       macro avg       0.74      0.74      0.74     21208
    weighted avg       0.74      0.74      0.74     21208
    


    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
logcfm=confusion_matrix(y_test, y_pred_log)
labels = ['Positive','Negative']
sns.heatmap(logcfm/np.sum(logcfm), annot=True,
            fmt='.2%', cmap='rocket',xticklabels=labels, yticklabels=labels)

```




    <Axes: >




    
![png](output_39_1.png)
    


Overall, the logistic regression model has a better true negative prediction rate (38.31%) than the KNN Model. Both models have similar accuracy and precision, but the logistic regression has a slightly better precision.



*   Precision for diabetes = 1.0 : KNN(0.72), Logistic Regression(0.74)
*   Precision for diabetes = 0.0 : KNN(0.60), Logistic Regression(0.75)




```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```


    
![png](output_41_0.png)
    


An AUC (Area Under the Curve) of 0.82 suggests that the model has a good ability to distinguish between the positive and negative diabetes cases.

### Model Analysis

**What are the leading indicators of diabetes?**

By ranking the absolute value of co-efficients we can compare variables by their prediction power. From the graph below, it appears that lifestyle choices play a substantial role in contributing to the onset of diabetes.

**HighBP**: is indicative of factors such as stress
**HeavyAlcoholconsumption and HighChol**: Poor lifestyle choices and lack of health discipline also contributes significantly to the incidence of diabetes.

Some factors may also be out of our control!

Characteristics such as age and income, are not neccessarily under individual control.  However, given that indicators such as high cholestrol and BP are determinants of diabetes risk, individuals can take early measures to prevent diabetes by doing regular checks.


```python
import matplotlib.pyplot as plt


feature_importance = abs(logreg.coef_[0])
sorted_indices = feature_importance.argsort()[::-1]
plt.bar(range(len(feature_importance)), feature_importance[sorted_indices], color='slateblue')
plt.xticks(range(len(feature_importance)), X.columns[sorted_indices], rotation=60)
plt.title('Feature Importance')
plt.ylabel('Absolute Coefficient')
plt.show()
```


    
![png](output_45_0.png)
    


**How do our models compare overall?**




*   The Logistic regression model fares better in terms of coherence. KNN has a better diabetes prediction rate, however, with KNN it is harder to interpret which features correlate with diabetes the best.
*   Hence, it is a tradeoff between the model with better prediction rate vs the model with the better explanatory power. Additionally, the logistic regression model also has a better true negative prediction rate. With disease prediction, it is as important to predict true negatives correctly as it is to predict true positives.
*   A model with a lower true negative rate might lead to more instances where individuals without diabetes are incorrectly predicted to have diabetes. This can result in significant costs, both financially and in terms of health implications


