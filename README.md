# NYC Airbnb Price Classification

## Project Overview
This project applies the machine learning life cycle to a binary classification problem using NYC Airbnb listings data, predicting whether a listing falls in the top 25% of prices ("high") or not ("low").

Trove Analytics helps hospitality clients understand and price short-term rental listings. Their client, a hospitality investment group, currently classifies listings into pricing tiers manually, which is a slow, subjective process that doesn't scale with a growing portfolio. A model that predicts whether a listing is high-priced would let the client automate this triage step, saving manual work and improving consistency.

**Note:** This is an educational project and should not be used as the sole basis for pricing, investment, or business decisions.

## Dataset
NYC Airbnb listings data with 28,022 examples and 51 original features. The label column is `price_category`, converted to binary: 1 if the listing's price is at or above the 75th percentile of all listings, 0 otherwise.

Class distribution: 74.2% low-priced, 25.8% high-priced.

## Methodology

**1. Exploratory Data Analysis**
Examined class balance, missing values, outliers, feature distributions, and feature-label relationships (e.g. `accommodates` and `room_type` against `price_category`).

**2. Data Preparation**
- Dropped free-text fields (`description`, `neighborhood_overview`, `host_about`, `amenities`), identifiers (`host_name`, `host_location`), `price` (label leakage), and `host_total_listings_count` (redundant with `host_listings_count`)
- Imputed `host_response_rate` and `host_acceptance_rate` with the median and added binary "missing" indicator columns, since non-response could itself be informative
- Imputed `bedrooms` and `beds` with the median (lower missingness, no indicator needed)
- Capped `minimum_nights`, `reviews_per_month`, and `host_listings_count` at their 99th percentile to reduce the influence of extreme outliers without dropping rows
- Converted boolean columns to 0/1 integers
- One-hot encoded `room_type` and `neighbourhood_group_cleansed`, dropping the first category of each to avoid multicollinearity
- Did not scale features for the Decision Tree, since tree splits are threshold-based rather than distance-based

**3. Decision Tree**
Trained a baseline `DecisionTreeClassifier`, then tuned via `GridSearchCV` (5-fold, scored on F1) over `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion`.

Best parameters: `criterion='gini'`, `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=5`

**4. Neural Network**
Built with TensorFlow/Keras:
- Input layer (48 features)
- Hidden layer, 64 units, ReLU
- Hidden layer, 32 units, ReLU
- Output layer, 1 unit, sigmoid

Trained with SGD (learning rate 0.1), binary cross-entropy loss, 100 epochs, 20% validation split.

## Results

| Model          | Accuracy | F1 Score |
|----------------|----------|----------|
| Decision Tree  | 0.8264   | 0.6296   |
| Neural Network | 0.8180   | 0.6473   |

The Decision Tree scored higher on accuracy; the neural network scored higher on F1, suggesting it was somewhat better at correctly identifying high-priced listings specifically, even with a lower overall accuracy.

**Recommendation:** Decision Tree. Comparable performance, no training-time cost (versus ~2 minutes for the neural network), and direct interpretability through feature importances — which lets the client understand *why* a listing was flagged, not just receive a prediction.

## Key Findings
- `accommodates` was the single strongest predictor (importance ≈ 0.29), followed by `bathrooms`, `room_type_Private room`, and `neighbourhood_group_cleansed_Manhattan`.
- Listing characteristics (capacity, room type) mattered more to the model than host reputation or review scores.
- The neural network's validation loss began climbing almost immediately after ~epoch 10 while training loss kept falling — a clear overfitting signal, consistent with the smaller test-set F1 gain relative to its training performance.

## Responsible Use and Limitations
This dataset reflects patterns in the NYC short-term rental market, which is itself shaped by existing housing and economic inequities. `neighbourhood_group_cleansed` (borough) ranked as an important feature, and borough is a plausible proxy for socioeconomic status. Review-based features may also encode bias, since guest ratings of hosts can vary by race, ethnicity, and gender. The dataset is further limited to listings hosts chose to put on Airbnb, so it doesn't represent all NYC housing equally.

Incorrect predictions carry real risk: if a listing in a lower-income neighborhood is mispredicted as low-priced, it could receive less marketing investment, reinforcing existing inequality. If listings in gentrifying areas are systematically flagged high-priced, the model could indirectly contribute to displacement pressure. This model should be used as a triage aid alongside human review, not as an automated pricing or investment decision-maker.

## Repository Structure

```text
nyc-airbnb-price-classification/
├── data/
│   └── airbnbListingsData.csv
├── Capstone.ipynb
├── LICENSE
└── README.md
```

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, Jupyter Notebook

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/simsima23/nyc-airbnb-price-classification.git
cd nyc-airbnb-price-classification
```

**2. Create and activate a virtual environment**

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install required packages**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

**4. Launch Jupyter Notebook**
```bash
jupyter notebook
```
Open `Capstone.ipynb`.

**5. Run the project**
In Jupyter, select **Kernel > Restart & Run All** to reproduce the full preprocessing, training, evaluation, and visualizations.

## Individual Contributions
This was an individual capstone project completed as part of the Break Through Tech Machine Learning Foundations program.

My contributions included defining the ML problem and business framing, performing EDA, preparing and engineering features, training and tuning the Decision Tree, building and training the neural network, evaluating and comparing both models, and documenting ethical considerations, limitations, and next steps.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
