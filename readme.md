# My Hello World to Kaggle
House Prices: Advanced Regression Techniques

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# Approach
### 1st attempt
  - linear regression merely on numeric feaures

rank: 2757, score: 0.15143

🤔 it is surprising not bad cos i remove all of the non-numeric features

### 2nd attempt
  - feature-encode categorical(non-numeric) feaures
  - remove outliers
  - linear regression merely on all feaures

🚧 working in progress

# Run it

install

```
python3 -m venv .env
source .env/bin/activate
pip install -r packages.txt
```

run

```
source .env/bin/activate
python3 main
```

exit

`deactivate`
