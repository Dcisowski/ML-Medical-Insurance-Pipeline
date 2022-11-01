# ML-Medical-Insurance-Pipeline

This project contains simple regression model pipeline responsible for predicting
insurance cost for given patient.

To train a model on **train.csv**, and run all the tests
use the following command:

`tox`

if you only want to train a model use:

`tox -e train`

**Remember that data needs to be split for train.csv and test.csv files respectively**.

Future plans:

1. Adding api endpoints for REST api communication (**Fast API or Flask**).
2. Adding CI/CD.
3. Containerising the app.
4. adding Differential testing.