<h2>Motivation</h2>
python code to do basic machine learning stuff

<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>utils.py</td>
    <td>
    <ul>
    <li>function to compute sigmond function</li>
    <li>generic and array based function to compute cost function for logistic regression</li>
    <li>generic and array based function to compute cost function for linear regression</li>
    <li>generic and array based function to compute gradient descent for linear regression</li>
    <li>generic and array based function to compute gradient descent for logistic regression</li>
    <li>generic and array based function to compute high order polynomial</li>
    <li>generic and array based function to compute mean normalization</li>
    <li>generic and array based function to compute normal distribution</li>
    <li>generic and array based function to compute F1score</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>linear-regression-exact.py</td>
    <td>
    <ul>
    <li>fit h_teta=teta0+teta1*x to a data set</li>
    <li>no need for iterations</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>cost-function-teta1.py</td>
    <td>
    <ul>
    <li>fit h_teta=teta1*x to a data set</li>
    <li>plot cost function J for different teta1</li>
    <li>get the teta1 for min(J) plot</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>linear-regression-gradient-descent-teta1.py</td>
    <td>same as cost-function-teta1.py but now use gradient descent to solve this</td>
  </tr>
  <tr>
    <td>linear-regresion-gradient-descent-profit-for-population.py</td>
    <td>gradient descent to solve linear regression. this is from Andrew Ng machine learning course @ coursera</td>
  </tr>
  <tr>
    <td>logistics-regression-minimize-success-for-grades-1feature.py</td>
    <td>use optimize.minimize to solve logistics regression</td>
  </tr>
  <tr>
    <td>logistics-regression-minimize-success-for-grades-2features.py</td>
    <td>
    <ul>
    <li>use optimize.minimize to solve logistics regression. the dataset is from Andrew Ng machine learning course @ coursera</li>
    <li>feature scaling is a must</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>logistics-regression-gradient-descent-success-for-grades-2features.py</td>
    <td>
    <ul>
    <li>gradient descent to solve logistics regression. the dataset is from Andrew Ng machine learning course @ coursera</li>
    <li>feature scaling is a must</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>regularization.py</td>
    <td>
    <ul>
    <li>use optimize.minimize to solve linear regression. the dataset is from Andrew Ng machine learning course @ coursera</li>
    <li>use 8 order polynomial which overfit with lamda = 0 but is better with lamda @ minimum cross validation cost</li>
    <li>plot Jcv for different lambda and choose the lambda with minimum cost</li>
    <li>feature scaling is a must</li>
    </ul>
    </td>
  </tr>
<tr>
    <td>learning_curves.py</td>
    <td>
    <ul>
    <li>use learning curves to show high bias for 1 order ploynomial</li>
    <li>use learning curves to show high variance for 8 order ploynomial</li>
    <li>use 1 and 8 order polynomial</li>
    <li>feature scaling is a must</li>
    <li>use linear regression and solve with optimize.minimize</li>
    </ul>
    </td>
  </tr>
<tr>
    <td>anomaly_detection.py</td>
    <td>
    <ul>
    <li>detect anomaly using two features</li>
    <li>there is no test set here only train and cv so we can not test the results properly</li>
    <li>there are two anomlies in the middle - high p so it is not possible to detect it and it is clear that more features are needed</li>
    <li>it is important to get a feel of the problem by making plots</li>
    <li>best eps is computed by looping over few value of eps and computing F1Score. Here we look for the maximal F1Score</li>
    <li>this is from Andrew Ng machine learning course @ coursera</li>
    </ul>
    </td>
  </tr>
</table>
