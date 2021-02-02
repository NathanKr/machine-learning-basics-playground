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
  <tr>
    <td>neuron_logic_and_gate.py</td>
    <td>
    <ul>
    <li>model a neuron by logistic regression</li>
    <li>use to create logic AND using a neuron</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>nn_learn_analytic_back_propagation.py</td>
    <td>
    <ul>
    <li>solve the data set with the nural network as in https://www.youtube.com/watch?v=IN2XmBhILt4</li>
    <li>this is solved using gredient descent</li>
    <li>derivative of cost with respect to the features is computed using the chain rule ad gradient descent in a nice lazy programming manner</li>
    <li>using the chain rule evaluate the derivative from right (cost function) to left thus back propagating</li>
    <li>the derivatives are checked via the numeric derivative in debug_check_analytical_derivative()</li>
    <li>taking initial values suggested by StatsQuest and AndrewNg the algorithm is converging but not to the global minima - ssr ~0.7 so result are not good but the learning code is working</li>
    <li>taking initial values around the solution of minimize - check neural_network_learn_minimize is working even when we add eps of 4</li>
    <li>TODO : understand how to get initial values that will cause covergence to the global minima ssr ~ 0</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>nn_learn_minimize.py</td>
    <td>
    <ul>
    <li>solve the data set with the nural network as in https://www.youtube.com/watch?v=IN2XmBhILt4</li>
    <li>this is solved using minimize thus no derivative are used here</li>
    <li>the algorithm is converging in general to almost zero cost function</li>
    </ul>
    </td>
  </tr>
</table>
