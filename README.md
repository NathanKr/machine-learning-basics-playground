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
    <li>function to compute sigmoid function</li>
    <li>generic and array-based function to compute cost function for logistic regression</li>
    <li>generic and array-based function to compute cost function for linear regression</li>
    <li>generic and array-based function to compute gradient descent for linear regression</li>
    <li>generic and array-based function to compute gradient descent for logistic regression</li>
    <li>generic and array-based function to compute high order polynomial</li>
    <li>generic and array-based function to compute mean normalization</li>
    <li>generic and array-based function to compute a normal distribution</li>
    <li>generic and array-based function to compute F1score</li>
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
    <td>linear-regression-gradient-descent-profit-for-population.py</td>
    <td>gradient descent to solve linear regression. this is from Andrew Ng machine learning course @ Coursera</td>
  </tr>
  <tr>
    <td>logistics-regression-minimize-success-for-grades-1feature.py</td>
    <td>use optimize.minimize to solve logistics regression</td>
  </tr>
  <tr>
    <td>logistics-regression-minimize-success-for-grades-2features.py</td>
    <td>
    <ul>
    <li>use optimize.minimize to solve logistics regression. the dataset is from Andrew Ng machine learning course @ Coursera</li>
    <li>feature scaling is a must</li>
    <li>plot the resulted decision boundary</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>logistics-regression-gradient-descent-success-for-grades-2features.py</td>
    <td>
    <ul>
    <li>gradient descent to solve logistics regression. the dataset is from Andrew Ng machine learning course @ Coursera</li>
    <li>feature scaling is a must</li>
    </ul>
    </td>
  </tr>
  <tr>
    <td>regularization.py</td>
    <td>
    <ul>
    <li>use optimize.minimize to solve linear regression. the dataset is from Andrew Ng machine learning course @ Coursera</li>
    <li>use 8 order polynomial which overfits with lamda = 0 but is better with lamda @ minimum cross-validation cost</li>
    <li>plot Jcv for different lambda and choose the lambda with minimum cost</li>
    <li>feature scaling is a must</li>
    </ul>
    </td>
  </tr>
<tr>
    <td>learning_curves.py</td>
    <td>
    <ul>
    <li>use learning curves to show high bias for 1 order polynomial</li>
    <li>use learning curves to show high variance for 8 order polynomial</li>
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
    <li>there are two anomalies in the middle-high p so it is not possible to detect it and it is clear that more features are needed</li>
    <li>it is important to get a feel of the problem by making plots</li>
    <li>best eps is computed by looping over few value of eps and computing F1Score. Here we look for the maximal F1Score</li>
    <li>this is from Andrew Ng machine learning course @ Coursera</li>
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
    <li>solve the data set with the neural network as  <a href="https://www.youtube.com/watch?v=IN2XmBhILt4">here</a></li>
    <li>a neural network is solved using gradient descent</li>
    <li>the derivative of cost with respect to the features is computed using the chain rule ad gradient descent in a nice lazy programming manner</li>
    <li>using the chain rule evaluates the derivative from the right (cost function) to left thus back-propagating</li>
    <li>the derivatives are checked via the numeric derivative in debug_check_analytical_derivative()</li>
    <li>taking initial values suggested by StatsQuest and AndrewNg the algorithm is converging but not to the global minima - ssr ~0.7 so result are not good but the learning code is working</li>
    <li>The same neural network is solved using optimise.minimize in nn_learn_minimize.py</li>
    <li>taking initial values around the solution of minimize - check neural_network_learn_minimize is working even when we add eps of 4</li>
    <li>TODO : understand how to get initial values that will cause convergence to the global minima ssr ~ 0</li>
    <li>the cost function used here is ssr</li>
    <li>activation function used here is softplus</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>nn_learn_minimize.py</td>
    <td>
    <ul>
    <li>solve the data set with the neural network as  <a href="https://www.youtube.com/watch?v=IN2XmBhILt4">here</a></li>
    <li>this is solved using minimize thus no derivative are used here</li>
    <li>the algorithm is converging in general to almost zero cost function</li>
    <li>the cost function used here is ssr</li>
    <li>activation function used here is softplus</li>
    </ul>
    </td>
  </tr>
   <tr>
    <td>nn_learn_back_propagation_engine.py</td>
    <td>
    <ul>
    <li>this is a class that represent a neural network and is based on <a href="https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py">this code</a> which is based on <a href="http://neuralnetworksanddeeplearning.com/chap2.html">this book</a></li>
    <li>the Network class is simple but generic - you can define the number of layers and number of neurons per layer</li>
    <li>back-propagation algorithm is used here in the function backprop to compute the derivative of the cost function (nabla) of a sample point (x,y) with respect to the weights and biases and then create the mean over the mini-batch. the algorithm is based on 4 equations BP1,BP2,BP3,BP4 which are defined in the book link</li>
    <li>the function backprop compute also the new weights and biases given the nabla and the learning rate alfa thus doing gradient descent step </li>
    <li>I have made few changes: 1. activation function and its derivative are parameters of the constructor 2. replace xrange with range because there is no xrange in current python version 3. for simplicity I am not using SGD , instead I am using gradient descent in a new function called train</li>
    <li>The cost function Cx is the cost function per sample and is define implicitly as (1/2)*(output_activations-y)^2 where output_activations is a^L which is h (hypothesis). It is not used directly, only its derivative is used as cost_derivative function. it is possible very easily to extend Network class such that the cost derivative will be a constructor argument</li>
    </ul>
    </td>
  </tr>
 <tr>
    <td>nn_learn_back_propagation.py</td>
    <td>
    <ul>
    <li>this file uses nn_learn_back_propagation_engine.py via learn_logical_and() to solve a logical && using a neural network with two layers</li>
    <li>learn_logical_and() is working only if I start around the solution of the features, maybe because I have only 4 data set samples, BTW I had the same problem with nn_learn_analytic_back_propagation.py which have 3 data set samples</li>
    <li>The same neural network is solved using optimise.minimize in neuron_logic_and_gate.py</li>
    <li>I have implemented learn_StatsQuest() but it will NOT match StatsQuest as nn_learn_minimize.py and nn_learn_analytic_back_propagation.py because StatasQuest neuron network has activation function on layer 3 which is linear, and activation function on layer 2 is softplus . However , currently Network class has the same activation function for every neuron. It is not difficult to support activation per nuron but currently it is not supported</li>
    </ul>
    </td>
  </tr>
  
  
</table>
