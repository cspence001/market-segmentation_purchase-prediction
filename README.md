# market-segmentation_purchase-prediction

Customer analysis and market segmentation based on site user demographics (Age, Gender, Salary, Purchased/Not Purchased) to identify target markets and optimize model to predict purchase behavior following ad click. 
<br><br>
<b>primary analysis:</b><br>
<ul>
<li><b>purchase models</b> used to identify market clusters based on cluster and sub-cluster feature variants: Gender, Age, Salary, Purchased/Not Purchased<br> </li>
<li><b>K-means</b> used to identify and label target range distributions of market clusters<br> </li> (sub-process used to estimate distribution tiers for binary target weighted features) 
</ul>
<b>prediction models:</b><br><br>
<ul>
<li><b>Logistic Regression using Standard Scalar, TTS</b> 
<li><b>Logistic Regression using Standard Scalar, TTS WoE Encoded Variables</b> </li>
<li><b>Decision Tree, TTS</b> 
<br><br>
<b>Data Tier-level exclusive 'K-wise' Optimization Models:</b>
<br>*models used to determine efficacy of tier-market labeling as sub-divided within k-clusters, i.e. if distribution among sub-divided salary and age categorical tier thresholds, contributes effectively to model accuracy for classification of binary dependent variable.<br><br>
<li><b>Logistic Regression using one-hot and K-folds</b></li> 
<li><b>Logistic regression using one-hot and TTS</b> 
<li><b>Decision Tree using one-hot and TTS</b> 
<li><b>Logistic Regression using WoE, IV and K-folds </b> 
<li><b>classification report, confusion matrix</b> used to determine accuracy of each model at predicting Purchased/Not Purchased <br> </li>
</ul>

<a href="https://github.com/cspence001/market-segmentation_purchase-prediction
/blob/main/notebooks/purchase_models.ipynb">purchase analysis</a>
<br>
<a href="https://github.com/cspence001/market-segmentation_purchase-prediction
/blob/main/notebooks/log-regression_k-folds.ipynb">prediction models</a>
<br>
<a href="https://github.com/cspence001/market-segmentation_purchase-prediction/blob/main/Resources/Models_Results.docx">custom tier training confluence map</a>
<br>
<h5>jupyter notebook running pandas dataframes using matplotlib, seaborn, sklearn</h5>

