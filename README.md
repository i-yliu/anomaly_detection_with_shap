# anomaly_detection_with_shap

Detecting anomalous structures in images is very critical among numerous tasks in the field of computer vision. For example, in Fig 1 (b), a deep learning model can successfully classify the screw, but it will not be able to identify the defect region. It can be very dangerous in many situations. In fact, anomalies are typically the most informative regions in an image such as defects in images used for quality control. Therefore, detection problems have been widely investigated in image processing and computer vision, and are very important in many applications like quality inspection, health monitoring etc.

Existing methods include two major approaches, supervised learning, and unsupervised learning. In supervised learning methods, it is assumed that anomalies manifest themselves in the form of images of an entirely different class which can be reduced to a simple classification problem. There is only little work done to segment the anomalous region directly. Another way to approach this problem is to perform semantic segmentation, where anomaly regions are labeled as pixel-level classes. However, this requires lots of labeling and is not robust to unseen artifacts.  In unsupervised learning methods, generative models such as Generative Adversarial Networks or variational autoencoders can detect anomalies by reconstructing the input image, and the anomaly area will have high reconstruction errors. However, this method has been shown problematic because the reconstructions can be very inaccurate and thus fail to detect the region. 

In this work, we propose a novel approach based on Shapley value to detect anomaly areas in images. Shapley value is a concept from cooperative game theory, which provides a way to fairly divide the payoff between the players in a coalition. In another way, the Shapley value measures each player's average marginal contribution to each coalition where players can form different coalitions and have varying degrees of influence.  In the context of anomaly detection, we segment each image into different regions by a k-means based segmentation method and each region acts as a player (As shown in Fig 1 (c)), noting that it would be computationally expensive if we consider each pixel as a player. The high-level intuition of our method is that the anomaly region will contribute the most to the reconstruction loss to which the model has not seen the region during the training session. 

Shapley value was used in several previous approaches for interpreting general machine learning outputs [1-6]. In fact, Shapley value was already used for anomaly scores interpretation in different contexts [7-10], where they applied Shapley value to explain the anomalies detected by a model. However, none of the previous work has directly used Shapley value for anomaly detection. Also, it is not to our knowledge that Shapley value has been used for Image anomaly regions detection. 

In this work, we aim to use reconstruction loss as a characteristic function anomaly score, upon which the Shapley value is computed from equation (1). We plan to adopt an monte carlo sampling based method [11] to estimate Shapley value for each region by masking off segmented regions. The idea of the proposed method is approximating the absence of some features and the anomaly region should contribute most to the reconstruction loss.

References:
Explaining deep neural networks with a polynomial time algorithm for Shapley values approximation (2019)
https://arxiv.org/pdf/1903.10992.pdf 

Entropy criterion in logistic regression and Shapley value of predictors (2006) https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=1263&context=jmasm 

A unified approach to interpreting model predictions (2017) https://arxiv.org/pdf/1705.07874.pdf 

On Shapley value for measuring importance of dependent inputs (2017) https://arxiv.org/pdf/1610.02080.pdf 

Explaining prediction models and individual predictions with feature contributions (2014)
https://search-proquest-com.udel.idm.oclc.org/docview/1621073940?pq-origsite=360link

Axiomatic attribution for deep network (2017)
https://arxiv.org/pdf/1703.01365.pdf

Explaining anomalies detected by autoencoders using SHAP (2019)
https://arxiv.org/pdf/1903.02407.pdf 

Additive explanations for anomalies detected from multivariate temporal data (2019)
https://dl.acm.org/doi/abs/10.1145/3357384.3358121 

Shapley values of reconstruction errors of PCA for explaining anomaly detection (2019)
https://arxiv.org/pdf/1909.03495.pdf 

On anomaly interpretation via shapley values (2020)
https://arxiv.org/pdf/2004.04464.pdf 

Efficient Computation of the Shapley Valuefor Game-Theoretic Network Centrality (2013) https://arxiv.org/pdf/1402.0567.pdf


