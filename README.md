# A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin
A Method for Guaranteeing Wireless Communication  Based on a Combination of Deep and Shallow Learning

![image](https://github.com/FlamingJay/https://github.com/FlamingJay/A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin/blob/master/figure/framework.jpg)

The work in this paper can be taken into two parts.

The first part is to use the autoencoder to extract features and compress the dimension to a lower value.
![image](https://github.com/FlamingJay/A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin/blob/master/figure/deep%20autoencodereps.png)

Then, the output of autnencoder is as the input of the SVM. But there are some parameters in the SVM. 
So we want to use the Artificial Bee Colony Algorithm to find the optimal parameters.
![image](https://github.com/FlamingJay/A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin/blob/master/figure/Optimazationeps.png)

Finally, we compare our work with some existed work.

The PCA method is adopted as the baseline for featrue extraction. So some experiments are first conducted based on PCA.
![image](https://github.com/FlamingJay/A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin/blob/master/figure/%E8%B4%A1%E7%8C%AE%E7%8E%87%E6%9B%B2%E7%BA%BF%E5%92%8C%E4%B8%8D%E5%90%8CC%E4%B8%8B%E7%9A%84%E6%A3%80%E6%B5%8B%E7%8E%87%20(2).png)

Based on above, the corresponding of reduced dimension is selected and the experiment result on the autoencoder is shown below. 
![image](https://github.com/FlamingJay/A-Method-for-Guaranteeing-Wireless-Communication-Based-on-a-Combination-of-Deep-and-Shallow-Learnin/blob/master/figure/%E5%87%86%E7%A1%AE%E7%8E%87%E5%92%8C%E8%99%9A%E8%AD%A6%E7%8E%87%E5%AF%B9%E6%AF%94.png)
