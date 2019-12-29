# ComboLoss for Facial Beauty Prediction

## Data Description
| Dataset | Median | Mean |
| :---: | :---: | :---: |
| SCUT-FBP | 2.549 | 2.694 |
| HotOrNot | 0.0369 | 0.0039 |
| SCUT-FBP5500 | 3 | 2.99 |


## Performance Evaluation
### Evaluation on SCUT-FBP5500 (6/4 splitting strategy)
| Backbone | Loss | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: | :---: |
| SEResNeXt50 | L1 | 0.2212 | 0.2941 | 0.9012 |
| SEResNeXt50 | MSE | 0.2195 | 0.2947 | 0.9008 |
| ComboNet (SEResNeXt50) | CombinedLoss (alpha=1, beta=1) | 0.2135 | 0.2818 | 0.9099 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=2, beta=1) | 0.2191 | 0.2891 | 0.9066 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=2, beta=1) | 0.2124 | 0.2803 | 0.9108 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=3, beta=1) | 0.2190 | 0.2894 | 0.9053 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=1, beta=2) | 0.2150 | 0.2868 | 0.9063 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=1, beta=2) | 0.2176 | 0.2895 | 0.9044 |
| ComboNet (SEResNeXt50)  | CombinedLoss (alpha=1, beta=3) | 0.2171 | 0.2862 | 0.9071 |

### Evaluation on SCUT-FBP
| Backbone | CV | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: | :---: |
| SEResNeXt50 | 1 | 0.2689 | 0.3340 | 0.9144 |
| SEResNeXt50 | 2 | 0.2456 | 0.3050 | 0.9063 |
| SEResNeXt50 | 3 | 0.2242 | 0.3000 | 0.8880 |
| SEResNeXt50 | 4 | 0.2282 | 0.2992 | 0.9238 |
| SEResNeXt50 | 5 | 0.2171 | 0.2889 | 0.9051 |
| SEResNeXt50 | AVG | 0.2368 | 0.3054 | 0.9075 |

### Evaluation on HotOrNot
| Backbone | CV | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: | :---: |
| ComboNet (SEResNeXt50) | 1 | 0.8450 | 1.0689 | 0.4973 |
| ComboNet (SEResNeXt50) | 2 | 0.8201 | 1.0490 | 0.5059 |
| ComboNet (SEResNeXt50) | 3 | 0.8124 | 1.0399 | 0.5027 |
| ComboNet (SEResNeXt50) | 4 | 0.8111 | 1.0216 | 0.4965 |
| ComboNet (SEResNeXt50) | 5 | 0.8110 | 1.0409 | 0.4888 |
| ComboNet (SEResNeXt50) | AVG | 0.8119 | 1.0441 | 0.4982 |


### Evaluation on SCUT-FBP5500 (5-Fold Cross Validation)
| Backbone | CV | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: | :---: |
| ComboNet (SEResNeXt50)  | 1 | 0.2124 | 0.2780 | 0.9137 |
| ComboNet (SEResNeXt50)  | 2 | 0.2121 | 0.2842 | 0.9105 |
| ComboNet (SEResNeXt50)  | 3 | 0.2056 | 0.2745 | 0.9194 |
| ComboNet (SEResNeXt50)  | 4 | 0.2037 | 0.2708 | 0.9199 |
| ComboNet (SEResNeXt50)  | 5 | 0.2011 | 0.2633 | 0.9237 |
| ComboNet (SEResNeXt50)  | AVG | 0.2170 | 0.2742 | 0.9177 |


## Reference
1. Xu L, Xiang J, Yuan X. CRNet: Classification and Regression Neural Network for Facial Beauty Prediction[C]//Pacific Rim Conference on Multimedia. Springer, Cham, 2018: 661-671.
2. Lin L, Liang L, Jin L, et al. [Attribute-aware convolutional neural networks for facial beauty prediction](https://www.ijcai.org/proceedings/2019/0119.pdf)[C]//Proceedings of the 28th International Joint Conference on Artificial Intelligence. AAAI Press, 2019: 847-853.
3. Xu L, Fan H, Xiang J. Hierarchical Multi-Task Network For Race, Gender and Facial Attractiveness Recognition[C]//2019 IEEE International Conference on Image Processing (ICIP). IEEE, 2019: 3861-3865.
4. Liu X, Li T, Peng H, et al. [Understanding beauty via deep facial features](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AMFG/Liu_Understanding_Beauty_via_Deep_Facial_Features_CVPRW_2019_paper.pdf)[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2019: 0-0.