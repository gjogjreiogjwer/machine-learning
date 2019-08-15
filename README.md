# machineLearning
现对之前学习的机器学习算法进行重新编写和整理，改为python3语言并对格式规范进行处理，持续更新中

监督学习：  
  分类：有限取值  
        (1)kNN  
        (2)trees  
        (3)bayes  
        (4)logistic  
        (5)svm  
        (6)AdaBoost  
  回归：连续数值型   
        (1)regression  
        (2)regtrees
        
无监督学习：  
  (1)k均值聚类
  (2)Apriori发现频繁集
  (3)FP树


L1和L2正则化：缓和过拟合  
https://blog.csdn.net/red_stone1/article/details/80755144  

l2正则化，它对于最后的特征权重的影响是，尽量打散权重到每个特征维度上，不让权重集中在某些维度上，出现权重特别高的特征。    
l1正则化，它对于最后的特征权重的影响是，让特征获得的权重稀疏化，也就是对结果影响不那么大的特征，干脆就拿不着权重。    


解决过拟合：    
  1.增大样本量    
  2.减少特征的量 
  3.增强正则化作用（改变C参数，在交叉验证集上做grid-search查找最好的正则化系数）  

解决欠拟合：  
  1.更有效的特征  
  2.更复杂的模型（e.g.非线性核函数）  


