# _*_coding:utf-8_*_
"""
Author: Lingren Kong
Created Time: 2020/4/24 14:16
"""


# 打出常用的array属性
def npCheck(arr,display=False):
    import numpy as np
    print(f'当前对象的id为：{id(arr)}')#numpy有些是原位操作，要注意这点所以要检查id
    if isinstance(arr,np.matrix):
        print('这是一个np的matrix')
        print(f'矩阵是{arr.shape[0]}*{arr.shape[1]}的')
    elif isinstance(arr,np.ndarray):# 因为matrix也是array的实例
        print('这是一个np的ndarray')
        print(f'维数（轴的数目）是{arr.ndim}')
        print(f'各维度的长度是{arr.shape}')
        print(f'元素的数据类型是{arr.dtype}')
    else:
        print('这似乎不是numpy的对象吖')
    if display:
        print(arr)
    print('-----------')

def modelCheck(y,pred):
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    print("错误率error：",np.mean(y==pred))
    print("查准率precision：",precision_score(y,pred))
    print("查全率recall：",recall_score(y,pred))
    print("F1 score：",f1_score(y,pred))
    print(confusion_matrix(y,pred))
    return

