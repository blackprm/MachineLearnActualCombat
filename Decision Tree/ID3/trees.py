from math import log
import treeplot as tr
import operator

def calcShannonEnt(dataSet):            #计算集合的香农熵
    numEntries = len(dataSet)           #数据集的个数
    labelCounts = {}                    #将特征计数
    for featVec in dataSet:             #循环将特征数载入字典
        currentLabel = featVec[-1]      #假设特征不在字典的key中，则创建并初始化为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:             #计算香农熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt 
def createDateSet():
    dateSet = [[1, 1, 'no'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [1, 1, 'yes']]

    labels = ['no surfacing','flippers']
    return dateSet,labels


def splitDataSet(dataSet, axis, value):#待划分的数据集、 划分数据集的特征的索引、 需要返回的特征的值  本函数的作用是找出符合value的值的个体，并将其去除对应位置的特征值后单独拿出
    retDataSet = []
    for featVec in dataSet:             #遍历训练集
        if featVec[axis] == value:      #符合对应条件
            reducedFeatVec = featVec[:axis]         #对特征个体进行分片
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)       #加入到新的集合中     
    return retDataSet




def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1                #减去一个特征标签
    baseEntropy = calcShannonEnt(dataSet)           #计算初始的香农熵
    bestInfoGain = 0.0                              #记录新熵与旧熵的差值
    bestFeature = -1                                #记录是香农熵变化最大的特征的索引
    for i in range(numFeatures):                    #对所有特征进行判断
        featList = [example[i] for example in dataSet]      #把相应位置的特征可能的离散值 递推构造到列表中
        uniqueVals = set(featList)                          #去除特征中的重复值
        newEntropy = 0.0                                    #存储新的香农熵
        for value in uniqueVals:                            #将各个特征可能取值去除后计算香农熵的和
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #计算出前后的差值
        if (infoGain > bestInfoGain):       #如果差值变大
            bestInfoGain = infoGain         #更新差值
            bestFeature = i                 #存储索引
    return bestFeature                      #返回使系统混乱度减少最多的特征的索引

def majorityCnt(classList):         #假设所有标签都判断完，则取最多的选择
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  #取出所有训练个体中最后一个特征

    if classList.count(classList[0]) == len(classList): 
        return classList[0]             #当所有特征相同时停止分类
   
    if len(dataSet[0]) == 1:                #当特征只有一个时
        return majorityCnt(classList)       #返会出现次数多的标签名
    bestFeat = chooseBestFeatureToSplit(dataSet) #找出使系统的熵减至最底的特征的索引


    
    bestFeatLabel = labels[bestFeat]            #取得特征名字

    myTree = {bestFeatLabel:{}}                 #创建根节点
    del(labels[bestFeat])                       #删除已经插入树中的元素的特征
    featValues = [example[bestFeat] for example in dataSet]     #找到所有特征的离散值
    uniqueVals = set(featValues)                                #去除重复

    for value in uniqueVals:
        subLabels = labels[:]                                 #复制标签列表        
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)     #递归生成树的节点
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]            #找出决策树的第一个节点
    secondDict = inputTree[firstStr]                #决策树的分支
    featIndex = featLabels.index(firstStr)          #找出第一个判断的特征的索引
    key = testVec[featIndex]                        #提取出判断对象中的特征
    valueOfFeat = secondDict[key]                   #找出该特征下的子树
    if isinstance(valueOfFeat, dict):               #判断子树是否为字典
        classLabel = classify(valueOfFeat, featLabels, testVec)#如果是就继续寻找
    else:
        classLabel = valueOfFeat        #如果不是就返回内容
    return classLabel
def storeTree(inputTree,filename): #序列化文件，存储到本地磁盘
    import pickle
    fw = open(filename,'wb+')
    pickle.dump((inputTree),fw)
    fw.close()
def grabTree(filename): # 反序列化文件，加载到内存
    import pickle
    fr = open(filename,'rb') 
    return pickle.load(fr)


file = open('mg.txt')
fileList = []
for line in file.readlines():
    fileList.append(line.strip().split('\t'))
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
myTree = createTree(fileList,lensesLabels)
tr.createPlot(myTree)
    
