from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
def file2matrix(filename):
    fr = open(filename)                 #打开数据文件
    arrayOlines=fr.readlines()          #按行读取数据存储为一个一维list
    numberOfLines = len(arrayOlines)    #求出list的长度即文件的行数
    returnMat = zeros((numberOfLines,3)) #❷ 创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
        #❸ （以下三行） 解析文件数据到列表
    for line in arrayOlines:
        line = line.strip() #去掉前后空格
        listFromLine = line.split('\t')#按制表符进行分割
        returnMat[index] = listFromLine[0:3]
        
        if listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
        elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
        else:
                classLabelVector.append(4)
        index += 1
    return returnMat,classLabelVector
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]          #取得矩阵行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #inX矩阵在行方向上重复数据集的行数
    sqDiffMat = diffMat**2                          #平方
    sqDistances = sqDiffMat.sum(axis=1)         #压缩矩阵将一行的数据进行相加成一列
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     #排序返回从小到大的根据每行的距离最小的返回索引
    classCount={}                               #创建一个字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #从索引行数得知标签名
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #将前K个标签名加入到字典中，假如原字典中无key则创建key并初始化为0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按字典的value进行排序
    return sortedClassCount[0][0]  #返回字典第一对key/value中的key

def datingClassTest():
    hoRatio = 0.6     #hold out 10%
    datingDataMat,datingLabels = file2matrix('a.txt')       #加载数据集以及标签集
    normMat, ranges, minVals = autoNorm(datingDataMat)      #将数据集进行归一化
    m = normMat.shape[0]                                    #取得归一化数据集的行数
    numTestVecs = int(m*hoRatio)                            #测试次数
    errorCount = 0.0                                        #初始化判断错误的个数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)  #将数据喂给训练函数
        #print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))       
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0                     #如果不相等，计数器加一
    print("错误的总数为 %d" % errorCount)
    print("训练集总数为 %d" % numTestVecs)
    print ("错误率为  %f" % (100*(errorCount/float(numTestVecs))))         #打印错误率
    
	
def autoNorm(dataSet):
    minVals = dataSet.min(0)     #取得array中每一列中每一列的最小
    maxVals = dataSet.max(0)     #取得array中每一列中每一列的最大
    ranges = maxVals - minVals   
    normDataSet = zeros(shape(dataSet))     #创建和传入数组一样的array
    m = dataSet.shape[0]                  #获取array的行数

    normDataSet = dataSet - tile(minVals, (m,1))    #根据公式newValue = (oldValue-min)/(max-min)算法   tile函数重复minVals 
    normDataSet = normDataSet/tile(ranges, (m,1))   #❶征值相除
    return normDataSet,ranges,minVals
def classifyPersin():
    infact = []
    infact.append(float(input('每年获得的飞行常客里程数：')))
    infact.append(float(input('玩视频游戏所耗时间百分比:')))
    infact.append(float(input('每周消费的冰琪淋公升数:')))
    Labels = ['不喜欢','魅力一般','极具魅力']
    returnMat,classVect = file2matrix('Date.txt')
    retrurMat,ranges,minvals = autoNorm(returnMat)
    objected = classify0((array(infact)-minvals)/ranges,returnMat,classVect,3)
    print(Labels[objected - 1])
    

classifyPersin()
#datingDateMat,datingLables = file2matrix('a.txt')   #打开文件
#datingDateMat,ranges,minvals = autoNorm(datingDateMat)  #归一化处理
#fig = plt.figure() #创建图像整体对象
#ax = fig.add_subplot(111) #创建子图
#ax.scatter(datingDateMat[:,0], datingDateMat[:,1],15.0*array(datingLables), 15.0*array(datingLables)) #画散点图
#plt.show()  #展示图像
