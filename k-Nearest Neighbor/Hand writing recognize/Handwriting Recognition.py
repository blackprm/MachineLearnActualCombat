from DatingTest import *
from numpy import *
from os import listdir       #此包可以读取文件夹文件名字，并按自然排序排序
def imgVector(filename):    #将0-1字符画转化为 一维 向量
    returnVector = zeros((1,1024))    #创建1* 1024的array来存储图片字符的信息
    fr = open(filename)             #打开要转换的文件
    for i in range(32):             #总共读取32行
        lineStr = fr.readline()     #读取一行
        for j in range(32):         #将每一个数字写入数组
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector
def handwritingClassTest():  #测试错误率
    hwLabels = []           #测试集标签
    trainingFileList = listdir('trainingDigits')           #加载测试集中文件名字
    m = len(trainingFileList)                               #测试集总个数
    trainingMat = zeros((m,1024))                           #创建即将做运算的数组
#================将所有测试集中的文件转化为向量====================================#
    for i in range(m):                                      
        fileNameStr = trainingFileList[i]                   #取得文件的名字
        fileStr = fileNameStr.split('.')[0]                 #去掉名字中的 .txt
        classNumStr = int(fileStr.split('_')[0])            #取得文件代表的数字
        hwLabels.append(classNumStr)                        #加入标签数组
        trainingMat[i,:] = imgVector('trainingDigits/%s' % fileNameStr)#将对应文件转化为向量
        
#================将所有测试集中的文件转化为向量====================================#        
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     # 去掉名字中的 .txt
        classNumStr = int(fileStr.split('_')[0])#取得文件代表的数字
        vectorUnderTest =imgVector('testDigits/%s' % fileNameStr)#将对应文件转化为向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)#去前三个最近的点中最多的一个
        #print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
        
    print( "\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))



def HandWritingInterface(testFileName,trainNameMat):        #提供测试图片的接口
    hwLables = []
    trainFileList = listdir(trainNameMat)
    maxValueRows = len(trainFileList)
    trainMat = zeros((maxValueRows,1024))
    print('正在加载训练集....')
    for i in range(maxValueRows):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        fileStr = fileStr.split('_')[0]
        hwLables.append(int(fileStr))
        trainMat[i,:] = imgVector(trainNameMat + '/' + fileNameStr)
    print('加载训练集完成....')
    testPoint = imgVector(testFileName)
    print('正在计算....')
    return classify0(testPoint,trainMat,hwLables,3)
fileName = input('输入要识别的文件完整拓展名：')
print('识别出的数字为 ：'+ str(HandWritingInterface(fileName,'trainingDigits')))

