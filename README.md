# 电脑摄像头扫描答题卡

## 主要参考
* [python CV 趣味项目 答题卡识别](http://www.jianshu.com/p/2bbdb27ee7b3)
* [Tutorial: Creating a Multiple Choice Scanner with OpenCV](http://blog.ayoungprogrammer.com/2013/03/tutorial-creating-multiple-choice.html/)

## 市面上开源代码主要缺点
* 代码里面都有选项距离等等相关参数，从而导致摄像头扫描或照片识别效率低下
* 答题卡太简单，易于识别，但是实际使用中不可能有如此简单的答题卡

所以，[answer-sheet-scan](https://github.com/inuyasha2012/answer-sheet-scan)基本上是市面上答题卡识别准确率最高的开源代码

## 整个流程如下，详细请看代码注释
* 打开摄像头扫描答题卡，看，是这个挫样

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p1.png)

* 把这个比较挫的图片进行二值化腐蚀膨胀边缘检测，变成了这样

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p2.png)

* 计算轮廓，并且看最大轮廓是否具有4个顶点，如果有的话，就OK了

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p3.png)

* 纠偏，把斜的图片变正

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p4.png)




