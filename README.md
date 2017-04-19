# 电脑摄像头扫描答题卡

可能是最灵活准确率最高的摄像头答题卡扫描开源实现

## 主要参考
* [python CV 趣味项目 答题卡识别](http://www.jianshu.com/p/2bbdb27ee7b3)
* [Tutorial: Creating a Multiple Choice Scanner with OpenCV](http://blog.ayoungprogrammer.com/2013/03/tutorial-creating-multiple-choice.html/)

## 应用场景
答题卡铺在桌子上，椅子上，键盘上（如测试图片），豆腐上等等随便你，然后拿起摄像头对着答题卡。。。，识别单选答题卡效率最高！

## 市面上开源代码主要缺点
* 代码里面都有选项距离等等相关硬参数，从而导致摄像头扫描或照片识别效率低下
* 答题卡太简单，易于识别，但是实际使用中不可能有如此简单的答题卡，比如用HoughCircle检测圆形选项框，选项一多，直接坑爹
* 答题卡太复杂，在答题卡上加了一堆定位图形，比如答题卡是3列20排，定位图形足足有23个，累死编制答题卡的人

所以，[answer-sheet-scan](https://github.com/inuyasha2012/answer-sheet-scan)基本上是市面上答题卡识别准确率最高的开源代码

## 整个流程如下，详细请看代码注释
* 打开摄像头扫描答题卡，看，是这个挫样

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p1.png)

* 把这个比较挫的图片进行二值化腐蚀膨胀边缘检测，变成了这样，还是很挫

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p2.png)

* 计算轮廓，并且看最大轮廓是否具有4个顶点，如果有的话，就OK了

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p3.png)

* 纠偏，把斜的图片变正，看上去终于不挫了

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p4.png)

* 调整图片的亮度，方便二值化

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p5.png)

* 通过二值化膨胀腐蚀再二值化，获得涂写的区域

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p6.png)

* 通过二值化膨胀腐蚀，获得所有的选项框加题号区域

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p7.png)

* 依据面积大小和其他条件提取合适的轮廓，与涂写区域重叠

![](https://github.com/inuyasha2012/answer-sheet-scan/blob/master/pic/p8.png)

* 依据轮廓左上角坐标从上而下排序轮廓，若1排有3题，每题4个选项，则认为前5个轮廓是第1题，其中第1个轮廓是题号，第6-10个轮廓是第2题，其中第6个轮廓是题号

* 根据提取的轮廓的左上角坐标和长宽，计算轮廓区域内的白点个数，白点个数低于某个阈值，初步认为是选择了该选项

## 其他
其他细节（例如选项框轮廓个数检测）详见代码以及注释
