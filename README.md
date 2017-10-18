# face_detection_base_on_mtcnn

It based on tensorflow for training implementation , very simple and easy.
<br/>

##Prerequisites
1.You can use gpu or cpu to train this model
2.Our data comes from WIDER Face(http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and mmlab(http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)

## train
1.Base issues
a.sample,we use three different kinds of data annotation in our training
process.(i)Positives:IoU(the Intersection-over-Union) above equal 0.65 to a ground truth face. 
(ii) Negatives:IoU ratio less than 0.3 to any ground-truth faces
(iii)Part faces:IoU between 0.4 and 0.65 to a ground truth face
2.Network structure, mtcnn is divided into three small networks
![net.png](https://github.com/zhangcheng007/face_detection_base_on_mtcnn/blob/master/netgraph/net.png)

##Result
![1.jpg](https://github.com/zhangcheng007/face_detection_base_on_mtcnn/blob/master/test/result/1.jpg)
![3.jpg](https://github.com/zhangcheng007/face_detection_base_on_mtcnn/blob/master/test/result/3.jpg)
![5.jpg](https://github.com/zhangcheng007/face_detection_base_on_mtcnn/blob/master/test/result/5.jpg)

## References
https://github.com/dlunion/mtcnn<br/>
https://github.com/CongWeilin/mtcnn-caffe<br/>
https://github.com/AlphaQi/MTCNN-light<br/>
https://github.com/dlunion/CCDL<br/>








