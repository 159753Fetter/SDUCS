# Review_of_Computer_Graphics
计算机图形学复习

> 适用人群： 上课没好好听课或者似懂非懂迷迷糊糊的同学

- 教材：Interactive Computer Graphics: A Top-Down Approach with Shader-Based OpenGL Sixth Edition. 
https://inspirit.net.in/books/academic/Interactive%20Computer%20Graphics.pdf

- 教材课后习题部分答案：https://sxbb.me/GUsHv

计算机图形学的**pipeline**分为以下几个部分：
> 1. 顶点处理模块
> 2. 裁剪模块和图元组装模块
> 3. 光栅化模块
> 4. 片元处理模块


下面就每个部分的重点整理出一些高质量的链接来帮助理解与复习

### **顶点处理模块**

1. **关于坐标变换**

> 模型坐标系到世界坐标系再到相机坐标系是如何处理的？
  
> 到达相机坐标系后是如何进行投影的(正交投影与透视投影)？
  
> 投影完成后是如何进行视口变换的？
  
> https://blog.csdn.net/zhongjling/article/details/8488844

> https://blog.csdn.net/zhanglongit/article/details/7525918  
  
> http://www.360doc.com/content/10/0917/09/1066008_54279529.shtml

> https://blog.csdn.net/hiramtan/article/details/9020419

2. **关于glLookAt()和glPerspective() 函数的解释**

> https://blog.csdn.net/ivan_ljf/article/details/8764737

> https://blog.csdn.net/augusdi/article/details/19936239

3. **关于图元绘制**

> http://www.mamicode.com/info-detail-1139707.html

4. **关于四元数**

> https://www.zhihu.com/question/23005815 第二个回答


### **裁剪模块**

1. **直线裁剪:**

> **Cohen-Sutherland Algorithm**

    -这个算法见教材 p311,总共分为四步，但是想要了解真正的步骤，还需要一个例子。参考 exam/CG2017spring试卷-B-答案.doc PartII 的第四题。
    
> **Liang-Barsky Clipping**
    
    -这个算法的实现在 powerpoint/AngelCG21.ppt 的18，19页。
    
一个关于两个算法的链接 https://blog.csdn.net/pleasecallmewhy/article/details/8393445    
    
2. **多边形裁剪:**

> **sutherland-hodgman algorithm**

> 一个粗略的解释:https://blog.csdn.net/damotiansheng/article/details/43274183

> 一个让你理解算法是如何运行的视频: https://www.youtube.com/watch?v=Euuw72Ymu0M

### **直线/圆生成与多边形填充算法**

1. **DDA-line generation**

> https://www.tutorialspoint.com/computer_graphics/line_generation_algorithm.htm

2.  **Bresenham-line generation**

> https://www.tutorialspoint.com/computer_graphics/line_generation_algorithm.htm

> 一个例子: https://wenku.baidu.com/view/af9fa8f15fbfc77da269b1ad.html

3. **MidPoint-line generation**

> http://www.mat.univie.ac.at/~kriegl/Skripten/CG/node25.html

4. **Bresenham-circle generation**

> http://run.usc.edu/cs420-s15/recitations/circleRasterization.pdf

5. **MidPoint-circle generation**

> http://www.mat.univie.ac.at/~kriegl/Skripten/CG/node27.html

6. **奇偶规则和非零环绕树规则**

>  https://blog.csdn.net/mmogega/article/details/53066735

7. **Scan-line filling**

> https://blog.csdn.net/u013044116/article/details/49737585

> https://www.jianshu.com/p/d9be99077c2b

### **光照模型**

1. **phong 模型**

2. **改进的phong模型**

看课本。

3. **多边形shading方法：**
>  flat shading 

>  gouraud shading

>  phong shading 

这三种方法读课本足以明白。

一个关于phong模型 和改进phong模型区别的很好的例子：
> https://blog.csdn.net/wangdingqiaoit/article/details/52964990

4 **相关函数使用**
> http://blog.sina.com.cn/s/blog_6163bdeb0100p01x.html

### **纹理映射**
1. **三种映射方法**
> 纹理映射
> 环境映射
> 凹凸映射

> 纹理映射的一堆坐标的关系....
> https://blog.csdn.net/wangdingqiaoit/article/details/51457675 

> 环境映射:https://blog.csdn.net/u010223072/article/details/45193283

> 凹凸映射: https://www.cnblogs.com/lancidie/archive/2010/11/05/1869979.html

### **空间数据的划分与查找（四叉，八叉，kd-tree等）**

> http://insaneguy.me/attachments/spatial_ds--bsp_tree-octree-kd-tree.pdf

### **隐藏面消除**

1. z-buffer 算法 实现以及时间复杂度
2. 画家算法
3  深度排序
见课本 p331(通读一遍就懂了)

### **光线追踪**

原理+实现: P559-P565

### **曲线和曲面**

见powerpoint/AngelCG36&37.ppt

### **走样与反走样**

见powerpoint/aliasing and anti-aliasing.ppt

https://www.colabug.com/2098676.html

### **编程**
1. **opengl 核心模式编程**
https://learnopengl-cn.github.io/intro/

2. **立即渲染模式编程**
http://www.glprogramming.com/red/

### **一个很好的题库**
https://www.student.cs.uwaterloo.ca/~cs488/Fall2017/q.pdf

berkeley/CG exam: https://hkn.eecs.berkeley.edu/exams/course/cs/184


### **各种小知识点比如求交计算之类的**
1. **Ray-triangle intersection**
https://courses.cs.washington.edu/courses/csep557/10au/lectures/triangle_intersection.pdf
2. **如何根据如何光线和法线计算反射光线(Phong 光照模型)**
见课本 P275
3. **比较有用的课后习题**
2.9,2.11,2.19,3.1,3.4,3.7,3.8,3.9,3.18,4.1,4.7,4.17,5.2,5.3,5.7,6.1,6.2,6.3,6.9,6.12,6.13,7.1,7.3

### **闲着没事可以看的东西**
1. 计算几何(听过的最好的课程)
http://www.xuetangx.com/courses/course-v1:TsinghuaX+70240183x_1+sp/about
2. 线性代数的本质(看完这个应该坐标系变换很容易理解了)
https://www.bilibili.com/video/av5977466/


### **常用英文词汇**
1.hexagon 六边形
2.Specular surface 镜面
3.Diffuse surface 漫反射
4.opaque 不透明的
5.angle of incidence：入射角
6.angle of reflection: 反射角
7.jaggedness 锯齿
8.oblique 倾斜的
9.Singularities 奇点
10.monochromatic 单色的
11.sphere, cylinder, or cube 球面，圆柱面，立方体
12.dependent variable 因变量
13.independent variable 自变量
14.curvature 曲率
15.refraction 折射
16.oblique 倾斜的
17.Gouraud
18.recusion: 递归
19.object viewer light materials

