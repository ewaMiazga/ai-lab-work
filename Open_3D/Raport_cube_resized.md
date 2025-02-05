I created some raport about my progress.
In short:
**1.** I experimented with mono, stereo depth maps and algorithms to extract edges from 3d pictures, but none of the ones I tried gave me satisfactory results.
**2.** I followed the course of computer vision and it seems to be tricky task and in many cases the results are highly dependent on the lighting the position of the camera relative to scene, occlusion etc...
**3.** Also nowadays people mainly use deep nets or transformers to do any computer vision tasks. It is still mostly domain specific and it is hard to cover variety of scenes by one algorithm.
**4.** The high contrasts on the image between objects, especially foreground - potentially our object and background matters.
**5.** When I tried to do some more research and solve the problems the methods I was testing was vulnerable to I discovered open3D - so for now it seems to be the way to approach the task we have.
**6.** **Open3D** helps to model 3d structures pretty successfully. It requires to have an access to color image and its depth map - i didn't try to test any methods with stereo pictures. 
- link: https://www.open3d.org/
- it allows to set the detailed params of the camera used for taking a pictures, we work on. So the results will be better in the case of using this library with robot navigating the environment.
- it basically models 3d shapes obtained from pictures creating point clouds out of it. You can then create mashes, down sample points, create bounding boxes around the objects, cut them from environment, calculate their normals, drop outliers and many more.
- So I made a pipeline to try to apply some logic and extract the object from the photo. I will send you the example I tried on with some of intermediate results.
- My idea is to create a convex hull ot of visualization - you can see it as read lines around the object. Or choose only certain points that roughly define edges of the object from point cloud to be able to model the hierarchical and spatial characteristics of the object. 
- each point in the cloud has assigned x,y,z coordinates os I think it is suitable. 
- Let me know if you have some questions and what do you think about this idea. I will send you the video how I run the notebook I prepared with my results and the notebook itself.

**Results**:
Here is the video showing the results:

[Watch the video](Results-video.mp4)
