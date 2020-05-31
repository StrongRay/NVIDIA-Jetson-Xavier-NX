# NVIDIA-Jetson-Xavier-NX
Unboxing the NVIDIA Xavier NX 

https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-xavier-nx/

Got hold of a Xavier NX from SEEED, flew over from SZ to SG in 7 days.
Here's what this IOT super-charged "PI" can do.  It's simply amazing.  I complied codes and do concurrent tasks and the system is stable.
I can also toggle the use of CORES and wattage.  More importantly, got to download a few stuff. such as jtop.

# First Impression

Hunt for jTop and install to see the various stuff running.  It's like top except tailored for NVIDIA

![alt text](https://github.com/StrongRay/NVIDIA-Jetson-Xavier-NX/blob/master/jTop.png)

Not going to write alot, but here's some videos that I captured to show the power of this device
Deepstream 5 is not pre packaged with the intial load.  You can just go hunt and do some git clone and compliation.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=tFOx3nSJKW8" target="_blank">
  <img src="http://img.youtube.com/vi/tFOx3nSJKW8/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

Once, you get the compliation right, you can run the various kinds of application from the samples

This second sample shows object detection

<a href="http://www.youtube.com/watch?feature=player_embedded&v=q2VBGvSnWl4" target="_blank">
  <img src="http://img.youtube.com/vi/q2VBGvSnWl4/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

This 3rd item is to test out my Pi camera and doing some OpenCV exploration.  I wanted to find out the response time and how easy it is.
Paul McWhorter has some very nice videos on OpenCV masking and on Nano.  I brought in the codes, made some modifications to detect the BLUE marker.  Got out my servos but did not do the coding to move the motors when the item moves.  That should be exciting too.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=QBlN-mBAQ60" target="_blank">
  <img src="http://img.youtube.com/vi/QBlN-mBAQ60/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

The final part is actually my favourite, using YOLO v4 https://github.com/AlexeyAB/darknet
I can easily compile without much modification.  The performance is AMAZING.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=L13paMq-cXU" target="_blank">
  <img src="http://img.youtube.com/vi/L13paMq-cXU/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

Given I have done this on my UBUNTU laptop with 2 cores, my NANO and now this XAVIER, definitely no drop of speed and continuously processing the video file.

Enjoy! .. GTC 2020 was like May 15, 16 days later, I am able to see this and from Singapore. Without much training.  Perhaps, my unix is abit rusty. But was able to move my 256GB MicroSD card and salvage a number of pieces of hardware from my tool box.

Cheers.  And if I am 55 years old and can do this, any younger person MUST be able to do better than me =)





