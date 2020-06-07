# NVIDIA-Jetson-Xavier-NX
Unboxing the NVIDIA Xavier NX 

https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-xavier-nx/

Got hold of a Xavier NX from SEEED, flew over from SZ to SG in 7 days.
Here's what this IOT super-charged "PI" can do.  It's simply amazing.  I complied codes and do concurrent tasks and the system is stable.
I can also toggle the use of CORES and wattage.  More importantly, got to download a few stuff. such as jtop.

# What to install 1st

## jtop

jtop gives you the needed view of what is running in the background.  The raw version is top. So, this is a custom top.
Here are the commands

```
git clone https://github.com/rbonghi/jetson_stats.git
sudo apt-get install python3-pip
sudo -H pip3 install -U jetson-stats

```
## DeepStream 5.0 Preview

Deepstream toolkit allows you to natively develop Computer vision codes leveraging on the hardware. 

```
sudo dkpg -i deepstream-5.0_5.0.0-1_arm64.deb 
cd /opt/nvidia/deepstream/deepstream
tar xf ~/Downloads/deepstream_python_v0.9.tbz2 -C sources
```

## Install Visual Code

As an IDE, its abit better than ATOM but ATOM is more colourful.  Still, with this, you can test codes easier and supposingly debugging. But appeared i needed curl


```
sudo apt-get install curl
. <( wget -O - https://code.headmelted.com/installers/apt.sh )
sudo apt-get install libcanberra-gtk-module 
sudo apt-get install v4l-utils
cd /opt/nvidia/deepstream/deepstream/samples
deepstream-app -c ./configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```
if you needed more stuff, i think you need to compile correctly to run
```
deepstream-app -c ./configs/tlt_pretrained_models/deepstream_app_source1_dashcamnet_vehiclemakenet_vehicletypenet.txt
```

## Install kazam

Kazam is your video capture for the screen. Useful for documentation
```
sudo apt install kazam
```

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
  
# OPENCV and Camera

This 3rd item is to test out my Pi camera and doing some OpenCV exploration.  I wanted to find out the response time and how easy it is.   Will expand on the details next update. 

Paul McWhorter has some very nice videos on OpenCV masking and on Nano.  I brought in the codes, made some modifications to detect the BLUE marker.  Got out my servos but did not do the coding to move the motors when the item moves.  That should be exciting too.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=QBlN-mBAQ60" target="_blank">
  <img src="http://img.youtube.com/vi/QBlN-mBAQ60/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

# YOLO V4

You Only Look Once (YOLO) is a fast library for Computer Vision.

The final part is actually my favourite, using YOLO v4 https://github.com/AlexeyAB/darknet
I can easily compile without much modification.  The performance is AMAZING.

However, you will need to do the following changes to your
```
git clone https://github.com/AlexeyAB/darknet.git
```
go install nano [editor for quick editing of any text file]
then go add this to your ~/.bashrc file
Basically, when you do a make, it expects to find opencv4.pc and the parameter **PKG_CONFIG_PATH**  must be updated with the following
```
export PKG_CONFIG_PATH='/usr/lib/aarch64-linux-gnu/pkgconfig'
```
Also, nvcc will hit an error. Do a find to locate where the nvcc is installed and add to the bashrc.
```
export PATH=/usr/local/cuda-10.2/bin:$PATH
```
Once done, you can follow the https://github.com/AlexeyAB/darknet to test out the detector
Do make changes to the Makefile to set OPENCV and other GPU stuff.  And remember to download your weights via wget ..

<a href="http://www.youtube.com/watch?feature=player_embedded&v=L13paMq-cXU" target="_blank">
  <img src="http://img.youtube.com/vi/L13paMq-cXU/0.jpg" alt="Deep Stream v5 sample app" width="640" height="480" border="0" /></a>

Given I have done this on my UBUNTU laptop with 2 cores, my NANO and now this XAVIER, definitely no drop of speed and continuously processing the video file.

Enjoy! .. GTC 2020 was like May 15, 16 days later, I am able to see this and from Singapore. Without much training.  Perhaps, my unix is abit rusty. But was able to move my 256GB MicroSD card and salvage a number of pieces of hardware from my tool box.

# Mask or No Mask

Tensorflow 2.1 doesn't come prebuilt on NX. However, it is easy to install.  I wanted to see how it reacts to both 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=EnPb01ldcHQ" target="_blank">
  <img src="http://img.youtube.com/vi/EnPb01ldcHQ/0.jpg" alt="Mask or No Mask" width="640" height="480" border="0" /></a>



# Afterthoughts:

We drabble into these once awhile, and Harish Pillay (Redhat) once said that he used github to document his instructions on Wireless#SG.   I find it an extremely good advice.  As we wipe out cards or write in notebooks, these instructions are lost and there is no "cloud" version.  So this can be extremely useful for someone either fresh on UBUNTU or on NVIDIA.  I even reference the part of increasing my RAM in my earlier post here for the commands.  So, as this builds, I can slowly evolve  this into a digital engineering logbook.  That discipline is what ENGINEERING students do, keep a log book and document the observations and steps.    And given this is open and 24x7, someone out there might find this useful and my advice is pass it on. Do your own logbook and help others.  The knowledge is open but the beauty is the skill. This is an instance of an object. The FISH and not fishing itself.  To move around, requires more than just this basic knowledge as it involves disciplined mind, interest and above all, many other disciplines like programming and debugging.  






