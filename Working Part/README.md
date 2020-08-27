# Woking Part folder.

This folder contains 2 scripts names "script_with_ads.py" and "script_without_ads.py".

First download the model file from the following link

https://u.pcloud.link/publink/show?code=XZq63XXZJFvoKWU6iUXfeNXcN5XnOYpDmYsX

and save it with the name "age-gender.model" in the "Working Part" directory

To run script_with_ads.py type the following commands.

```sh
python script_with_ads.py --video "./videos/Trim 14.mp4" --frame_skipping_rate 8 --ad_time_duration 5
```
This script will fetch the specified video file from the videos folder and will also take frame_skipping_rate option, this option comes in handy when 
you don't have a GPU but need to run the project. The add_time_duration parameter will keep the ad for 5 frames and after that it will fetch new ad.

To run script_without_ads.py type the following commands.

```sh
python script_without_ads.py --video "./videos/Trim 15.mp4" --frame_skipping_rate 8
```
The only thing that is changed here is that it does not take ad_time_duration because it does not show ad. it just fetches the video file and will do 
age and gender recognition on the frame.
