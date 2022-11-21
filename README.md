# SaveTheWorld

SaveTheWorld is creative programming project, with the aim of raising the awareness of the impact of our gestures on the environment.

This installation is implemented with Python (back-end), TouchDesigner (front-end) and SuperCollider for the music landscape. 

The top feature is the extremely user-reactive interface.

<p align="center">
  <img src="BigliettoCPAC.png">
</p>

You can find more here: 
This version is avaiable only for desktop device.

***

# Abstract

Today, we hear about climate change almost everyday: on the news, tv shows and social
media. It’s a topic that is urgent to address, in order to preserve the planet we’re living in and be sure to make it habitable for further generations. 
The situation is critical and we all have to make an effort changing our lifestyle in order to reach a sustainable result.

The goal of this project is to make people aware of how simple daily actions can heavily impact the environment. 
With this concept in our mind, we thought that an interactive installation could send a clear message to the user, i.e. that everybody can contribute to change things for the better, and so to "Save the World". 
We gathered data from the web to understand how much our daily habits can threat or can help our planet in terms of CO2 emissions, that we used as a impact quantifier.

Save The World's interface consists on icons showing the daily actions we chose and a 3D  responsive globe. 
The user will look at the icon and mimic the action displayed. 
These gestures will be captured and recognized, then, with a strong real time visual feedback, the future impact of what
the user has done will be shown on the globe, that will change accordingly. 
The effects on the Earth, seen by the user, must be thought in large-scale terms: as if all population
of the world does that action at that very same moment. 
The degradation, or the improvement, of the health of our planet will make the user associate the actions mimiced and their impact. 
In this way, a sensibilization takes place with the aim that even in real life he will remember what effect his lifestyle can have and act as a consequence. 

To have a visual representation of the world Touchdesigner has been used, exploiting its compatibility 
through OSC messages with Python, the latter responsible for taking and classifying the gestures. In the end, Supercollider will generate soundscapes.



# Useful Links on Environment

- App to measure your environment impact: https://aworld.org/

# References

MediaPipe:
- GitHub: https://google.github.io/mediapipe/solutions/pose; 
- Classification article: https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
- Generative Human Models: https://github.com/google-research/google-research/tree/master/ghum
- Multiperson: https://www.section.io/engineering-education/multi-person-pose-estimator-with-python/

TouchDesigner: 
- Official documentation: https://docs.derivative.ca/Main_Page
- Introductive course (ITA): https://www.youtube.com/playlist?list=PLhhSecfe3762IytiKsT472xhnF1LciVJE
- Useful channel: https://www.youtube.com/channel/UCONptu0J1PCrW9YfBtSdqjA
- Introductive site with example: https://alltd.org/category/beginner/
