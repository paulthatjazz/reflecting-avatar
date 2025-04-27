# Avatar Emotion Reflecting Face Movement

This is the code used for my MSc dissertation titled: Avatar Emotion Reflecting Face Movement. The experiment uses the model, FaceMesh, as a base to detect face movements from a webcam feed. The first stage of the project was to take this input and project it onto a virtual avatar. Secondly, I experimented with two factors: voice and emotion.

## Voice

Data was gathered using fourier analysis. This would be an input for a CNN powered by TensorFlow.js. This model would be capable of inferring facial positions from sound by classifying segments based on the phonetic alphabet. 

## Emotion

Second, we developed a lightlight facial emotion recognition model via KNN classification and a selection of the datapoints from facemesh. This would become a parameter for the animation to make sufficient changes to reflect emotional state.

## Acknowledgements 

This project was built on top of the work here:  https://github.com/tensorflow/tfjs-models/tree/master/facemesh/demo
