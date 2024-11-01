# PosturePal - Your Personal AI Fitness Coach
link to the kaggle notebook: 'https://www.kaggle.com/code/vrajnandaknangunoori/posturedetectorcorrector/edit'

![pushups-sample-out-ezgif com-optimize](https://github.com/user-attachments/assets/47cd3d88-b9eb-43c8-8fc1-068aace59706)


https://github.com/user-attachments/assets/d5611251-b60f-4173-ad67-8c0c62684092


**PosturePal** automates exercise tracking and guidance, ensuring users can focus on their exercises while receiving real-time feedback. The system uses advanced computer vision techniques to recognize exercises, count repetitions, analyze posture, and track progress.

## Key Features

- **Exercise Recognition**: Automatically identifies exercises using 3D pose landmarks, ensuring proper form and technique throughout the workout.
- **Repetition Counter**: Tracks repetitions accurately using an enter-and-exit threshold system to prevent false rep counts.
- **Posture Analysis**: Provides real-time feedback on alignment and posture, helping reduce injury risks and improve performance.
- **Progress Tracking**: Offers detailed analytics on your workout, including the ability to review footage and pinpoint areas where posture may have faltered.
- **Seamless Exercise Switching**: Allows for smooth transitions between exercises during your workout.
- **Advanced Analysis Tools**: Analyze your workout with visualized data to identify areas for improvement.

## Dataset Used
The following dataset from kaggle has been used which contains videos of various workouts 'https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video'.

## Tech Stack
- python
- computer vision
- mediapipe framework
- kaggle

## Architecture Overview

![bg_image](https://github.com/user-attachments/assets/97642847-4157-4067-9f7c-297015c3dd1f)

### FullBodyPoseEmbedder
The `FullBodyPoseEmbedder` class converts 3D pose landmarks into a 3D embedding for pose analysis or classification. It normalizes the input landmarks by adjusting for translation (centered around the hips) and scaling (based on torso size or maximum distance to the pose center). The embedding captures key spatial relationships between body parts.

### PoseClassifier
The `PoseClassifier` class classifies exercises based on pose landmarks. It loads pose samples, embeds them, and classifies new poses by comparing them to the stored samples. It also filters outliers and detects pose anomalies using nearest neighbor comparisons.

### EMADictSmoothing
The `EMADictSmoothing` class smooths pose classification data using Exponential Moving Average (EMA), reducing abrupt changes and providing more stable predictions over time.

### RepetitionCounter
The `RepetitionCounter` class counts repetitions for poses of an exercise by monitoring pose classification confidence. It uses a dual-threshold system to track when a user enters and exits a pose, ensuring reliable counting.

### Bootstrap Helper
The `BootstrapHelper` class handles the initialization of the model by loading the training data, preparing the dataset, and setting up the necessary pre-requisites.

### ExerciseTracker
The 'ExerciseTracker' helps maintain the reps of each exercise by maintaing a RepetitionCounter for each of the intermediate states in a pose and gives real-time guidance using the same.

## To run
Clone the repo and run the command 'python app.py' on your terminal. Visit the local link generated to be able to use the application.

## What it does
The app/website accesses the device's camera and records users as they exercise in real-time. It automatically detects the exercise being performed, eliminating the need for users to manually specify their workout. Using machine learning algorithms like KNN classification and computer vision models and pre-processing techniques, it accurately identifies the exercise and tracks the number of reps. The standout feature is its real-time audio and visual feedback, which guides users on improving their posture and form. Additionally, the platform includes a community forum where users can share tips and advice. It also features a fun mini-game where participants earn coins for completing exercises correctly. These coins can later be used to unlock premium features that will be introduced on the platform.


## What's next for PosturePal – Your Personal AI Fitness Coach
Next, I plan to enhance the accuracy of exercise detection by further expanding the dataset and improving the AI algorithms. I am also looking to introduce more interactive features, such as personalized workout plans, and extend the platform to integrate with wearable devices for even more precise tracking. I hope to roll it out to local gyms, schools, and community centers to encourage group fitness activities across communities.
