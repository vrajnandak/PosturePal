# PosturePal - Your Personal AI Fitness Coach
link to the kaggle notebook: 'https://www.kaggle.com/code/vrajnandaknangunoori/posturedetectorcorrector/edit'

## To run
Clone the repo and 'python app.py' and visit the local link generated.

## Inspiration
I noticed in my community that many elderly individuals and parents tend to overlook the importance of regular exercise, often seeing it as a waste of time. A common concern I heard was, "How do I know if I'm doing it correctly?" This inspired me to develop a simple, accessible solution to help address these concerns and try to improve the overall health of the community. My idea was to create an app/website that uses a camera to observe the user during exercises and provides real-time guidance with interactive visuals and controls. The app would be great for personal use, while the website could be used for group sessions, offering advice to multiple participants simultaneously. Exercising together would also create a motivational environment.

## What it does
The app/website accesses the device's camera and records users as they exercise in real-time. It automatically detects the exercise being performed, eliminating the need for users to manually specify their workout. Using machine learning algorithms like KNN classification and computer vision models and pre-processing techniques, it accurately identifies the exercise and tracks the number of reps. The standout feature is its real-time audio and visual feedback, which guides users on improving their posture and form. Additionally, the platform includes a community forum where users can share tips and advice. It also features a fun mini-game where participants earn coins for completing exercises correctly. These coins can later be used to unlock premium features that will be introduced on the platform.

## How I built it
The first step was to accurately predict the body given the camera's feed. I used Google's Mediapipe framework to do the same. With this, the camera feed is run through a self-trained classification model, which is trained on a comprehensive dataset of various exercises, that can automatically predict the exercise that the user is doing without explicit/implicit instructions from the user/admin. 
I used Google's Mediapipe framework to detect body landmarks, which allowed us to train the system on a comprehensive dataset of various exercises for accurate recognition and feedback.

## Challenges we ran into
One of the main challenges was the positioning of the camera. If the dataset lacked enough data points for certain positionings of the camera, this could lead to errors. However, for the most common camera positions, the system works reliably.

## Accomplishments
The system can reliably identify exercises and count reps, even when the camera is placed at different angles, making the platform user-friendly and flexible. This eliminates the hassle of manually selecting exercises, providing a smoother experience.

## What I learned
I learned the importance of refining the dataset to handle diverse camera angles and body types for more accurate exercise detection. Additionally, understanding user engagement and how feedback (both audio and visual) enhances motivation was an insightful takeaway.

## What's next for PosturePal – Your Personal AI Fitness Coach
Next, I plan to enhance the accuracy of exercise detection by further expanding the dataset and improving the AI algorithms. I am also looking to introduce more interactive features, such as personalized workout plans, and extend the platform to integrate with wearable devices for even more precise tracking. I hope to roll it out to local gyms, schools, and community centers to encourage group fitness activities across communities.
