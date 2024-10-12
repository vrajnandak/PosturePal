import csv
import os
import io
import sys
import tqdm
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.framework.formats import landmark_pb2


class PoseEmbedder(object):
    def __init__(self, torso_size_multiplier=2.5):
        self._torso_size_multiplier = torso_size_multiplier
        self._landmark_names=[
            'nose',
            'left_eye_inner','left_eye','left_eye_outer',
            'right_eye_inner','right_eye','right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]
        
        #         Refer to this image file showing the landmarks of a body for identifying the different distances that are being used to create the 3D embedding: https://ai.google.dev/static/edge/mediapipe/images/solutions/pose_landmarks_index.png
        embedding_pairs_adj=[
            ('elbow', 'wrist'),
            ('shoulder', 'elbow'),
            ('hip', 'shoulder'),
            ('knee', 'hip'),
            ('ankle', 'knee')
        ]
        embedding_pairs_skip_1_joint=[
            ('shoulder', 'wrist'),
            ('hip', 'elbow'),
            ('knee', 'shoulder'),
            ('ankle', 'hip')
        ]
        embedding_pairs_skip_2_joints=[
            ('hip', 'wrist'),
            ('knee', 'elbow'),
            ('ankle', 'shoulder')
        ]
        embedding_pairs_skip_3_joints=[
            ('knee', 'wrist'),
            ('ankle', 'elbow')
        ]
        embedding_pairs_skip_4_joints=[
            ('ankle','wrist')
        ]
        
        find_embed_pairs=[]
        sides=['left_','right_']
        side_specific_pairs=[embedding_pairs_adj, embedding_pairs_skip_1_joint, embedding_pairs_skip_2_joints, embedding_pairs_skip_3_joints, embedding_pairs_skip_4_joints]
        for embedding_pairs_lst in side_specific_pairs:
            for embedding_pair in embedding_pairs_lst:
                lmk_from=embedding_pair[0]
                lmk_to=embedding_pair[1]
                for idx,side_from in enumerate(sides):
                    for side_to in sides[idx:]:
                        find_embed_pairs.append([side_from+lmk_from,side_to+lmk_to])
                        
        #Contains all the pairs of joints between which the distances will be measured.
        self._find_embed_pairs=find_embed_pairs
        
        
    #Creates an embedding from given landmarks(of the body). 'landmarks' is an numpy array.
    def __call__(self, landmarks):
        #landmarks should have the same number of values as there are in the landmarks we've decided to use(self._landmark_names).
#         assert landmarks.size[0]==len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])
        
        landmarks_copy=np.copy(landmarks)
        landmarks_copy_normalized=self._normalize_pose_landmarks(landmarks_copy)
        landmarks_embedding=self._get_pose_distance_embedding(landmarks_copy_normalized)
        
        return landmarks_embedding
    
    #Normalize the landmarks
    def _normalize_pose_landmarks(self,landmarks):
        landmarks_copy=landmarks.copy()
        
        pose_center=self._get_pose_center(landmarks_copy)
        landmarks_copy-=pose_center
        
        pose_size=self._get_pose_size(landmarks_copy)
        landmarks_copy_normalized=landmarks_copy/pose_size
        landmarks_copy_normalized*=100
        
        return landmarks_copy_normalized
    def _get_pose_center(self, landmarks):
        left_hip=landmarks[self._landmark_names.index('left_hip')]
        right_hip=landmarks[self._landmark_names.index('right_hip')]
        center=(left_hip+right_hip)*(0.5)
        return center
    def _get_pose_size(self,landmarks):
        landmarks_copy=landmarks.copy()
        landmarks_copy=landmarks_copy[:,:2]      #Discarding the z-dimension.
        
        left_hip=landmarks_copy[self._landmark_names.index('left_hip')]
        right_hip=landmarks_copy[self._landmark_names.index('right_hip')]
        hip_center=(left_hip+right_hip)*(0.5)
        
        left_shoulder=landmarks_copy[self._landmark_names.index('left_shoulder')]
        right_shoulder=landmarks_copy[self._landmark_names.index('right_shoulder')]
        shoulder_center=(left_shoulder+right_shoulder)*(0.5)
        
        #Calculating the euclidean distance between shoulder_center and hip_center.
        torso_size=np.linalg.norm(shoulder_center-hip_center)
        
        #Subtract the 'pose_center' from each landmark (i.e., x,y subtraction of pose_center for each landmark in landmarks).
        #This considers the landmark furthest from the pose center(but shouldn't pose_center be '0' here???) and then returns the distance of it.
        pose_center=self._get_pose_center(landmarks_copy)
        max_dist=np.max(np.linalg.norm(landmarks_copy-pose_center, axis=1))
        
        answer=max(torso_size*self._torso_size_multiplier, max_dist)
        return answer
    
    def _get_pose_distance_embedding(self,landmarks):
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_avg_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_avg_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])
        return embedding
#         embedding_lst=[]
#         embedding_lst.append(self._get_distance(self._get_avg_by_names(landmarks, 'left_hip', 'right_hip'), self._get_avg_by_names(landmarks, 'left_shoulder', 'right_shoulder')))
#         for embed_pair in self._find_embed_pairs:
#             embedding_lst.append(self._get_distance_by_names(landmarks,embed_pair[0],embed_pair[1]))
#         embedding_lst=np.array(embedding_lst)
#         return embedding_lst
    def _get_avg_by_names(self, landmarks, name_from, name_to):
        lmk_from=landmarks[self._landmark_names.index(name_from)]
        lmk_to=landmarks[self._landmark_names.index(name_to)]
        return (lmk_from+lmk_to)*0.5
    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from=landmarks[self._landmark_names.index(name_from)]
        lmk_to=landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)
    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to-lmk_from


class PoseSample(object):
    def __init__(self, name, landmarks, percentage_landmarks, class_name, embedding):
        self.name=name
        self.landmarks=landmarks
        self.percentage_landmarks=percentage_landmarks      #This field takes the positions of the landmarks without taking into account the frame shape.
        self.class_name=class_name
        self.embedding=embedding
        
class PoseSampleOutlier(object):
    def __init(self, sample, detected_class, all_classes):
        self.sample=sample
        self.detected_class=detected_class
        self.all_classes=all_classes
        
class PoseClassifier(object):
    def __init__(self,
                 pose_samples_file_path,   #The folder which contains the dataset to train on.
                 pose_embedder,
                 file_separator=',',
                 n_landmarks=33,
                 n_dimensions=3,
                 top_n_by_max_distance=30,
                 top_n_by_mean_distance=10,
                 axes_weights=(1.,1.,0.2)
                ):
        self._pose_embedder=pose_embedder
        self._n_landmarks=n_landmarks
        self._n_dimensions=n_dimensions
        self._top_n_by_max_distanc=top_n_by_max_distance
        self._top_n_by_mean_distance=top_n_by_mean_distance
        self._axes_weights=axes_weights      #These are used for classification of the pose sample in finding it's closest neighbors. By keeping a very low value of z, we reduce the values of the z-dimensions of the landmarks and hence it is likely that these z-dimensions will not be selected in the max_dist_heap.
        self._pose_samples=self._load_pose_samples(pose_samples_file_path,file_separator,n_landmarks,n_dimensions,pose_embedder)
        
    #Called only during initialization of this class(for loading the training data).
    def _load_pose_samples(self,file_path,file_sep,landmarks_num,dim_num,pose_embedder):
        pose_samples=[]
        with open(file_path,'r') as dataset_landmarks_file:
            csv_reader=csv.reader(dataset_landmarks_file, delimiter=file_sep)
            for row in csv_reader:
                #Each row is of the form ['img_name','class_name', landmarks]
                assert len(row) == 2*(landmarks_num*dim_num)+3, 'Wrong number of values: {}'.format(len(row))

                landmarks=np.array(row[3:102], np.float32).reshape([landmarks_num,dim_num])
                landmarks_percentage=np.array(row[102:],np.float32).reshape([landmarks_num,dim_num])
                pose_samples.append(PoseSample(name=row[2], class_name=row[1], landmarks=landmarks,percentage_landmarks=landmarks_percentage, embedding=pose_embedder(landmarks)))
        return pose_samples
        pass
    
    #Getting rid of all the input pose samples that are close to other samples belonging to a different class.
    def find_pose_sample_outliers(self):
        outliers=[]
        for sample in self._pose_samples:
            pose_landmarks=sample.landmarks.copy()
            pose_classification=self.__call__(pose_landmarks)
            
            #Considering only those classes with the highest values.
            class_names=[class_name for class_name, count in pose_classification.items() if count==max(pose_classification.values())]
            
            if sample.class_name not in class_names or len(class_names)!=1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))
        return outliers
    
    def __call__(self, pose_landmarks):
        assert pose_landmarks.shape==(self._n_landmarks, self._n_dimensions), 'unexpected shape: {}'.format(pose_landmarks.shape)
        
        pose_embedding=self._pose_embedder(pose_landmarks)
        flipped_pose_embedding=self._pose_embedder(pose_landmarks*np.array([-1,1,1]))
        
        max_dist_heap=[]
        #If you're iterating over the entire pose_samples available, then while you're trying to remove outliers, wouldn't we end up considering the same data point as the one we're trying to classify????
        for sample_idx, sample in enumerate(self._pose_samples):
            #Between the normal and the flipped embedding, it takes the value of some landmark numbered 'a' in it's dimension 'b' such that this landmark has the largest distance from our sample's embedding.
            #We could consider adding multiple landmarks in the order of max to min and then sort based on all of them.
            max_dist=min(
                np.max(np.abs(sample.embedding - pose_embedding)*self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding)*self._axes_weights)
            )
            max_dist_heap.append([max_dist,sample_idx])
        max_dist_heap=sorted(max_dist_heap, key=lambda x: x[0])   #Sort based on max_dist i.e., difference of landmarks.
        max_dist_heap=max_dist_heap[:self._top_n_by_max_distanc]
        
        mean_dist_heap=[]
        for _,sample_idx in max_dist_heap:
            sample=self._pose_samples[sample_idx]
            mean_dist=min(
                np.mean(np.abs(sample.embedding-pose_embedding)*self._axes_weights),
                np.mean(np.abs(sample.embedding-flipped_pose_embedding)*self._axes_weights)
            )
            mean_dist_heap.append([mean_dist,sample_idx])
        mean_dist_heap=sorted(mean_dist_heap, key=lambda x: x[0])    #Sorting on mean_dist.
        mean_dist_heap=mean_dist_heap[:self._top_n_by_mean_distance]
        
        class_names=[self._pose_samples[sample_idx].class_name for _,sample_idx in mean_dist_heap]
        result={class_name: class_names.count(class_name) for class_name in set(class_names)}
        return result
        
    def get_closest_point_to_current_point(self, user_dict):
        ideal_trainer_landmarks={class_name: [None,None] for class_name in user_dict.keys()}
        
        #the goal right now is to find the best pose for the user's frame.
        #We could either use the entire dataset and find the closest point to the user's landmarks and use this (or) we could filter data based on class_name(using the user's frame's class name) and then find the best landmarks.
        #Not sure if the first one would work as intended so we decided to use the latter method of figuring the ideal landmarks.
        
        #Instead of filtering the data all the time, we filter and store it once. Thus, giving priority to time than space.
        #We could just store this filtered data somewhere else, where we have to do this task of filtering only once but for now we've decided to do it here.
        filtered_samples_by_class_name={class_name: [] for class_name in user_dict.keys()}
        for pose_sample in self._pose_samples:
            if pose_sample.class_name in user_dict.keys():
                filtered_samples_by_class_name[pose_sample.class_name].append(pose_sample)
                
        
        for curr_class_name, user_best_frame_details in user_dict.items():
            #For the user's best frame, the below are the values.
            best_confidence_score=user_best_frame_details[0]
            best_output_frame=user_best_frame_details[1]
            best_landmarks=user_best_frame_details[2]
            
            if(best_confidence_score is None or best_output_frame is None):
                continue
            
            #THE CODE BELOW IS BASICALLY A REPETITION OF THE ABOVE STUFF IN THE '__call__' FUNCTION. LATER, MAKE FUNCTIONS FOR THESE AS WELL.
            pose_embedding=self._pose_embedder(pose_landmarks)
            flipped_pose_embedding=self._pose_embedder(pose_landmarks*np.array([-1,1,1]))

            max_dist_heap=[]
            for sample_idx, sample in enumerate(filtered_samples_by_class_name[curr_class_name]):
                #Between the normal and the flipped embedding, it takes the value of some landmark numbered 'a' in it's dimension 'b' such that this landmark has the largest distance from our sample's embedding.
                #We could consider adding multiple landmarks in the order of max to min and then sort based on all of them.
                max_dist=min(
                    np.max(np.abs(sample.embedding - pose_embedding)*self._axes_weights),
                    np.max(np.abs(sample.embedding - flipped_pose_embedding)*self._axes_weights)
                )
                max_dist_heap.append([max_dist,sample_idx])
            top_n_by_max_distanc=10
            max_dist_heap=sorted(max_dist_heap, key=lambda x: x[0])   #Sort based on max_dist i.e., difference of landmarks.
            max_dist_heap=max_dist_heap[:top_n_by_max_distanc]

            mean_dist_heap=[]
            for _,sample_idx in max_dist_heap:
                sample=self._pose_samples[sample_idx]
                mean_dist=min(
                    np.mean(np.abs(sample.embedding-pose_embedding)*self._axes_weights),
                    np.mean(np.abs(sample.embedding-flipped_pose_embedding)*self._axes_weights)
                )
                mean_dist_heap.append([mean_dist,sample_idx])
            top_n_by_mean_distance=1
            mean_dist_heap=sorted(mean_dist_heap, key=lambda x: x[0])    #Sorting on mean_dist.
            mean_dist_heap=mean_dist_heap[:top_n_by_mean_distance]

            ideal_trainer_landmarks[curr_class_name]=[filtered_samples_by_class_name[curr_class_name][mean_dist_heap[0][1]]]
        return ideal_trainer_landmarks


class EMADictSmoothing(object):
    def __init__(self, window_size=10, alpha=0.2):
        self._window_size=window_size
        self._alpha=alpha
        self._data_in_window=[]   #Keeps track of the recent most 'window_size' number of frames in the camera. The recent most frame is at the 0th position.
        
    def __call__(self, data):
        self._data_in_window.insert(0,data)
        self._data_in_window=self._data_in_window[:self._window_size]
        
        keys=set([key for data in self._data_in_window for key,_ in data.items()])
        
        smoothed_data=dict()
        for key in keys:
            factor=1.0
            top_sum=0.0   
            bottom_sum=0.0  
            for data in self._data_in_window:
                value=data[key] if key in data else 0.0 
                
                top_sum+=factor*value    #To keep track of the total sum of all frames weighed with respective weights. The recent frame affects the top_sum the most.
                bottom_sum+=factor       #To keep track of the total sum of all the weights.
                
                #Decreasing the weight for the next frame, thus reducing it's effect on the total sum.
                factor*=(1.0-self._alpha)
                
            smoothed_data[key]=top_sum/bottom_sum
        return smoothed_data

  class RepetitionCounter(object):
    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name=class_name
        
        #These threshold values are to put limits on the confidence scores. Right now, they represent the number of correct classes in the given list of pose_classification.
        self._enter_threshold=float(enter_threshold)
        self._exit_threshold=float(exit_threshold)
        
        self._pose_entered=False
        self._n_repeats=0
        
    @property            #This is basically a getter, you can access the '_n_repeats' without using parenthesis and using just rep_counter.n_repeats
    def n_repeats(self):
        return self._n_repeats
    
    def __call__(self, pose_classification):
        pose_confidence=0.0 
        if self._class_name in pose_classification:
            pose_confidence=float(pose_classification[self._class_name])
        
        if not self._pose_entered:
            print('self._enter_threshold: ',self._enter_threshold)
            print('pose confidence: ', pose_confidence)
            self._pose_entered=pose_confidence > float(self._enter_threshold)  #If there are more number of points of the class than 'enter_threshold' in the pose_classification(i.e., it's closest points).
            print('for ', self._class_name, ' the pose entered',self._pose_entered)
            return self._n_repeats
        print('the pose_confidence in the thing: ', pose_confidence)
        print('the self.exit threshold: ',self._exit_threshold)
        print(type(pose_confidence))
        print(type(self._exit_threshold))
        
        #If in it's closest points (pose_classification), there are less number of points of the class than the 'exit_threshold'.
        if pose_confidence < self._exit_threshold:
            self._pose_entered=False
            self._n_repeats+=1
            print('counter inc for ',self._class_name)
        
        return self._n_repeats


class ExercisePoseTracker(object):
    def __init__(self, exercise_name, x=30, y=30):
        self._exercise_name=exercise_name
        self.StartRepCounter=None
        self.IntermediateRepCounters=None
        self.EndRepCounter=None
        self.currentState=0
        self.reps=0
        self.has_left_end_pos=0
        self.AllRepCounters=[]   #Used for making it simpler to find the best frame of the user input.
        self._x=x
        self._y=y
    
    def AddRepCounters(self,Counters):  #Counters is a list of the Repetition Counters in order of the exercise being performed.
        if len(Counters)>0:
            self.StartRepCounter=Counters[0]
            if len(Counters)>1:
                self.EndRepCounter=Counters[-1]
                self.IntermediateRepCounters=[rep_counter for idx,rep_counter in enumerate(Counters) if idx!=0 and idx!=len(Counters)-1]
                self.IntermediateRepCounters=self.IntermediateRepCounters if self.IntermediateRepCounters else None
        self.AllRepCounters=[]
        if self.StartRepCounter is not None:
            self.AllRepCounters.append(self.StartRepCounter)
        if self.IntermediateRepCounters is not None:
            for rep_counter in self.IntermediateRepCounters:
                self.AllRepCounters.append(rep_counter)
        if self.EndRepCounter is not None:
            self.AllRepCounters.append(self.EndRepCounter)
        if len(self.AllRepCounters)==0:
            self.AllRepCounters=None
    
    def update_rep_count(self, classification_filtered_data):
        for rep_counter in self.AllRepCounters:
            rep_counter(classification_filtered_data)
        # if self.EndRepCounter is not None and not self.EndRepCounter._pose_entered:
        #     if self.StartRepCounter is not None and self.StartRepCounter._pose_entered:
        #         self.reps += 1
        
        if self.StartRepCounter._n_repeats>=1 and self.EndRepCounter._n_repeats>=1:
            self.reps+=1
            self.StartRepCounter._n_repeats-=1
            self.EndRepCounter._n_repeats-=1
        
        ###### FIRST YOU HAVE TO MAINTAIN THE START REP STATE, LATER YOU HAVE TO CHECK THE FIRST INTERMEDIATE
        ###### STATE AND THEN YOU HAVE TO UPDATE THE REP COUNTER ONLY IF THE PREVIOUS STATES HAVE ALL REP COUNTED
        ###### AS SHOULD HAVE BEEN.
        pass
    
    def write_reps(self, output_frame):
        print('printing at (x,y): ',self._x, ' and ',self._y ,'  for class: ', self._exercise_name)
        cv2.putText(output_frame,f'{self._exercise_name}: {self.reps}',(self._x,self._y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        pass



class BootstrapHelper(object):
    def __init__(self, images_in_folder, images_out_folder, csvs_out_folder):
        self._images_in_folder=images_in_folder     #Is the path of the Training_data_Images.
        self._images_out_folder=images_out_folder   #Is the path to where you wish to place the processed images of the input.
        self._csvs_out_folder=csvs_out_folder       #Is the file name itself.
    
    #Processes the training data, draws landmarks on them and stores in a separate folder. Also, creates the csv file containing the landmarks.
    def bootstrap(self, per_pose_class_limit=None):
        #Opening the csv file to write the landmark values into.
        csv_out_path=self._csvs_out_folder
        with open(csv_out_path, 'w') as csv_out_file:
            #Creating a writer to write landmarks into csv file.
            csv_out_writer=csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            
            #Creating a pose tracker to identify the landmarks in the training data images.
            pose_tracker=mp_pose.Pose()
            
            #Listing the different exercises available in the training dataset.
            diff_exercises_folder_lst=os.listdir(self._images_in_folder)
            for exercise in diff_exercises_folder_lst:
                diff_exercise_positions=os.listdir(os.path.join(self._images_in_folder,exercise))
                for exercise_position in diff_exercise_positions:
                    curr_path=os.path.join(self._images_in_folder,exercise,exercise_position)
                    img_names=os.listdir(curr_path)
                    for img_name in img_names:
                        input_frame=cv2.imread(os.path.join(curr_path,img_name))
                        
                        input_frame=cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                    
                        #Generating the landmark positions on the input frame.
                        result=pose_tracker.process(image=input_frame)
                        pose_landmarks=result.pose_landmarks
                    
                        #Drawing on the output frame and saving it to the output folder.
                        output_frame=input_frame.copy()
                        if pose_landmarks is not None:
                            mp_drawing.draw_landmarks(
                                image=output_frame,
                                landmark_list=pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS
                            )
                        output_frame=cv2.cvtColor(output_frame,cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(self._images_out_folder,exercise,exercise_position,img_name),output_frame)
    #                     print('writing to path: ',os.path.join(self._images_out_folder,curr_posture,img_name))
    #                     plt.imshow(output_frame)
    #                     plt.show()
                    
                        #Saving the landmarks on detection of pose.
                        if pose_landmarks is not None:
                            frame_height, frame_width, _ = output_frame.shape     #Could use input_frame instead of output_frame i think. They're going to have the same shape only.
                            pose_landmarks_percentage=np.array([[lmk.x,lmk.y,lmk.z] for lmk in pose_landmarks.landmark],dtype=np.float32)
                            pose_landmarks=np.array(
                                #We use the 'frame_width' for getting the z-axis position as i think it is normalized using the 'wrist' and so we use frame_width.
                                [[lmk.x*frame_width, lmk.y*frame_height, lmk.z*frame_width] for lmk in pose_landmarks.landmark],
                                dtype=np.float32
                            )
                            assert len(pose_landmarks)==33,'Unexpected landmarks shape: {}'.format(len(pose_landmarks))

                            csv_out_writer.writerow([exercise]+[exercise_position] + [img_name] + pose_landmarks.flatten().astype(str).tolist() + pose_landmarks_percentage.flatten().astype(str).tolist())
#                             csv_out_writer.writerow(pose_landmarks_percentage.flatten().astype(str).tolist())
    
    #Showing all the outliers.
    def analyze_outliers(self, outliers):
        for outlier in outliers:
            image_path=os.path.join(self._images_out_folder, outlier.sample.class_name,outlier.sample.name)
            
            print('Outlier')
            print('Sample path: ', image_path)
            print('Sample class: ', outlier.sample.class_name)
            print('Detected class: ', outlier.detected_class)
            print('All Classes: ', outlier.all_classes)
            
            img=cv2.imread(image_path)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
            
    def remove_outliers(self, outliers):
        for outlier in outliers:
            image_path=os.path.join(self._images_out_folder, outlier.sample.class_name,outlier.sample.name)
            os.remove(image_path)
            
    def print_images_statistics(self, images_folder):
        all_exercises=os.listdir(images_folder)
        for pose_class_name in all_exercises:
            n_images=len([n for n in os.listdir(os.path.join(images_folder,pose_class_name))])
            print(pose_class_name, ': ', n_images)



bootstrap_images_in_folder='/kaggle/input/trainingdata/training_dataset'
bootstrap_images_out_folder='/kaggle/working/training-dataset/Processed_Train_data_Images'
bootstrap_csvs_out_folder='/kaggle/working/training-dataset'

#Creating the required directory to write the processed images of the input images into.
diff_exercises_folder_lst=os.listdir(bootstrap_images_in_folder)
for exercise in diff_exercises_folder_lst:
    exercise_dir_in=os.path.join(bootstrap_images_in_folder,exercise)
    
    #Creating the output folder for the exercise.
    exercise_dir_out=os.path.join(bootstrap_images_out_folder,exercise)
    os.makedirs(exercise_dir_out,exist_ok=True)
    
    #Creating the directory for each of the exercise position as is present in the input training data Images.
    diff_exercise_positions=os.listdir(exercise_dir_in)
    for exercise_position in diff_exercise_positions:
        os.makedirs(os.path.join(exercise_dir_out,exercise_position),exist_ok=True)
    
#Creating the required directory to write the landmarks of the processed input images.
os.makedirs(bootstrap_csvs_out_folder,exist_ok=True)


bootstrap_helper=BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=os.path.join(bootstrap_csvs_out_folder,'landmarks_of_dataset.csv')
)

# bootstrap_helper.print_images_statistics(bootstrap_helper._images_in_folder)
diff_exercises_folder_lst=os.listdir(bootstrap_helper._images_in_folder)
for exercise in diff_exercises_folder_lst:
    bootstrap_helper.print_images_statistics(os.path.join(bootstrap_helper._images_in_folder,exercise))

bootstrap_helper.bootstrap(per_pose_class_limit=None)

# bootstrap_helper.print_images_statistics(bootstrap_helper._images_out_folder)
diff_exercises_folder_lst=os.listdir(bootstrap_helper._images_out_folder)
for exercise in diff_exercises_folder_lst:
    bootstrap_helper.print_images_statistics(os.path.join(bootstrap_helper._images_out_folder,exercise))



#Basic variables required if using a uploaded testing video. WILL HAVE TO BE HANDLED FOR WHEN YOU WANT TO USE A CAMERA.
video_path='/kaggle/input/workoutfitness-video/push-up/push-up_16.mp4'     #Will be '0' if you wish to use real-camera.
# video_path='/kaggle/input/testingvideo-shreyank/testingVideo.mp4'
class_names=list(os.listdir('/kaggle/input/trainingdata/training_dataset'))
out_video_path='pushups-sample-out.mov'                    #Will have to be displayed on the window if using real-time camera.

#Information related to the uploaded input video file.
camera=cv2.VideoCapture(video_path)
input_video_total_frames=camera.get(cv2.CAP_PROP_FRAME_COUNT)
input_video_fps=camera.get(cv2.CAP_PROP_FPS)
input_video_width=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
input_video_height=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Creating a list of Repetition Counters.
repCounterInfoFile=open('/kaggle/input/repcounterinfo2/RepCounterInfo.csv','r')
repCounterInfoReader=csv.reader(repCounterInfoFile,delimiter=',')
ExerciseRepCounters={}
for row in repCounterInfoReader:
    exercise_name=row[0]        #ex: 'PushUps'
    class_name=row[1]           #ex: 'PushUps_UP' or 'PushUps_DOWN'
    enter_threshold=row[2]
    exit_threshold=row[3]
    rep_counter=RepetitionCounter(
        class_name=class_name,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
    )
    if exercise_name in ExerciseRepCounters:
        ExerciseRepCounters[exercise_name].append(rep_counter)
    else:
        ExerciseRepCounters[exercise_name]=[rep_counter]

#Creating the Exercise Trackers for each pose.
ExerciseTrackers={exercise_name:ExercisePoseTracker(exercise_name,((i%3))*300,((i//3)+1)*50) for i,exercise_name in enumerate(ExerciseRepCounters.keys())}
for exercise_name, exercise_tracker in ExerciseTrackers.items():
    exercise_tracker.AddRepCounters(ExerciseRepCounters[exercise_name])
    

#Creating the other instances required.
pose_tracker=mp_pose.Pose()
pose_embedder=PoseEmbedder()
pose_classifier=PoseClassifier(
    pose_samples_file_path='/kaggle/working/training-dataset/landmarks_of_dataset.csv',
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10
)
pose_classification_filter=EMADictSmoothing(
    window_size=10,
    alpha=0.2
)

UniquePoses=[]
input_data_path='/kaggle/input/trainingdata/training_dataset'
for exercise in os.listdir(input_data_path):
    poses=os.listdir(os.path.join(input_data_path,exercise))
    for pose in poses:
        UniquePoses.append(pose)
        
#This dictioanry is used to select the best frame(i.e., the pose of the user in the frame is good) for a pose_name in the 'UniquePoses'
#This dictionary will be used by the 'analysis tab' of the website to compare the posture of the user against the ideal trainer's pose.
BestUserDetectedPoses={pose_name: [None,None,None] for pose_name in UniquePoses}



#Creating a video writer to create the processed video.
out_video=cv2.VideoWriter('/kaggle/working/ProcessedTestingVideo.mov',cv2.VideoWriter_fourcc(*'mp4v'),input_video_fps,(input_video_width,input_video_height))

def show_image(img,figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

frame_idx=0
output_frame=None
with tqdm.tqdm(total=input_video_total_frames, position=0, leave=True) as pbar:
    
    while True:
        success,input_frame=camera.read()
        if not success:
            print('breaking: ')
            break
            
        input_frame=cv2.cvtColor(input_frame,cv2.COLOR_BGR2RGB)
        result=pose_tracker.process(image=input_frame)
        pose_landmarks=result.pose_landmarks
        
        output_frame=input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0),circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0),thickness=4)
            )
            
            frame_height,frame_width, _ = output_frame.shape
            pose_landmarks=np.array([[lmk.x*frame_width,lmk.y*frame_height,lmk.z*frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape==(33,3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            
            pose_classification=pose_classifier(pose_landmarks)
            pose_classification_filtered=pose_classification_filter(pose_classification)
            print('pose_classification filtered of the frame: ',pose_classification_filtered)
            
            ###### IMPLEMENT THE REP COUNTER FUNCTIONALITY AS WELL.
            majority_classified=max(pose_classification_filtered, key=pose_classification_filtered.get)
            print('majority classified as: ',majority_classified)
            
            for exercise_tracker in ExerciseTrackers.values():
                exercise_tracker.update_rep_count(pose_classification_filtered)
                
            if pose_landmarks is not None:
                for exercise_tracker in ExerciseTrackers.values():
                    for rep_counter in exercise_tracker.AllRepCounters:
                        print('for ', rep_counter._class_name, ' the pose_entered: ', rep_counter._pose_entered, ' the pose classification has it', (rep_counter._class_name in pose_classification_filtered))
                        ###### THE BELOW LINE HAS TO BE UNCOMMENTED AND USED INSTEAD. COMMENTED IT BECAUSE RIGHT NOW THE VERY LESS DATA IS THERE.
#                         if rep_counter is not None and rep_counter._pose_entered and (rep_counter._class_name in pose_classification_filtered):
                        if rep_counter is not None and (rep_counter._class_name in pose_classification_filtered):
                            pose_confidence=pose_classification_filtered[rep_counter._class_name]
                            curr_best_pose_confidence=BestUserDetectedPoses[rep_counter._class_name][0]
                            # print('can classify right now')
                            if curr_best_pose_confidence==None or pose_confidence>curr_best_pose_confidence:
                                BestUserDetectedPoses[rep_counter._class_name]=[pose_confidence, output_frame, pose_landmarks]
                                print('helllskdjflksdjflksdjflksdjflkj')
            
        else:
            pose_classification=None
            pose_classification_filtered=None
        
        ###### HAVE TO WRITE THE REP COUNT VALUES ON THE OUTPUT FRAME.
        for exercise_tracker in ExerciseTrackers.values():
            exercise_tracker.write_reps(output_frame)
        out_video.write(cv2.cvtColor(output_frame,cv2.COLOR_RGB2BGR))
        
        if frame_idx%50==0:
            show_image(output_frame)
        frame_idx+=1
        pbar.update()



IdealTrainerLandmarks=pose_classifier.get_closest_point_to_current_point(BestUserDetectedPoses)
def find_file(root_directory, target_filename):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None  # If the file is not found

from PIL import Image
print(IdealTrainerLandmarks)

for class_name, user_details in BestUserDetectedPoses.items():
    frame=user_details[1]
    if frame is None:
        print('continuing')
        continue
        
    ideal_pose_sample=(IdealTrainerLandmarks[class_name][0])
    print(ideal_pose_sample.name)
    ideal_landmarks=ideal_pose_sample.percentage_landmarks
    frame_height,frame_width, _ = frame.shape
    
    #Converting the ideal landmarks from being an numpy.array into the format used by mediapipe's 'landmark_list' for drawing the landmarks.
    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=[
            landmark_pb2.NormalizedLandmark(
                x=lmk[0], y=lmk[1], z=lmk[2]
            ) for lmk in ideal_landmarks
        ])
    
#     actual_landmarks=np.array([[lmk[0]*frame_width,lmk[1]*frame_height,lmk[2]*frame_width] for lmk in ideal_landmarks],dtype=np.float32)
#     print(ideal_landmarks)
    if ideal_landmarks is not None:
        mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),thickness=4)
            )
    
    plt.imshow(frame)
    # plt.show()

    # # img = Image.open(os.path.join(,ideal_pose_sample.name))
    root_dir='/kaggle/working/training-dataset/Processed_Train_data_Images'
    img_name=ideal_pose_sample.name.split('/')[-1]
    img_path= find_file(root_dir, img_name)
    if img_path:
        print(img_path)
        plt.show()
        img=Image.open(img_path)
        plt.imshow(img)
        plt.show()
    # plt.imshow(img_path)
    # plt.show()
    
