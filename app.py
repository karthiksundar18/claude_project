import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
import os
import pickle
import time

class ActionDetector:
    def __init__(self, actions_list=None, sequence_length=30, detection_threshold=0.5):  # Lowered threshold from 0.7 to 0.5
        """
        Initialize the Action Detector with MediaPipe Pose and LSTM model
        
        Args:
            actions_list: List of actions to detect
            sequence_length: Number of frames to use for prediction
            detection_threshold: Confidence threshold for detection
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Set default actions if not provided
        if actions_list is None:
            self.actions = ['wave', 'thumbs_up', 'running', 'jumping', 'idle']
        else:
            self.actions = actions_list
            
        self.sequence_length = sequence_length
        self.sequence = deque(maxlen=sequence_length)
        self.detection_threshold = detection_threshold
        self.model = None
        self.scaler = None
        
        # Create model directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
    def extract_keypoints(self, results):
        """
        Extract keypoints from MediaPipe pose results
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            numpy array of keypoints (x, y, visibility) flattened
        """
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
            return pose
        else:
            # Return zeros if no pose detected
            return np.zeros(33 * 3)
            
    def preprocess_keypoints(self, keypoints):
        """
        Preprocess keypoints for model input
        
        Args:
            keypoints: Extracted keypoints
            
        Returns:
            Preprocessed keypoints
        """
        # Apply scaling if scaler exists
        if self.scaler is not None:
            return self.scaler.transform([keypoints])[0]
        return keypoints
            
    def build_model(self):
        """
        Build LSTM model for action recognition
        """
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 33*3)),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(len(self.actions), activation='softmax')
        ])
        
        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        self.model = model
        return model
        
    def train_model(self, X_train, y_train, epochs=50, batch_size=16, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        return history
        
    def save_model(self, model_name='action_model'):
        """
        Save the trained model and scaler
        
        Args:
            model_name: Name to save the model as
        """
        if self.model is not None:
            self.model.save(f'models/{model_name}.h5')
            print(f"Model saved as models/{model_name}.h5")
            
        if self.scaler is not None:
            with open(f'models/{model_name}_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved as models/{model_name}_scaler.pkl")
            
    def load_model(self, model_name='action_model'):
        """
        Load a trained model and scaler
        
        Args:
            model_name: Name of the model to load
        """
        try:
            self.model = tf.keras.models.load_model(f'models/{model_name}.h5')
            print(f"Model loaded from models/{model_name}.h5")
            
            # Try to load scaler if it exists
            try:
                with open(f'models/{model_name}_scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Scaler loaded from models/{model_name}_scaler.pkl")
            except:
                print("No scaler found, using raw keypoints")
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def collect_training_data(self, action, num_sequences=30, sequence_length=30):
        """
        Collect training data for an action
        
        Args:
            action: Action name to collect data for
            num_sequences: Number of sequences to collect
            sequence_length: Length of each sequence
            
        Returns:
            List of collected sequences
        """
        cap = cv2.VideoCapture(0)
        collected_sequences = []
        
        # Loop through sequences
        for sequence in range(num_sequences):
            # Temporary storage for the current sequence
            temp_sequence = []
            
            # Countdown before starting
            for countdown in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Flip the frame horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Draw countdown text
                cv2.putText(frame, f"Starting collection in {countdown}", (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"Collecting {action} - Sequence {sequence+1}/{num_sequences}", 
                           (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('Collect Training Data', frame)
                cv2.waitKey(1000)
            
            # Start collecting frames for the sequence
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Flip the frame horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect pose
                results = self.pose.process(image)
                
                # Draw pose landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS)
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                temp_sequence.append(keypoints)
                
                # Display collection progress
                cv2.putText(frame, f"Collecting {action} - Sequence {sequence+1}/{num_sequences}", 
                           (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Frame {frame_num+1}/{sequence_length}", 
                           (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.imshow('Collect Training Data', frame)
                cv2.waitKey(1)
            
            # Add the completed sequence to our collection
            if len(temp_sequence) == sequence_length:
                collected_sequences.append(temp_sequence)
        
        cap.release()
        cv2.destroyAllWindows()
        return collected_sequences
    
    def collect_data_for_all_actions(self, num_sequences=30):
        """
        Collect training data for all actions
        
        Args:
            num_sequences: Number of sequences to collect per action
            
        Returns:
            X: Features
            y: Labels
        """
        action_sequences = []
        action_labels = []
        
        for action_idx, action in enumerate(self.actions):
            print(f"Collecting data for action: {action}")
            sequences = self.collect_training_data(action, num_sequences, self.sequence_length)
            
            for sequence in sequences:
                action_sequences.append(sequence)
                # One-hot encode the labels
                label = np.zeros(len(self.actions))
                label[action_idx] = 1
                action_labels.append(label)
                
        X = np.array(action_sequences)
        y = np.array(action_labels)
        
        return X, y
    
    def predict_action(self):
        """
        Predict action from the current sequence
        
        Returns:
            Predicted action and confidence
        """ 
        # In the predict_action method, modify the stabilization approach:
        detection_buffer = []  # Make this a class variable instead of local
        
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None, 0
            
        if len(self.sequence) < self.sequence_length:
            return None, 0
            
        # Prepare the sequence for prediction
        sequence_array = np.array(list(self.sequence))
        
        # Make prediction
        res = self.model.predict(np.expand_dims(sequence_array, axis=0), verbose=0)[0]  # Added verbose=0 to reduce console output
        predicted_class_idx = np.argmax(res)
        confidence = res[predicted_class_idx]
        
        if confidence > self.detection_threshold:
            return self.actions[predicted_class_idx], confidence
        else:
            return None, confidence
    
    def run_detection(self):
        """
        Run real-time action detection using webcam feed
        """
        # Check if model exists and load it
        if self.model is None:
            print("Attempting to load model...")
            success = self.load_model()
            if not success:
                print("No model available. Checking for uploaded dataset to train a new model...")
                try:
                    # Try to load your uploaded dataset
                    X = np.load('models/X_data.npy')
                    y = np.load('models/y_data.npy')
                    print(f"Found uploaded dataset: {X.shape} features and {y.shape} labels")
                    
                    # Build and train model on uploaded data
                    print("Training model on uploaded dataset...")
                    self.build_model()
                    self.train_model(X, y, epochs=20)
                    self.save_model()
                    print("Model trained and saved successfully!")
                except Exception as e:
                    print(f"Error loading or training with uploaded data: {e}")
                    print("Please make sure your dataset is in the correct format and location.")
                    return
        
        print("Starting webcam capture for action detection...")
        # Start webcam
        # In the run_detection method, try adding this right after you create the cap:
# Clear any previous state
        self.sequence.clear()
        detection_buffer = []
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera. Please check your webcam connection.")
            return
        
        # Variables for FPS calculation and debugging
        prev_time = 0
        curr_time = 0
        frame_count = 0
        detection_buffer = [] # For storing recent detections
        
        print("Press 'q' to quit, 'd' to toggle debug mode")
        debug_mode = True  # Start with debug mode on to help troubleshooting
        
        # Create a named window that can be resized
        cv2.namedWindow('Real-time Action Detection', cv2.WINDOW_NORMAL)
        
        # Add visual indicator for sequence buffer status
        sequence_indicator = np.zeros((30, self.sequence_length * 10, 3), dtype=np.uint8)
        frame_counter = 0
        reset_interval = 100  # Reset buffers every N frames
        detection_buffer = []  # Make this a local variable in the method
        if len(detection_buffer) >= 3:  # Increase this threshold
            # Count occurrences of each action in the buffer
            action_counts = {}
            for det_action, _ in detection_buffer[-10:]:  # Look at more recent detections
                if det_action:
                    action_counts[det_action] = action_counts.get(det_action, 0) + 1
            
            # Find the most common action with higher minimum count
            most_common_action = None
            max_count = 0
            for act, count in action_counts.items():
                if count > max_count and count >= 3:  # Require at least 3 occurrences
                    max_count = count
                    most_common_action = act
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            frame_count += 1
            # Inside the main loop, add:
            frame_counter += 1
            if frame_counter % reset_interval == 0:
                # Periodically reset sequence buffer to avoid getting stuck
                self.sequence.clear()
                detection_buffer.clear()
                if debug_mode:
                    print(f"Auto-reset buffers at frame {frame_counter}")
            
            # Create debug view
            if debug_mode:
                debug_frame = np.zeros((400, 600, 3), dtype=np.uint8)  # Increased size for more debug info
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose
            results = self.pose.process(image)
            
            # Current prediction info
            action = None
            confidence = 0
            
            # Extract keypoints
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS)
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                
                # Preprocess keypoints if needed
                processed_keypoints = self.preprocess_keypoints(keypoints)
                
                # Add to the sequence
                self.sequence.append(processed_keypoints)
                
                # Update visual indicator for sequence buffer
                sequence_indicator = np.zeros((30, self.sequence_length * 10, 3), dtype=np.uint8)
                for i in range(len(self.sequence)):
                    sequence_indicator[:, i*10:(i+1)*10, 1] = 255  # Green for filled slots
                
                # Make prediction when we have enough frames
                if len(self.sequence) == self.sequence_length:
                    # Debug info about sequence shape
                    if debug_mode:
                        seq_array = np.array(list(self.sequence))
                        cv2.putText(debug_frame, f"Sequence shape: {seq_array.shape}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    action, confidence = self.predict_action()
                    
                    # Store prediction in buffer
                    detection_buffer.append((action, confidence))
                    if len(detection_buffer) > 15:  # Increased buffer size for more stable predictions
                        detection_buffer.pop(0)
            
            # Show sequence buffer status on frame
            frame[frame.shape[0]-40:frame.shape[0]-10, 10:10+self.sequence_length*10, :] = sequence_indicator
            cv2.putText(frame, "Buffer:", (10, frame.shape[0]-45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display prediction with improved stabilization
            if len(detection_buffer) >= 2:  # Reduced from 3 to 2 for faster response
                # Count occurrences of each action in the buffer
                action_counts = {}
                for det_action, _ in detection_buffer[-5:]:  # Look at the last 5 detections
                    if det_action:
                        action_counts[det_action] = action_counts.get(det_action, 0) + 1
                
                # Find the most common action
                most_common_action = None
                max_count = 0
                for act, count in action_counts.items():
                    if count > max_count:
                        max_count = count
                        most_common_action = act
                
                # Only display if the action appears multiple times
                if most_common_action and max_count >= 2:
                    # Get the latest confidence for this action
                    latest_confidence = next((conf for act, conf in reversed(detection_buffer) if act == most_common_action), 0)
                    
                    # Display with green box for better visibility
                    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
                    cv2.putText(frame, f"ACTION:", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    cv2.putText(frame, f"{most_common_action.upper()}", 
                            (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {latest_confidence:.2f}", 
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (frame.shape[1] - 120, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Show debug information
            if debug_mode:
                # Display sequence fill level
                fill_percentage = len(self.sequence) / self.sequence_length * 100
                cv2.putText(debug_frame, f"Sequence buffer: {len(self.sequence)}/{self.sequence_length} ({fill_percentage:.0f}%)", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Model info
                cv2.putText(debug_frame, f"Model loaded: {'Yes' if self.model is not None else 'No'}", 
                        (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_frame, f"Detection threshold: {self.detection_threshold}", 
                        (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_frame, f"Actions: {', '.join(self.actions)}", 
                        (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display recent predictions
                cv2.putText(debug_frame, "Recent predictions:", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for i, (pred_action, pred_conf) in enumerate(detection_buffer[-10:]):
                    color = (0, 255, 0) if pred_action else (0, 0, 255)
                    action_text = pred_action if pred_action else "none"
                    cv2.putText(debug_frame, f"{i+1}: {action_text} ({pred_conf:.2f})", 
                            (10, 120 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Show keypoint stats if available
                if results.pose_landmarks:
                    # Get key landmarks for action recognition
                    left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    cv2.putText(debug_frame, f"L wrist (x,y): ({left_wrist.x:.2f}, {left_wrist.y:.2f})", 
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(debug_frame, f"R wrist (x,y): ({right_wrist.x:.2f}, {right_wrist.y:.2f})", 
                            (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show arm positions which can help debug specific actions
                    left_arm_angle = np.arctan2(left_wrist.y - left_shoulder.y, left_wrist.x - left_shoulder.x) * 180 / np.pi
                    right_arm_angle = np.arctan2(right_wrist.y - right_shoulder.y, right_wrist.x - right_shoulder.x) * 180 / np.pi
                    
                    cv2.putText(debug_frame, f"L arm angle: {left_arm_angle:.1f}°", 
                            (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(debug_frame, f"R arm angle: {right_arm_angle:.1f}°", 
                            (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show debug frame
                cv2.imshow('Debug Info', debug_frame)
                    
            # Show the frame
            cv2.imshow('Real-time Action Detection', frame)
            
            # Process key presses
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                if not debug_mode:
                    cv2.destroyWindow('Debug Info')
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            # Add this key handler to your key processing section
            elif key == ord('r'):
                # Manual reset of all buffers
                self.sequence.clear()
                detection_buffer.clear()
                print("Manually reset all detection buffers")
            elif key == ord('c'):
                # Clear the sequence buffer
                self.sequence.clear()
                detection_buffer.clear()
                print("Cleared detection buffers")
                
        cap.release()
        cv2.destroyAllWindows()


def create_dummy_model(detector):
    """
    Create a dummy model for testing
    """
    print("Creating dummy model with realistic action patterns...")
    num_samples = 200  # More samples for better training
    
    # Create more realistic action patterns
    sequences = []
    labels = []
    
    for action_idx, action in enumerate(detector.actions):
        print(f"Generating dummy data for action: {action}")
        for i in range(num_samples // len(detector.actions)):
            # Create base sequence filled with zeros
            seq = np.zeros((detector.sequence_length, 33*3))
            
            # Add action-specific patterns
            if action == 'wave':
                # Create wave pattern for right arm
                for frame in range(detector.sequence_length):
                    # Right wrist movement pattern
                    wrist_x = 0.7 + 0.1 * np.sin(frame * 0.4)  # Oscillating x position
                    wrist_y = 0.3 + 0.2 * np.sin(frame * 0.4)  # Oscillating y position
                    
                    # Right wrist coordinates (x, y, visibility)
                    seq[frame, 16*3:16*3+3] = [wrist_x, wrist_y, 1.0]  # Right wrist landmark
            
            elif action == 'thumbs_up':
                # Create thumbs up pattern
                for frame in range(detector.sequence_length):
                    # Right wrist stays relatively still
                    wrist_x = 0.6
                    wrist_y = 0.4
                    
                    # Right wrist coordinates (x, y, visibility)
                    seq[frame, 16*3:16*3+3] = [wrist_x, wrist_y, 1.0]  # Right wrist landmark
            
            elif action == 'running':
                # Create running pattern
                for frame in range(detector.sequence_length):
                    # Oscillating legs and arms
                    cycle = frame * 0.2
                    
                    # Right knee (oscillates)
                    seq[frame, 26*3:26*3+3] = [0.5, 0.7 + 0.1 * np.sin(cycle), 1.0]
                    
                    # Left knee (opposite phase)
                    seq[frame, 25*3:25*3+3] = [0.5, 0.7 + 0.1 * np.sin(cycle + np.pi), 1.0]
            
            elif action == 'jumping':
                # Create jumping pattern
                for frame in range(detector.sequence_length):
                    # Vertical movement of the whole body
                    y_offset = 0.1 * np.sin(frame * 0.2)
                    
                    # Hip positions
                    seq[frame, 24*3+1] = 0.6 + y_offset  # Right hip y
                    seq[frame, 23*3+1] = 0.6 + y_offset  # Left hip y
                    
                    # Knee positions
                    seq[frame, 26*3+1] = 0.75 + y_offset  # Right knee y
                    seq[frame, 25*3+1] = 0.75 + y_offset  # Left knee y
            
            elif action == 'idle':
                # Create idle pattern (slight random movement)
                for frame in range(detector.sequence_length):
                    # Small random movements
                    for j in range(33):
                        seq[frame, j*3:j*3+2] = [0.5 + np.random.normal(0, 0.01), 
                                               0.5 + np.random.normal(0, 0.01)]
                        seq[frame, j*3+2] = 1.0  # Visibility
            
            # Add some random noise to all sequences for realism
            seq += np.random.normal(0, 0.01, seq.shape)
            
            sequences.append(seq)
            
            # One-hot encode the label
            label = np.zeros(len(detector.actions))
            label[action_idx] = 1
            labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # Save the generated dummy data
    np.save('models/X_data_dummy.npy', X)
    np.save('models/y_data_dummy.npy', y)
    
    print(f"Generated dummy data: {X.shape} features and {y.shape} labels")
    
    # Train the model with the generated data
    detector.build_model()
    detector.train_model(X, y, epochs=20, batch_size=32)
    detector.save_model('action_model_dummy')
    
    return detector


def main():
    # Actions to detect
    actions = ['wave', 'thumbs_up', 'running', 'jumping', 'idle']
    
    # Create detector with shorter sequence length for faster response
    detector = ActionDetector(actions_list=actions, sequence_length=30, detection_threshold=0.6)
    
    while True:
        print("\nHuman Action Detection System")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run real-time detection")
        print("4. Create dummy model (for testing)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            X, y = detector.collect_data_for_all_actions()
            
            # Save the collected data
            np.save('models/X_data.npy', X)
            np.save('models/y_data.npy', y)
            print(f"Collected data saved: {X.shape} features and {y.shape} labels")
            
        elif choice == '2':
            # Check if we have saved data
            try:
                X = np.load('models/X_data.npy')
                y = np.load('models/y_data.npy')
                print(f"Loaded data: {X.shape} features and {y.shape} labels")
                
                detector.build_model()
                history = detector.train_model(X, y, epochs=50)  # Increased epochs for better training
                detector.save_model()
                
                print("Training complete! Model saved.")
                
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Please collect training data first.")
                
        elif choice == '3':
            detector.run_detection()
            
        elif choice == '4':
            print("Creating dummy model for testing...")
            detector = create_dummy_model(detector)
            print("Dummy model created and saved!")
            
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()