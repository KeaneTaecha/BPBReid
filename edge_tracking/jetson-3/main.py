from PTZ_camera import PTZCamera
import cv2
import subprocess
import time
import numpy as np
from ultralytics import YOLO
import os

def main():
    try:
        """Main function to run the camera viewer with PTZ controls and human tracking"""
        # Initialize PTZ Camera
        ptz_camera = PTZCamera()

        cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera View', 640, 480)

        cv2.namedWindow('PTZ Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('PTZ Controls', 400, 150)

        # Create trackbars
        cv2.createTrackbar('Pan Speed (-1 to 1)', 'PTZ Controls', 1, 2, lambda x: None)
        cv2.createTrackbar('Tilt Speed (-1 to 1)', 'PTZ Controls', 1, 2, lambda x: None)
        cv2.createTrackbar('Zoom (100-1000)', 'PTZ Controls', 0, 255, lambda x: None)

        print("\nCamera viewer started")
        print("Controls:")
        print("Pan Speed: 0=Left, 1=Stop, 2=Right")
        print("Tilt Speed: 0=Down, 1=Stop, 2=Up")
        print("Zoom: Slide to adjust zoom level")
        print("Re-ID Threshold: Lower value = easier to re-identify (more false positives)")
        print("Feature Update Rate: Higher value = adapt faster to appearance changes")
        print("- Press 't' to toggle tracking mode")
        print("- Press 'c' to select the closest person as tracking target")
        print("- Press '<- or ->' to select next person as tracking target")
        print("- Press 'd' to toggle debug mode")
        print("- Press 'h' to clear history")
        print("- Press 'r' to reset camera position")
        print("- Press 'q' to quit")

        last_values = {'pan': 1, 'tilt': 1, 'zoom': -1}
        last_update_time = time.time()
        update_interval = 0.1

        # Color palette for IDs (to ensure consistent colors)
        np.random.seed(42)  # For reproducible colors
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

        try:
            while True:
                ret, frame = ptz_camera.cap.read()
                if not ret:
                    continue

                # Perform detection
                boxes, features_list = ptz_camera.person_tracker.detect_persons(frame)

                # Get action classifications
                action_detections = ptz_camera.action_classifier.detect_persons_and_poses(frame)

                # Update tracking
                tracked_persons = ptz_camera.person_tracker.update(frame, boxes, features_list)

                # Create overlay for transparent boxes
                overlay = frame.copy()

                # Draw boxes and IDs
                for person_id, person_data in tracked_persons.items():
                    x, y, w, h = person_data["bbox"]

                    # Find corresponding action for this detection
                    action = "unknown"
                    action_color = (128, 128, 128)  # Default gray
                    for i, action_det in enumerate(action_detections):
                        # Match by bounding box similarity (simple overlap check)
                        ax, ay, aw, ah = action_det['bbox']
                        if abs(x - ax) < 50 and abs(y - ay) < 50:  # Rough matching
                            action = action_det['action']
                            action_color = ptz_camera.action_classifier.get_action_color(action)
                            break

                    # Draw filled transparent rectangle on overlay
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), action_color, -1)

                    id_color = (int(colors[person_id % 100][0]),
                            int(colors[person_id % 100][1]),
                            int(colors[person_id % 100][2]))

                    # Draw box with ID
                    cv2.rectangle(frame, (x, y), (x + w, y + h), id_color, 6)
                    cv2.putText(frame, f"ID: {person_id} - {action}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

                    # Highlight tracked person
                    if ptz_camera.tracking_enabled and ptz_camera.target_id == person_id:
                        cv2.rectangle(frame, (x-5, y-5), (x + w + 5, y + h + 5), (0, 255, 255), 2)

                # Blend overlay with original frame (30% transparency)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw pose keypoints from action classifier
                for detection in action_detections:
                    if detection['keypoints'] and len(detection['keypoints']) >= 51:
                        ptz_camera.action_classifier.draw_keypoints(frame, detection['keypoints'])

                # Handle tracking
                if ptz_camera.tracking_enabled and ptz_camera.target_id is not None:
                    if ptz_camera.target_id in tracked_persons:
                        # Target found - reset search state if it was active
                        if hasattr(ptz_camera, 'current_search_phase'):
                            print("Target found, resetting search state")
                            ptz_camera.current_search_phase = "initial"
                            ptz_camera.target_lost_time = None
                            ptz_camera.set_pan_speed(0)
                            ptz_camera.set_tilt_speed(0)

                        target_person = tracked_persons[ptz_camera.target_id]
                        target_center = np.array([
                            target_person["centroid"][0],
                            target_person["centroid"][1]
                        ])

                        # Calculate and apply movement
                        pan, tilt = ptz_camera.calculate_movement(target_center)

                        # Only apply movement if not in search mode
                        ptz_camera.set_pan_speed(pan)
                        print(f'Panning {pan}')

                        ptz_camera.set_tilt_speed(tilt)
                        print(f'Tilting {tilt}')

                    else:
                        print(f"Target ID {ptz_camera.target_id} not found in tracked persons")
                        ptz_camera.pan_search_for_target()

                # Add tracking status to frame
                status_text = f"Tracking: {'ON' if ptz_camera.tracking_enabled else 'OFF'}"
                if ptz_camera.tracking_enabled and ptz_camera.target_id is not None:
                    status_text += f" - Target ID: {ptz_camera.target_id}"
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Update FPS calculation
                ptz_camera.curr_frame_time = time.time()

                # Calculate and display FPS
                if ptz_camera.prev_frame_time > 0:
                    ptz_camera.fps = 1 / (ptz_camera.curr_frame_time - ptz_camera.prev_frame_time)
                ptz_camera.prev_frame_time = ptz_camera.curr_frame_time

                # Draw FPS at top right corner
                fps_text = f"FPS: {ptz_camera.fps:.1f}"
                fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                fps_x = frame.shape[1] - fps_text_size[0] - 20  # 20 pixels from right edge
                cv2.putText(frame, fps_text, (fps_x, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Camera View', frame)

                # Update controls at fixed interval
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # Get current control values
                    pan = cv2.getTrackbarPos('Pan Speed (-1 to 1)', 'PTZ Controls') - 1
                    tilt = cv2.getTrackbarPos('Tilt Speed (-1 to 1)', 'PTZ Controls') - 1
                    zoom = cv2.getTrackbarPos('Zoom (100-1000)', 'PTZ Controls')

                    # Only update if values changed
                    if pan != last_values['pan']:
                        ptz_camera.set_pan_speed(pan)
                        last_values['pan'] = pan

                    if tilt != last_values['tilt']:
                        ptz_camera.set_tilt_speed(tilt)
                        last_values['tilt'] = tilt

                    if zoom != last_values['zoom']:
                        ptz_camera.set_zoom(zoom)
                        last_values['zoom'] = zoom

                    last_update_time = current_time

                # Get key press without masking to 0xFF to preserve the full key code
                key = cv2.waitKey(1)

                # Use masked version for regular keys
                key_masked = key & 0xFF

                if key_masked == ord('q'):
                    break
                elif key_masked == ord('r'):
                    # Reset position and tracking
                    print("Resetting Position")
                    ptz_camera.tracking_enabled = False
                    ptz_camera.target_id = None
                    cv2.setTrackbarPos('Pan Speed (-1 to 1)', 'PTZ Controls', 1)
                    cv2.setTrackbarPos('Tilt Speed (-1 to 1)', 'PTZ Controls', 1)
                    cv2.setTrackbarPos('Zoom (100-1000)', 'PTZ Controls', 0)
                elif key_masked == ord('t'):
                    # Toggle tracking
                    ptz_camera.tracking_enabled = not ptz_camera.tracking_enabled
                    print(f"Tracking {'enabled' if ptz_camera.tracking_enabled else 'disabled'}")
                elif key_masked == ord('c'):
                    # Select closest person to center as target
                    if tracked_persons:
                        center = np.array([320, 240])
                        closest_id = min(tracked_persons.keys(),
                                        key=lambda id: np.linalg.norm(
                                            np.array(tracked_persons[id]["centroid"]) - center))
                        ptz_camera.target_id = closest_id
                        ptz_camera.tracking_enabled = True
                        print(f"Now tracking person ID: {ptz_camera.target_id}")

                # For right arrow key (next person)
                elif key == 83:  # Right arrow in OpenCV
                    # Select next person as target
                    if tracked_persons:
                        person_ids = list(tracked_persons.keys())
                        if ptz_camera.target_id in person_ids:
                            idx = person_ids.index(ptz_camera.target_id)
                            idx = (idx + 1) % len(person_ids)
                            ptz_camera.target_id = person_ids[idx]
                        else:
                            ptz_camera.target_id = person_ids[0]

                        # Reset search state when manually selecting a target
                        ptz_camera.current_search_phase = "initial"
                        ptz_camera.search_pan_complete = False
                        ptz_camera.target_lost_time = None

                        # Force stop panning movement
                        ptz_camera.set_pan_speed(0)
                        ptz_camera.set_tilt_speed(0)

                        ptz_camera.tracking_enabled = True
                        print(f"Now tracking person ID: {ptz_camera.target_id}")

                elif key == 81:  # Left arrow in OpenCV
                    # Select previous person as target
                    if tracked_persons:
                        person_ids = list(tracked_persons.keys())
                        if ptz_camera.target_id in person_ids:
                            idx = person_ids.index(ptz_camera.target_id)
                            idx = (idx - 1) % len(person_ids)
                            ptz_camera.target_id = person_ids[idx]
                        else:
                            ptz_camera.target_id = person_ids[-1]

                        # Reset search state when manually selecting a target
                        ptz_camera.search_started = False
                        ptz_camera.search_pan_complete = False
                        ptz_camera.target_lost_time = None

                        # Force stop panning movement
                        ptz_camera.set_pan_speed(0)
                        ptz_camera.set_tilt_speed(0)

                        ptz_camera.tracking_enabled = True
                        print(f"Now tracking person ID: {ptz_camera.target_id}")

        finally:
            # Save all currently tracked persons to history before closing
            if hasattr(ptz_camera.person_tracker, 'persons') and ptz_camera.person_tracker.persons:
                print("Saving active persons to history before exit...")

                # Load current history
                history = ptz_camera.person_tracker.load_history()

                # Add all active persons to history
                for person_id, person_data in ptz_camera.person_tracker.persons.items():
                    history[person_id] = {
                        "features": person_data["features"],
                        "last_seen": time.time(),
                        "bbox": person_data["bbox"]
                    }
                    print(f"Saved person ID {person_id} to history")

                # Save updated history
                ptz_camera.person_tracker.save_history(history)
                print(f"Total {len(ptz_camera.person_tracker.persons)} active persons saved to history")

            # Make sure to stop any movement before closing
            ptz_camera.set_pan_speed(0)
            ptz_camera.set_tilt_speed(0)
            ptz_camera.cap.release()
            cv2.destroyAllWindows()
            for i in range(4):
                cv2.waitKey(1)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()